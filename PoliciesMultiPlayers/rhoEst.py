# -*- coding: utf-8 -*-
r""" rhoEst: implementation of the 2nd multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has a random rank_i from 1 to M, and when a collision occurs, rank_i is sampled from a uniform distribution on :math:`[1, \dots, \hat{M}_i(t)]` where :math:`\hat{M}_i(t)` is the current estimated number of player by player i,
- The procedure to estimate :math:`\hat{M}_i(t)` is not so simple, but basically everyone starts with :math:`\hat{M}_i(0) = 1`, and when colliding :math:`\hat{M}_i(t+1) = \hat{M}_i(t) + 1`, for some time (with a complicated threshold).

- My choice for the threshold function, see :func:`threshold_on_t`, does not need the horizon either, and uses :math:`t` instead.

.. note:: This is **fully decentralized**: each child player does NOT need to know the number of players and does NOT require the horizon :math:`T`.

.. note:: For a more generic approach, see the wrapper defined in :class:`EstimateM.EstimateM`.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.8"

import numpy as np
import numpy.random as rn

from .rhoRand import oneRhoRand, rhoRand
from .EstimateM import threshold_on_t_with_horizon, threshold_on_t_doubling_trick, threshold_on_t


# --- Class oneRhoEst, for children

class oneRhoEst(oneRhoRand):
    r""" Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new random rank is sampled after observing a collision,
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy,
    - The rhoEst policy is used to keep an estimate on the total number of players, :math:`\hat{M}_i(t)`.
    - The procedure to estimate :math:`\hat{M}_i(t)` is not so simple, but basically everyone starts with :math:`\hat{M}_i(0) = 1`, and when colliding :math:`\hat{M}_i(t+1) = \hat{M}_i(t) + 1`, for some time (with a complicated threshold).
    """

    def __init__(self, threshold, *args, **kwargs):
        if 'horizon' in kwargs:
            self.horizon = kwargs['horizon']
            del kwargs['horizon']
        else:
            self.horizon = None
        super(oneRhoEst, self).__init__(*args, **kwargs)
        # Parameters
        if hasattr(self, 'maxRank'):
            self.maxRank = 1  # <-- make SURE that maxRank is NOT used by the policy!
        self.threshold = threshold  #: Threshold function
        # Internal variables
        self.nbPlayersEstimate = 1  #: Number of players. Optimistic: start by assuming it is alone!
        self.rank = None  #: Current rank, starting to 1
        self.collisionCount = np.zeros(self.nbArms, dtype=int)  #: Count collisions on each arm, since last increase of nbPlayersEstimate
        self.timeSinceLastCollision = 0  #: Time since last collision. Don't remember why I thought using this could be useful... But it's not!
        self.t = 0  #: Internal time

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<RhoEst{}-{}{}{}>".format(self.playerId + 1, "Plus" if self.horizon else "", self.mother._players[self.playerId], "(rank:{})".format(self.rank) if self.rank is not None else "", "($T={}$)".format(self.horizon) if self.horizon else "")

    def startGame(self):
        """Start game."""
        super(oneRhoEst, self).startGame()
        self.rank = 1  # Start with a rank = 1: assume she is alone.
        # Est part
        self.nbPlayersEstimate = 1  # Optimistic: start by assuming it is alone!
        self.collisionCount.fill(0)
        self.timeSinceLastCollision = 0
        self.t = 0

    def handleCollision(self, arm, reward=None):
        """Select a new rank, and maybe update nbPlayersEstimate."""
        # rhoRand UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: rhoRand UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneRhoEst, self).getReward(arm, reward)

        # First, pick a new random rank
        self.rank = 1 + rn.randint(self.nbPlayersEstimate)  # New random rank
        # print("\n - A oneRhoEst player {} saw a collision on {}, new random rank : {} ...".format(self, arm, self.rank))  # DEBUG

        # we can be smart, and stop all this as soon as M = K !
        if self.nbPlayersEstimate < self.nbArms:
            self.collisionCount[arm] += 1
            # print("\n - A oneRhoEst player {} saw a collision on {}, since last update of nbPlayersEstimate = {} it is the {} th collision on that arm {}...".format(self, arm, self.nbPlayersEstimate, self.collisionCount[arm], arm))  # DEBUG

            # Then, estimate the current ranking of the arms and the set of the M best arms
            currentBest = self.estimatedBestArms(self.nbPlayersEstimate)
            # print("Current estimation of the {} best arms is {} ...".format(self.nbPlayersEstimate, currentBest))  # DEBUG

            collisionCount_on_currentBest = np.sum(self.collisionCount[currentBest])
            # print("Current count of collision on the {} best arms is {} ...".format(self.nbPlayersEstimate, collisionCount_on_currentBest))  # DEBUG

            # And finally, compare the collision count with the current threshold
            threshold = self.threshold(self.t, self.nbPlayersEstimate, self.horizon)
            # print("Using timeSinceLastCollision = {}, and t = {}, threshold = {:.3g} ...".format(self.timeSinceLastCollision, self.t, threshold))

            if collisionCount_on_currentBest > threshold:
                self.nbPlayersEstimate = min(1 + self.nbPlayersEstimate, self.nbArms)
                # print("The collision count {} was larger than the threshold {:.3g} se we restart the collision count, and increase the nbPlayersEstimate to {}.".format(collisionCount_on_currentBest, threshold, self.nbPlayersEstimate))  # DEBUG
                self.collisionCount.fill(0)
            # Finally, restart timeSinceLastCollision
            self.timeSinceLastCollision = 0

    def getReward(self, arm, reward):
        """One transmission without collision."""
        self.t += 1
        # Obtaining a reward, even 0, means no collision on that arm for this time
        # So, first, we count one more step without collision
        self.timeSinceLastCollision += 1
        # Then use the reward for the arm learning algorithm
        return super(oneRhoEst, self).getReward(arm, reward)


# --- Class rhoEst

class rhoEst(rhoRand):
    """ rhoEst: implementation of the 2nd multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 threshold=threshold_on_t_doubling_trick, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - threshold: the threshold function to use, see :func:`EstimateM.threshold_on_t_with_horizon`, :func:`EstimateM.threshold_on_t_doubling_trick` or :func:`EstimateM.threshold_on_t` above.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = rhoEst(nbPlayers, UCB, nbArms, threshold=threshold_on_t)

        - To get a list of usable players, use ``s.children``.
        - Warning: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRand class has to be > 0."
        self.nbPlayers = nbPlayers  #: Number of players
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        fake_maxRank = None
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRhoEst(threshold, fake_maxRank, self, playerId)

    def __str__(self):
        return "rhoEst({} x {})".format(self.nbPlayers, str(self._players[0]))


# --- Class rhoEstPlus

class rhoEstPlus(rhoRand):
    """ rhoEstPlus: implementation of the 2nd multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo, horizon,
                 lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - horizon: need to know the horizon :math:`T`.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = rhoEstPlus(nbPlayers, UCB, nbArms, horizon)

        - To get a list of usable players, use ``s.children``.
        - Warning: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRand class has to be > 0."
        self.nbPlayers = nbPlayers  #: Number of players
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        fake_maxRank = None
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRhoEst(threshold_on_t_with_horizon, fake_maxRank, self, playerId, horizon=horizon)

    def __str__(self):
        return "rhoEstPlus({} x {})".format(self.nbPlayers, str(self._players[0]))
