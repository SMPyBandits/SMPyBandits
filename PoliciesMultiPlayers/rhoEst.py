# -*- coding: utf-8 -*-
r""" rhoEst: implementation of the 2nd multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has a random rank_i from 1 to M, and when a collision occurs, rank_i is sampled from a uniform distribution on :math:`[1, \dots, \hat{M}_i(t)]` where :math:`\hat{M}_i(t)` is the current estimated number of player by player i,
- The procedure to estimate :math:`\hat{M}_i(t)` is not so simple, but basically everyone starts with :math:`\hat{M}_i(0) = 1`, and when colliding :math:`\hat{M}_i(t+1) = \hat{M}_i(t) + 1`, for some time (with a complicated threshold).

- My choice for the threshold function, see :func:`threshold_on_t`, does not need the horizon either, and uses :math:`t` instead.

.. note:: This is fully decentralized: each child player does NOT need to know the number of players and does NOT require the horizon :math:`T`.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.6"

from math import log, sqrt
import numpy.random as rn

from .rhoRand import oneRhoRand, rhoRand


# --- threshold function xi(n, k)


def threshold_on_t_with_horizon(horizon, t, nbPlayersEstimate):
    r""" Function :math:`\xi(T, k)` used as a threshold in :class:`rhoEstPlus`.

    - `0` if `nbPlayersEstimate` is `0`,
    - `1` if `nbPlayersEstimate` is `1`,
    - any function such that: :math:`\xi(T, k) = \omega(\log T)` for all `k > 1`. (cf. http://mathworld.wolfram.com/Little-OmegaNotation.html). I chose :math:`T`, as :math:`\sqrt(T)` and :math:`T^{0.1}` were too small (the `nbPlayersEstimate` was always growing too fast).

    .. warning:: It requires the horizon :math:`T`.
    """
    if nbPlayersEstimate <= 1:
        return nbPlayersEstimate
    else:
        return log(1 + horizon) ** 2
        # return float(horizon) ** 0.7
        # return float(horizon) ** 0.5
        # return float(horizon) ** 0.1
        # return horizon


def make_threshold_on_t_from_horizon(horizon):
    """Use the horizon to create ONCE a function of t and nbPlayersEstimate, for :class:`rhoEstPlus."""
    def threshold_on_t(t, nbPlayersEstimate):
        return threshold_on_t_with_horizon(horizon, t, nbPlayersEstimate)
    return threshold_on_t


def threshold_on_t(t, nbPlayersEstimate):
    r""" Function :math:`\xi(t, k)` used as a threshold in :class:`rhoEst`.

    - `0` if `nbPlayersEstimate` is `0`,
    - `1` if `nbPlayersEstimate` is `1`,
    - My heuristic is to use a function of :math:`t` (current time) and not :math:`T` (horizon).
    - The choice which seemed to perform the best in practice was :math:`\xi(t, k) = t`.
    """
    if nbPlayersEstimate <= 1:
        return nbPlayersEstimate
    else:
        return log(1 + t) ** 2
        # return float(t) ** 0.7
        # return float(t) ** 0.5
        # return float(t) ** 0.1
        # return t


# --- Class oneRhoEst, for children

class oneRhoEst(oneRhoRand):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new random rank is sampled after observing a collision,
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy,
    - The rhoEst policy is used to keep an estimate on the total number of players, :math:`\hat{M}_i(t)`.
    - The procedure to estimate :math:`\hat{M}_i(t)` is not so simple, but basically everyone starts with :math:`\hat{M}_i(0) = 1`, and when colliding :math:`\hat{M}_i(t+1) = \hat{M}_i(t) + 1`, for some time (with a complicated threshold).
    """

    def __init__(self, threshold, *args, **kwargs):
        super(oneRhoEst, self).__init__(*args, **kwargs)
        # Parameters
        if hasattr(self, 'maxRank'):
            self.maxRank = 1  # <-- make SURE that maxRank is NOT used by the policy!
        self.threshold = threshold  #: Threshold function
        # Internal variables
        self.nbPlayersEstimate = 1  #: Number of players. Optimistic: start by assuming it is alone!
        self.rank = None  #: Current rank, starting to 1
        self.collisionCount = 0  #: Count collision since last increase of nbPlayersEstimate
        self.timeSinceLastCollision = 0  #: Time since last collision
        self.t = 0  #: Internal time

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<{}[{}{}]>".format(self.playerId + 1, r"$\rho^{\mathrm{Est}%s$" % ("" if self.threshold.__name__ == 'threshold_on_t' else "Plus"), self.mother._players[self.playerId], ", rank:{}".format(self.rank) if self.rank is not None else "")

    def startGame(self):
        """Start game."""
        super(oneRhoEst, self).startGame()
        self.nbPlayersEstimate = 1  # Optimistic: start by assuming it is alone!
        self.collisionCount = 0
        self.timeSinceLastCollision = 0
        self.t = 0
        self.rank = 1  # Start with a rank = 1: assume she is alone.

    def handleCollision(self, arm, reward=None):
        """Select a new rank, and maybe update nbPlayersEstimate."""
        # rhoRand UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: rhoRand UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneRhoEst, self).getReward(arm, reward)

        # First, pick a new random rank for this
        self.rank = 1 + rn.randint(self.nbPlayersEstimate)  # New random rank
        # print("\n - A oneRhoEst player {} saw a collision on {}, new random rank : {} ...".format(self, arm, self.rank))  # DEBUG

        # Then, estimate the current ranking of the arms
        order = self.estimatedOrder()

        # And try to see if the arm on which we are encountering a collision is one of the Uhat best
        if order[arm] >= self.nbPlayersEstimate:  # if arm is one of the best nbPlayersEstimate arms:
            self.collisionCount += 1
            # print("This arm {} was estimated as one of the Uhat = {} best arm, so we increase the collision count to {}.".format(arm, self.nbPlayersEstimate, self.collisionCount))  # DEBUG

        # And finally, compare the collision count with the current threshold
        threshold = self.threshold(self.timeSinceLastCollision, self.nbPlayersEstimate)

        if self.collisionCount > threshold:
            self.nbPlayersEstimate += 1
            # print("The collision count {} was larger than the threshold {:.3g} se we restart the collision count, and increase the nbPlayersEstimate to {}.".format(self.collisionCount, threshold, self.nbPlayersEstimate))  # DEBUG
            self.collisionCount = 0
        # Finally, restart timeSinceLastCollision
        self.timeSinceLastCollision = 0

    def getReward(self, arm, reward):
        """One transmission without collision"""
        # Obtaining a reward, even 0, means no collision on that arm for this time
        # So, first, we count one more step without collision
        self.timeSinceLastCollision += 1
        # Then use the reward for the arm learning algorithm
        return super(oneRhoEst, self).getReward(arm, reward)

    def choice(self):
        """Chose with the actual rank."""
        self.t += 1
        # Note: here we could do another randomization step, but it would just weaken the algorithm, cf. rhoRandRand
        chosenArm = super(oneRhoEst, self).choiceWithRank(self.rank)
        # print(" - A oneRhoEst player {} chose {} among the bests from rank {}...".format(self, chosenArm, self.rank))  # DEBUG
        return chosenArm


# --- Class rhoEst

class rhoEst(rhoRand):
    """ rhoEst: implementation of the 2nd multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms,
                 threshold=threshold_on_t, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - threshold: the threshold function to use, see :func:`default_threshold` or :func:`threshold_on_t` above.
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

    def __init__(self, nbPlayers, playerAlgo, nbArms, horizon,
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
        threshold = make_threshold_on_t_from_horizon(horizon)
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRhoEst(threshold, fake_maxRank, self, playerId)

    def __str__(self):
        return "rhoEstPlus({} x {})".format(self.nbPlayers, str(self._players[0]))
