# -*- coding: utf-8 -*-
""" rhoEst: implementation of the 2nd multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has rank_i = 1, but when a collision occurs, rank_i is sampled from a uniform distribution on [1, .., M] where M is the number of player.

- Note: this is fully decentralized: each child player does NOT need to know the number of players, but require the horizon.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.5"

import numpy.random as rn

from .rhoRand import oneRhoRand, rhoRand


# --- threshold function Xsi(n, k)

def default_threshold(horizon, nbPlayersEstimate):
    """Function Xsi(n, k) used as a threshold in rhoEst.

    - 0 if nbPlayersEstimate is 0,
    - 1 if nbPlayersEstimate is 1,
    - any function such that: Xsi(n, k) = omega(log n) for all k > 1. (cf. http://mathworld.wolfram.com/Little-OmegaNotation.html). I chose n, as sqrt(n) and n**0.1 were too small (the nbPlayersEstimate was always growing too fast).
    """
    if nbPlayersEstimate == 0:
        return 0
    elif nbPlayersEstimate == 1:
        return 1
    else:
        # return horizon ** 0.5
        # return horizon ** 0.1
        return horizon


# --- Class oneRhoEst, for children

class oneRhoEst(oneRhoRand):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new random rank is sampled after observing a collision,
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy,
    - The rhoEst policy is used to keep an estimate on the total number of players.
    """

    def __init__(self, horizon, threshold, *args, **kwargs):
        super(oneRhoEst, self).__init__(*args, **kwargs)
        # Parameters
        del self.nbPlayers
        self.horizon = horizon
        self.threshold = threshold
        # Internal variables
        self.nbPlayersEstimate = 1  # Optimistic: start by assuming it is alone!
        self.rank = None
        self.collisionCount = 0

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<{}, {}{}>".format(self.playerId + 1, r"$\rho^{\mathrm{Est}}$", self.mother._players[self.playerId], "rank:{}".format(self.rank) if self.rank is not None else "")

    def startGame(self):
        super(oneRhoEst, self).startGame()
        self.rank = 1  # Start with a rank = 1: assume she is alone.

    def handleCollision(self, arm):
        # First, pick a new random rank for this
        self.rank = 1 + rn.randint(self.nbPlayersEstimate)  # New random rank
        # print(" - A oneRhoEst player {} saw a collision, so she had to select a new random rank : {} ...".format(self, self.rank))  # DEBUG
        # Then, estimate the current ranking of the arms
        order = self.estimatedOrder()
        # And try to see if the arm on which we are encountering a collision is one of the Uhat best
        if order[arm] >= self.nbPlayersEstimate:
            # if arm is one of the best nbPlayersEstimate arms:
            self.collisionCount += 1
            # print("This arm {} was estimated as one of the Uhat = {} best arm, so we increase the collision count to {}.".format(arm, self.nbPlayersEstimate, self.collisionCount))  # DEBUG
        # And finally, compare the collision count with the current threshold
        threshold = self.threshold(self.horizon, self.nbPlayersEstimate)
        if self.collisionCount > threshold:
            self.nbPlayersEstimate += 1
            # print("The collision count {} was larger than the threshold {:.3g} se we reinitiliaze the collision count, and increase the nbPlayersEstimate to {}.".format(self.collisionCount, threshold, self.nbPlayersEstimate))  # DEBUG
            self.collisionCount = 0


    def choice(self):
        # Note: here we could do another randomization step, but it would just weaken the algorithm, cf. rhoRandRand
        result = super(oneRhoEst, self).choiceWithRank(self.rank)
        # print(" - A oneRhoEst player {} had to choose an arm among the best from rank {}, her choice was : {} ...".format(self, self.rank, result))  # DEBUG
        return result


# --- Class rhoEst

class rhoEst(rhoRand):
    """ rhoEst: implementation of the 2nd multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms, horizon, threshold=default_threshold, lower=0., amplitude=1., *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - horizon: needed for the estimate of nb of users.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = rhoEst(nbPlayers, UCB, nbArms, horizon)

        - To get a list of usable players, use s.children.
        - Warning: s._players is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRand class has to be > 0."
        self.nbPlayers = nbPlayers
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            fakeNbArms = None
            self.children[playerId] = oneRhoEst(horizon, threshold, fakeNbArms, self, playerId)
        self.nbArms = nbArms

    def __str__(self):
        return "rhoEst({} x {})".format(self.nbPlayers, str(self._players[0]))
