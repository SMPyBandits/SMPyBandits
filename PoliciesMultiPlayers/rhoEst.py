# -*- coding: utf-8 -*-
""" rhoEst: implementation of the 2nd multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has rank_i = 1, but when a collision occurs, rank_i is sampled from a uniform distribution on [1, .., M] where Mhat_i is the estimated number of player by player i,
- The procedure to estimate Mhat_i is not so simple, but basically everyone starts with Mhat_i = 1, and when colliding Mhat_i += 1, for some time (with a complicated threshold).

- Note: this is fully decentralized: each child player does NOT need to know the number of players, but require the horizon.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.5"

from math import log, sqrt
import numpy.random as rn

from .rhoRand import oneRhoRand, rhoRand


# --- threshold function Xsi(n, k)

def default_threshold(horizon, nbPlayersEstimate):
    """Function Xsi(T, k) used as a threshold in rhoEst.

    - 0 if nbPlayersEstimate is 0,
    - 1 if nbPlayersEstimate is 1,
    - any function such that: Xsi(T, k) = omega(log T) for all k > 1. (cf. http://mathworld.wolfram.com/Little-OmegaNotation.html). I chose T, as sqrt(T) and T**0.1 were too small (the nbPlayersEstimate was always growing too fast).
    """
    if nbPlayersEstimate == 0:
        return 0
    elif nbPlayersEstimate == 1:
        return 1
    else:
        # return horizon ** 0.5
        # return horizon ** 0.1
        return horizon


def threshold_on_t(t, nbPlayersEstimate):
    """Function Xsi(t, k) used as a threshold in rhoEst.

    - 0 if nbPlayersEstimate is 0,
    - 1 if nbPlayersEstimate is 1,
    - My heuristic is to use a function of t (current time) and not T (horizon).
    """
    if nbPlayersEstimate == 0:
        return 0
    elif nbPlayersEstimate == 1:
        return 1
    else:
        # return log(t)
        # return float(t) ** 0.7
        # return float(t) ** 0.5
        # return float(t) ** 0.1
        return t


# --- Class oneRhoEst, for children

class oneRhoEst(oneRhoRand):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new random rank is sampled after observing a collision,
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy,
    - The rhoEst policy is used to keep an estimate on the total number of players, Mhat_i.
    - The procedure to estimate Mhat_i is not so simple, but basically everyone starts with Mhat_i = 1, and when colliding Mhat_i += 1, for some time (with a complicated threshold).
    """

    def __init__(self, horizon, threshold, *args, **kwargs):
        super(oneRhoEst, self).__init__(*args, **kwargs)
        # Parameters
        del self.maxRank  # <-- make SURE that maxRank is NOT used by the policy!
        self.horizon = horizon
        self.threshold = threshold
        # Internal variables
        self.nbPlayersEstimate = 1  # Optimistic: start by assuming it is alone!
        self.rank = None
        self.collisionCount = 0
        self.timeSinceLastCollision = 0
        self.t = 0

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<{}[{}{}]>".format(self.playerId + 1, r"$\rho^{\mathrm{Est}}$", self.mother._players[self.playerId], ", rank:{}".format(self.rank) if self.rank is not None else "")

    def startGame(self):
        """Start game."""
        super(oneRhoEst, self).startGame()
        self.nbPlayersEstimate = 1  # Osptimistic: start by assuming it is alone!
        self.collisionCount = 0
        self.timeSinceLastCollision = 0
        self.t = 0
        self.rank = 1  # Start with a rank = 1: assume she is alone.

    def handleCollision(self, arm, reward=None):
        """Select a new rank, and maybe update nbPlayersEstimate."""
        # rhoRand UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: rhoRand UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneRhoRand, self).getReward(arm, reward)

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
        # threshold = self.threshold(self.horizon, self.nbPlayersEstimate)
        threshold = self.threshold(self.timeSinceLastCollision, self.nbPlayersEstimate)

        if self.collisionCount > threshold:
            self.nbPlayersEstimate += 1
            # print("The collision count {} was larger than the threshold {:.3g} se we reinitiliaze the collision count, and increase the nbPlayersEstimate to {}.".format(self.collisionCount, threshold, self.nbPlayersEstimate))  # DEBUG
            self.collisionCount = 0
        # Finally, reinitiliaze timeSinceLastCollision
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

    def __init__(self, nbPlayers, playerAlgo, nbArms, horizon, threshold=threshold_on_t, lower=0., amplitude=1., *args, **kwargs):
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
        self.nbArms = nbArms
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            fakemaxRank = None
            self.children[playerId] = oneRhoEst(horizon, threshold, fakemaxRank, self, playerId)

    def __str__(self):
        return "rhoEst({} x {})".format(self.nbPlayers, str(self._players[0]))
