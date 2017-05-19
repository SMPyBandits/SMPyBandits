# -*- coding: utf-8 -*-
""" rhoRandALOHA: implementation of a variant of the multi-player policy rhoRand from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has rank_i = 1, but when a collision occurs, rank_i is sampled from a uniform distribution on [1, .., M] where M is the number of player.
- The *only* difference with rhoRand is that when colliding, users have a small chance of keeping the same rank, following a Bernoulli experiment: with probability = p0, it stays, with proba 1 - p0 it leaves.
- There is also a variant, like in MEGA (ALOHA-like protocol), the proba change after time: p(t+1) = alpha p(t) + (1-alpha)

- Note: this is not fully decentralized: as each child player needs to know the (fixed) number of players.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy.random as rn

from .rhoRand import oneRhoRand, rhoRand

#: Default value for P0, ideally, it should be 1/M the number of player
P0 = 0.0

#: Default value for ALPHA_P0, FIXME I have no idea what the best possible choise ca be!
ALPHA_P0 = 0.98


def with_proba(eps):
    r"""Bernoulli test, with probability :math:`\varepsilon`, return `True`, and with probability :math:`1 - \varepsilon`, return `False`."""
    return rn.random() < eps


# --- Class oneRhoRandSticky, for children

class oneRhoRandSticky(oneRhoRand):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new random rank is sampled after observing a collision,
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy.
    """

    def __init__(self, maxRank, p0, alpha_p0, *args, **kwargs):
        super(oneRhoRandSticky, self).__init__(maxRank, *args, **kwargs)
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        # p0
        assert 0 <= p0 <= 1, "Error: for oneRhoRandSticky the parameter 'p0' should be in [0, 1], but it was {}.".format(p0)  # DEBUG
        self.p0 = p0  #: Initial probability, should not be modified.
        self.p = p0  #: Current probability of staying with the current rank after a collision. If 0, then it is like the initial rhoRand policy.
        # alpha
        assert 0 < alpha_p0 <= 1, "Error: parameter 'alpha_p0' for a ALOHA player should be in (0, 1]."
        self.alpha_p0 = alpha_p0  #: Parameter alpha for the recurrence equation for probability p(t)
        # rank
        self.rank = None  #: Current rank, starting to 1

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<{}[{}{}{}]>".format(self.playerId + 1, r"$\rho^{\mathrm{Rand}}$", self.mother._players[self.playerId], ", rank:{}".format(self.rank) if self.rank is not None else "", r", $p_0:{:.3g}, \alpha:{:.3g}$".format(self.p0, self.alpha_p0) if self.p0 > 0 else "")

    def startGame(self):
        """Start game."""
        super(oneRhoRandSticky, self).startGame()
        self.rank = 1  # Start with a rank = 1: assume she is alone.
        self.p = self.p0  # Reinitialize the probability

    def handleCollision(self, arm, reward=None):
        """ Get a new fully random rank, and give reward to the algorithm if not None."""
        # rhoRandALOHA UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: rhoRandALOHA UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneRhoRandSticky, self).getReward(arm, reward)
        # With probability p, keep the rank
        if with_proba(1. - self.p):  # With probability 1-p, change the rank
            self.rank = 1 + rn.randint(self.maxRank)  # New random rank
            # print(" - A oneRhoRandSticky player {} saw a collision, so she had to select a new random rank : {} ...".format(self, self.rank))  # DEBUG
            print(" - A oneRhoRandSticky player {} saw a reward, so she reinitialized her probability p from {:.3g} to {:.3g}...".format(self, self.p, self.p0))  # DEBUG
            self.p = self.p0  # Reinitialize the proba p

    def getReward(self, arm, reward):
        """ Pass the call to self.mother._getReward_one(playerId, arm, reward) with the player's ID number.

        - Additionally, if the current rank was good enough to not bring any collision during the last p0 time steps, the player "sits" on that rank.
        """
        super(oneRhoRandSticky, self).getReward(arm, reward)
        old_p = self.p
        self.p = self.p * self.alpha_p0 + (1 - self.Alpha_p0)  # Update proba p
        print(" - A oneRhoRandSticky player {} saw a reward, so she updated her probability p from {:.3g} to {:.3g}...".format(self, old_p, self.p))  # DEBUG


# --- Class rhoRandALOHA

class rhoRandALOHA(rhoRand):
    """ rhoRandALOHA: implementation of a variant of the multi-player policy rhoRand from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms,
                 p0=None, alpha_p0=ALPHA_P0,
                 lower=0., amplitude=1., maxRank=None,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - p0: given to the oneRhoRandSticky objects (see above).
        - alpha_p0: given to the oneRhoRandSticky objects (see above).
        - maxRank: maximum rank allowed by the rhoRandALOHA child (default to nbPlayers, but for instance if there is 2 × rhoRandALOHA[UCB] + 2 × rhoRandALOHA[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = rhoRandALOHA(nbPlayers, Thompson, nbArms, p0)

        - To get a list of usable players, use s.children.
        - Warning: s._players is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRandALOHA class has to be > 0."
        if maxRank is None:
            maxRank = nbPlayers
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        if p0 is None:  # If not given, use 1/M by default
            p0 = 1. / nbPlayers
        self.p0 = p0  #: Initial value for p, current probability of staying with the current rank after a collision
        self.alpha_p0 = alpha_p0  #: Parameter alpha for the recurrence equation for probability p(t)
        self.nbPlayers = nbPlayers  #: Number of players
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRhoRandSticky(maxRank, p0, self, playerId)

    def __str__(self):
        return "rhoRandALOHA({} x {}{})".format(self.nbPlayers, str(self._players[0]), r"$p_0:{:.3g}, \alpha:{:.3g}$".format(self.p0, self.alpha_p0) if self.p0 > 0 else "")
