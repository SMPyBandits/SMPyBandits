# -*- coding: utf-8 -*-
""" rhoRand: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has a random rank_i from 1 to M, and when a collision occurs, rank_i is sampled from a uniform distribution on [1, .., M] where M is the number of player.


.. note:: This is not fully decentralized: as each child player needs to know the (fixed) number of players.

"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.5"

import numpy.random as rn

from .BaseMPPolicy import BaseMPPolicy
from .ChildPointer import ChildPointer


# --- Class oneRhoRand, for children

class oneRhoRand(ChildPointer):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new random rank is sampled after observing a collision,
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy.
    """

    def __init__(self, maxRank, *args, **kwargs):
        super(oneRhoRand, self).__init__(*args, **kwargs)
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.rank = None  #: Current rank, starting to 1 by default

    def __str__(self):   # Better to recompute it automatically
        # return r"#{}<{}[{}{}]>".format(self.playerId + 1, r"$\rho^{\mathrm{Rand}}$", self.mother._players[self.playerId], ", rank:{}".format(self.rank) if self.rank is not None else "")
        return r"#{}<{}-{}{}>".format(self.playerId + 1, r"RhoRand", self.mother._players[self.playerId], ", rank:{}".format(self.rank) if self.rank is not None else "")

    def startGame(self):
        """Start game."""
        super(oneRhoRand, self).startGame()
        self.rank = 1 + rn.randint(self.maxRank)  # XXX Start with a random rank, safer to avoid first collisions.

    def handleCollision(self, arm, reward=None):
        """Get a new fully random rank, and give reward to the algorithm if not None."""
        # rhoRand UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: rhoRand UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneRhoRand, self).getReward(arm, reward)
        self.rank = 1 + rn.randint(self.maxRank)  # New random rank
        # print(" - A oneRhoRand player {} saw a collision, so she had to select a new random rank : {} ...".format(self, self.rank))  # DEBUG

    def choice(self):
        """Chose with the actual rank."""
        # Note: here we could do another randomization step, but it would just weaken the algorithm, cf. rhoRandRand
        result = super(oneRhoRand, self).choiceWithRank(self.rank)
        # print(" - A oneRhoRand player {} had to choose an arm among the best from rank {}, her choice was : {} ...".format(self, self.rank, result))  # DEBUG
        return result


# --- Class rhoRand

class rhoRand(BaseMPPolicy):
    """ rhoRand: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 maxRank=None, orthogonalRanks=False,
                 lower=0., amplitude=1., *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the rhoRand child (default to nbPlayers, but for instance if there is 2 × rhoRand[UCB] + 2 × rhoRand[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = rhoRand(nbPlayers, Thompson, nbArms)

        - To get a list of usable players, use ``s.children``.
        - Warning: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRand class has to be > 0."
        if maxRank is None:
            maxRank = nbPlayers
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.nbPlayers = nbPlayers  #: Number of players
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRhoRand(maxRank, self, playerId)

    def __str__(self):
        return "rhoRand({} x {}{})".format(self.nbPlayers, str(self._players[0]))
