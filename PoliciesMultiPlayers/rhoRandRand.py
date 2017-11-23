# -*- coding: utf-8 -*-
""" rhoRandRand: implementation of a variant of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the k-th best arm, for k again uniformly drawn from [1, ..., rank_i],
- At first, every player has a random rank_i from 1 to M, and when a collision occurs, rank_i is sampled from a uniform distribution on [1, ..., M] where M is the number of player.


.. note:: This algorithm is *intended* to be stupid! It does not work at all!!

.. note:: This is not fully decentralized: as each child player needs to know the (fixed) number of players.

"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.5"

import numpy.random as rn

from .BaseMPPolicy import BaseMPPolicy
from .ChildPointer import ChildPointer


# --- Class oneRhoRandRand, for children

class oneRhoRandRand(ChildPointer):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new random rank is sampled after observing a collision,
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy.
    """

    def __init__(self, maxRank, *args, **kwargs):
        super(oneRhoRandRand, self).__init__(*args, **kwargs)
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.rank = None  #: Current rank, starting to 1

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<{}[{}{}]>".format(self.playerId + 1, r"$\rho^{\mathrm{Rand}\mathrm{Rand}}$", self.mother._players[self.playerId], ", rank:{}".format(self.rank) if self.rank is not None else "")

    def startGame(self):
        """Start game."""
        super(oneRhoRandRand, self).startGame()
        self.rank = 1 + rn.randint(self.maxRank)  # XXX Start with a random rank, safer to avoid first collisions.

    def handleCollision(self, arm, reward=None):
        """Get a new rank."""
        self.rank = 1 + rn.randint(self.maxRank)  # New random rank
        # print(" - A oneRhoRandRand player {} saw a collision, so she had to select a new random rank : {} ...".format(self, self.rank))  # DEBUG

    def choice(self):
        """Chose with a RANDOM rank."""
        # We added another randomization step, but it would just weaken the algorithm!
        # I could select again a random rank to aim, uniformly from [1, ..., rank]
        result = super(oneRhoRandRand, self).choiceWithRank(1 + rn.randint(self.rank))  # That's rhoRandRand
        # result = super(oneRhoRandRand, self).choiceWithRank(self.rank)              # And that was rhoRand
        # print(" - A oneRhoRandRand player {} had to choose an arm among the best from rank {}, her choice was : {} ...".format(self, self.rank, result))  # DEBUG
        return result


# --- Class rhoRandRand

class rhoRandRand(BaseMPPolicy):
    """ rhoRandRand: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 lower=0., amplitude=1., maxRank=None, *args, **kwargs):  # Named argument to give them in any order
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the rhoRand child (default to nbPlayers, but for instance if there is 2 × rhoRand[UCB] + 2 × rhoRand[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = rhoRandRand(nbPlayers, Thompson, nbArms)

        - To get a list of usable players, use ``s.children``.
        - Warning: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRandRand class has to be > 0."
        if maxRank is None:
            maxRank = nbPlayers
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.nbPlayers = nbPlayers  #: Number of players
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRhoRandRand(maxRank, self, playerId)
        self.nbArms = nbArms  #: Number of arms

    def __str__(self):
        return "rhoRandRand({} x {})".format(self.nbPlayers, str(self._players[0]))
