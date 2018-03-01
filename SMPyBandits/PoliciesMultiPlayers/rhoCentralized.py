# -*- coding: utf-8 -*-
""" rhoCentralized: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- Every player has rank_i = i + 1, as given by the base station.


.. note:: This is not fully decentralized: as each child player needs to know the (fixed) number of players, and an initial orthogonal configuration.

.. warning:: This policy is NOT efficient at ALL! Don't use it! It seems a smart idea, but it's not.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy.random as rn

from .BaseMPPolicy import BaseMPPolicy
from .ChildPointer import ChildPointer


# --- Class oneRhoCentralized, for children

class oneRhoCentralized(ChildPointer):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - The player does not aim at the best arm, but at the rank-th best arm, based on her index policy.
    """

    def __init__(self, maxRank, rank=None, *args, **kwargs):
        super(oneRhoCentralized, self).__init__(*args, **kwargs)
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        assert rank is None or 1 <= rank <= maxRank, "Error: the 'rank' parameter = {} for oneRhoCentralized was not correct: only possible values are None or an integer 1 <= rank <= maxRank = {}.".format(rank, maxRank)  # DEBUG
        self.keep_the_same_rank = rank is not None  #: If True, the rank is kept constant during the game, as if it was given by the Base Station
        self.rank = int(rank) if self.keep_the_same_rank else None  #: Current rank, starting to 1 by default, or 'rank' if given as an argument

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<{}[{}{}]>".format(self.playerId + 1, r"$\rho^{\mathrm{Centralized}}$", self.mother._players[self.playerId], ", {}rank:{}".format("fixed " if self.keep_the_same_rank else "", self.rank) if self.rank is not None else "")

    def startGame(self):
        """Start game."""
        super(oneRhoCentralized, self).startGame()
        if not self.keep_the_same_rank:
            self.rank = 1  # Start with a rank = 1: assume she is alone.

    def handleCollision(self, arm, reward=None):
        """Get a new fully random rank, and give reward to the algorithm if not None."""
        # rhoCentralized UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: rhoCentralized UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneRhoCentralized, self).getReward(arm, reward)
        if not self.keep_the_same_rank:
            self.rank = 1 + rn.randint(self.maxRank)  # New random rank
            # print(" - A oneRhoCentralized player {} saw a collision, so she had to select a new random rank : {} ...".format(self, self.rank))  # DEBUG

    def choice(self):
        """Chose with the actual rank."""
        result = super(oneRhoCentralized, self).choiceWithRank(self.rank)
        # print(" - A oneRhoCentralized player {} had to choose an arm among the best from rank {}, her choice was : {} ...".format(self, self.rank, result))  # DEBUG
        return result


# --- Class rhoCentralized

class rhoCentralized(BaseMPPolicy):
    """ rhoCentralized: implementation of a variant of the multi-player rhoRand policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 maxRank=None, orthogonalRanks=True,
                 lower=0., amplitude=1., *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the rhoCentralized child (default to nbPlayers, but for instance if there is 2 × rhoCentralized[UCB] + 2 × rhoCentralized[klUCB], maxRank should be 4 not 2).
        - orthogonalRanks: if True, orthogonal ranks 1..M are directly affected to the players 1..M.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = rhoCentralized(nbPlayers, Thompson, nbArms)

        - To get a list of usable players, use ``s.children``.
        - Warning: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoCentralized class has to be > 0."
        if maxRank is None:
            maxRank = nbPlayers
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.nbPlayers = nbPlayers  #: Number of players
        self.orthogonalRanks = orthogonalRanks  #: Using orthogonal ranks from starting
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            if orthogonalRanks:
                self.children[playerId] = oneRhoCentralized(maxRank, self, playerId, rank=playerId + 1)
            else:
                self.children[playerId] = oneRhoCentralized(maxRank, self, playerId)

    def __str__(self):
        return "rhoCentralized({} x {}{})".format(self.nbPlayers, str(self._players[0]), ", orthogonal ranks" if self.orthogonalRanks else "")
