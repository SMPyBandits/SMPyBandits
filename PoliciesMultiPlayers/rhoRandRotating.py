# -*- coding: utf-8 -*-
""" rhoRandRotating: implementation of a variant of the multi-player policy rhoRand from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has rank_i = 1, but when a collision occurs, rank_i is sampled from a uniform distribution on [1, .., M] where M is the number of player.
- The *only* difference with rhoRand is that at every time step, the rank is updated by 1, and cycles in [1, .., M] iteratively.

- Note: this is not fully decentralized: as each child player needs to know the (fixed) number of players.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy.random as rn

from .rhoRand import oneRhoRand, rhoRand


# --- Class oneRhoRandRotating, for children

class oneRhoRandRotating(oneRhoRand):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new random rank is sampled after observing a collision,
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy.
    """

    def __init__(self, maxRank, *args, rank=None, **kwargs):
        super(oneRhoRandRotating, self).__init__(maxRank, *args, **kwargs)
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        assert rank is None or 1 <= rank <= maxRank, "Error: the 'rank' parameter = {} for oneRhoRand was not correct: only possible values are None or an integer 1 <= rank <= maxRank = {}.".format(rank, maxRank)  # DEBUG
        self.keep_the_same_rank = rank is not None  #: If True, the rank is kept constant during the game, as if it was given by the Base Station
        self.rank = int(rank) if self.keep_the_same_rank else None  #: Current rank, starting to 1 by default, or 'rank' if given as an argument

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<{}[{}{}]>".format(self.playerId + 1, r"$\rho^{\mathrm{RandRotating}}$", self.mother._players[self.playerId], ", rank:{}".format(self.rank) if self.rank is not None else "")

    def startGame(self):
        """Start game."""
        super(oneRhoRandRotating, self).startGame()
        self.rank = 1  # Start with a rank = 1: assume she is alone.

    def handleCollision(self, arm, reward=None):
        """ Get a new fully random rank, and give reward to the algorithm if not None."""
        # rhoRandRotating UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: rhoRandRotating UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneRhoRandRotating, self).getReward(arm, reward)
        self.rank = 1 + rn.randint(self.maxRank)  # New random rank
        # print(" - A oneRhoRandRotating player {} saw a collision, so she had to select a new random rank : {} ...".format(self, self.rank))  # DEBUG

    def choice(self):
        r"""Chose with the new rank, then update the rank:

        .. math:: \mathrm{rank}_j(t+1) := \mathrm{rank}_j(t) + 1 \;\mathrm{mod}\; M.
        """
        # Note: here we could do another randomization step, but it would just weaken the algorithm, cf. rhoRandRand
        result = super(oneRhoRandRotating, self).choiceWithRank(self.rank)
        # print(" - A oneRhoRand player {} had to choose an arm among the best from rank {}, her choice was : {} ...".format(self, self.rank, result))  # DEBUG
        self.rank = (self.rank % self.maxRank) + 1
        # print(" - A oneRhoRand player {} has a new rank {} ...".format(self, self.rank))  # DEBUG
        return result


# --- Class rhoRandRotating

class rhoRandRotating(rhoRand):
    """ rhoRandRotating: implementation of a variant of the multi-player policy rhoRand from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms,
                 maxRank=None, orthogonalRanks=False,
                 lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the rhoRandRotating child (default to nbPlayers, but for instance if there is 2 × rhoRandRotating[UCB] + 2 × rhoRandRotating[klUCB], maxRank should be 4 not 2).
        - orthogonalRanks: if True, orthogonal ranks 1..M are directly affected to the players 1..M.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = rhoRandRotating(nbPlayers, Thompson, nbArms)

        - To get a list of usable players, use s.children.
        - Warning: s._players is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRandRotating class has to be > 0."
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
                self.children[playerId] = oneRhoRandRotating(maxRank, self, playerId, rank=playerId + 1)
            else:
                self.children[playerId] = oneRhoRandRotating(maxRank, self, playerId)

    def __str__(self):
        return "rhoRandRotating({} x {}{})".format(self.nbPlayers, str(self._players[0]), ", orthogonal ranks" if self.orthogonalRanks else "")
