# -*- coding: utf-8 -*-
""" rhoLearned: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/), using a learning algorithm instead of a random exploration for choosing the rank.

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has rank_i = 1, but when a collision occurs, rank_i is given by a second learning algorithm, playing on arms = ranks from [1, .., M], where M is the number of player.
- If rankSelection = Uniform, this is like rhoRand, but if it is a smarter policy, it *might* be better! Warning: no theoretical guarantees exist!

- Note: this is not fully decentralized: as each child player needs to know the (fixed) number of players.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.6"

try:
    from sys import path
    path.insert(0, '..')
    from Policies import Uniform
except ImportError:
    print("Warning: ../Policies/Uniform.py was not imported correctly...")  # DEBUG
    # ... Just reimplement it here manually, if not found in ../Policies/Uniform.py
    from random import randint
    class Uniform():
        def __init__(self, nbArms, lower=0., amplitude=1.):
            self.nbArms = nbArms

        def startGame(self):
            pass

        def getReward(self, arm, reward):
            pass

        def choice(self):
            return randint(0, self.nbArms - 1)

import numpy as np

from .rhoRand import oneRhoRand, rhoRand


# --- Class oneRhoLearned, for children

class oneRhoLearned(oneRhoRand):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new rank is sampled after observing a collision, from the rankSelection algorithm.
    - When no collision is observed on a arm, a small reward is given to the rank used for this play, in order to learn the best ranks with rankSelection.
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy.
    """

    def __init__(self, nbPlayers, rankSelectionAlgo, *args, **kwargs):
        super(oneRhoLearned, self).__init__(nbPlayers, *args, **kwargs)
        self.rankSelection = rankSelectionAlgo(nbPlayers)  # FIXME I should give it more arguments?
        self.nbPlayers = nbPlayers
        self.rank = None
        # Keep in memory how many times a rank could be used while giving no collision
        self.timesUntilCollision = np.zeros(nbPlayers, dtype=int)

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<{}, {}{}, ranks ~ {}>".format(self.playerId + 1, r"$\rho^{\mathrm{Learned}}$", self.mother._players[self.playerId], ", rank:{}".format(self.rank) if self.rank is not None else "", self.rankSelection)

    def startGame(self):
        self.rankSelection.startGame()
        super(oneRhoLearned, self).startGame()
        self.rank = 1  # Start with a rank = 1: assume she is alone.
        self.timesUntilCollision.fill(0)

    def getReward(self, arm, reward):
        # Obtaining a reward, even 0, means no collision on that arm for this time
        # So, first, we count one more step for this rank
        self.timesUntilCollision[self.rank - 1] += 1
        # First give a reward to the rank selection learning algorithm (== collision avoidance)
        self.rankSelection.getReward(arm, 1)
        # Then use the reward for the arm learning algorithm
        return super(oneRhoLearned, self).getReward(arm, reward)

    def handleCollision(self, arm):
        # First, reset the time until collisions for that rank
        self.timesUntilCollision[self.rank - 1] = 0
        # Then, use the rankSelection algorithm to select a new rank
        self.rank = 1 + self.rankSelection.choice()
        print(" - A oneRhoLearned player {} saw a collision, so she had to select a new rank from her algorithm {} : {} ...".format(self, self.rankSelection, self.rank))  # DEBUG


# --- Class rhoRand

class rhoLearned(rhoRand):
    """ rhoLearned: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/), using a learning algorithm instead of a random exploration for choosing the rank.
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms, rankSelectionAlgo=Uniform,
                 lower=0., amplitude=1., *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - rankSelectionAlgo: algorithm to use for selecting the ranks.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = rhoLearned(nbPlayers, Thompson, nbArms, Uniform)  # Exactly rhoRand!
        >>> s = rhoLearned(nbPlayers, Thompson, nbArms, UCB)      # Possibly better than rhoRand!

        - To get a list of usable players, use s.children.
        - Warning: s._players is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRand class has to be > 0."
        self.nbPlayers = nbPlayers
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers
        self.rankSelectionAlgo = rankSelectionAlgo
        self.nbArms = nbArms
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRhoLearned(nbPlayers, rankSelectionAlgo, self, playerId)
        # Fake rankSelection
        self._rankSelection = rankSelectionAlgo(nbPlayers)

    def __str__(self):
        return "rhoLearned({} x {}, ranks ~ {})".format(self.nbPlayers, str(self._players[0]), self._rankSelection)
