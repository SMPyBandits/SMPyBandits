# -*- coding: utf-8 -*-
""" SmartMusicalChair: our proposal for an efficient multi-players learning policy.

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i constantly aims at *one* of the M best arms, according to its index policy (where M is the number of players),
- When a collision occurs or when the currently chosen arm lies outside of the current estimate of the set M-best, a new current arm is chosen.

.. note:: This is not fully decentralized: as each child player needs to know the (fixed) number of players.

.. note:: Based on a variant of the multi-player policy rhoRand from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy.random as rn

from .rhoRand import oneRhoRand, rhoRand


# --- Class oneSmartMusicalChair, for children

class oneSmartMusicalChair(oneRhoRand):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new random rank is sampled after observing a collision,
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy.
    """

    def __init__(self, maxRank, *args, **kwargs):
        super(oneSmartMusicalChair, self).__init__(maxRank, *args, **kwargs)
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different.
        self.chosen_arm = None  #: Current chosen arm.

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<SmartMusicalChair[{}, M-best: {}]>".format(self.playerId + 1, self.mother._players[self.playerId], self.Mbest)

    def startGame(self):
        """Start game."""
        super(oneSmartMusicalChair, self).startGame()
        self.chosen_arm = 1 + rn.randint(self.maxRank)  # XXX Start with a random arm, safer to avoid first collisions.

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def Mbest(self):
        """ Current estimate of the M-best arms. M is the maxRank given to the algorithm."""
        return self.estimatedBestArms(self.maxRank)

    def handleCollision(self, arm, reward=None):
        """ Get a new random arm from the current estimate of Mbest, and give reward to the algorithm if not None."""
        # SmartMusicalChair UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: SmartMusicalChair UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneSmartMusicalChair, self).getReward(arm, reward)
        self.chosen_arm = rn.choice(self.Mbest)  # New random arm
        print(" - A oneSmartMusicalChair player {} saw a collision, so she had to select a new random arm {} from her estimate of M-best = {} ...".format(self, self.chosen_arm))  # DEBUG

    def getReward(self, arm, reward):
        """ Pass the call to self.mother._getReward_one(playerId, arm, reward) with the player's ID number. """
        super(oneSmartMusicalChair, self).getReward(arm, reward)
        current_Mbest = self.Mbest
        if self.chosen_arm not in current_Mbest:
            old_arm = self.chosen_arm
            self.chosen_arm = rn.choice(current_Mbest)  # New random arm
            print(" - A oneSmartMusicalChair player {} had chosen arm = {}, but it lied outside of M-best = {}, so she selected a new one = {} ...".format(self, old_arm, current_Mbest, self.chosen_arm))  # DEBUG


# --- Class SmartMusicalChair

class SmartMusicalChair(rhoRand):
    """ SmartMusicalChair: our proposal for an efficient multi-players learning policy.
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms,
                 maxRank=None, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the SmartMusicalChair child (default to nbPlayers, but for instance if there is 2 × SmartMusicalChair[UCB] + 2 × SmartMusicalChair[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = SmartMusicalChair(nbPlayers, Thompson, nbArms)

        - To get a list of usable players, use ``s.children``.

        .. warning:: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for SmartMusicalChair class has to be > 0."
        if maxRank is None:
            maxRank = nbPlayers
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.nbPlayers = nbPlayers  #: Number of players
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneSmartMusicalChair(maxRank, self, playerId)

    def __str__(self):
        return "SmartMusicalChair({} x {})".format(self.nbPlayers, str(self._players[0]))
