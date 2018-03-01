# -*- coding: utf-8 -*-
""" rhoRandSticky: implementation of a variant of the multi-player policy rhoRand from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has a random rank_i from 1 to M, and when a collision occurs, rank_i is sampled from a uniform distribution on [1, .., M] where M is the number of player.
- The *only* difference with rhoRand is that once a player selected a rank and did not encounter a collision for STICKY_TIME time steps, he will never change his rank. rhoRand has STICKY_TIME = +oo, MusicalChair is something like STICKY_TIME = 1, this variant rhoRandSticky has this as a parameter.


.. note:: This is not fully decentralized: as each child player needs to know the (fixed) number of players.

"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy.random as rn

from .rhoRand import oneRhoRand, rhoRand

#: Default value for STICKY_TIME
STICKY_TIME = 10


# --- Class oneRhoRandSticky, for children

class oneRhoRandSticky(oneRhoRand):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new random rank is sampled after observing a collision,
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy.
    """

    def __init__(self, maxRank, stickyTime, *args, **kwargs):
        super(oneRhoRandSticky, self).__init__(maxRank, *args, **kwargs)
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.stickyTime = stickyTime  #: Number of time steps needed without collisions before sitting (never changing rank again)
        self.rank = None  #: Current rank, starting to 1 by default
        self.sitted = False  #: Not yet sitted. After stickyTime steps without collisions, sit and never change rank again.
        self.stepsWithoutCollisions = 0  #: Number of steps since we chose that rank and did not see any collision. As soon as this gets greater than stickyTime, the player sit.

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<{}[{}{}{}]>".format(self.playerId + 1, r"$\rho^{\mathrm{Rand}}$", self.mother._players[self.playerId], ", rank:{}".format(self.rank) if self.rank is not None else "", ", $T_0:{}$".format(self.stickyTime) if self.stickyTime > 0 else "")

    def startGame(self):
        """Start game."""
        super(oneRhoRandSticky, self).startGame()
        self.rank = 1 + rn.randint(self.maxRank)  # XXX Start with a random rank, safer to avoid first collisions.
        self.sitted = False  # Start not sitted, of course!
        self.stepsWithoutCollisions = 0  # No previous steps without collision, of course

    def handleCollision(self, arm, reward=None):
        """ Get a new fully random rank, and give reward to the algorithm if not None."""
        # rhoRandSticky UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: rhoRandSticky UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneRhoRandSticky, self).getReward(arm, reward)
        if not self.sitted:
            self.stepsWithoutCollisions = 0
            self.rank = 1 + rn.randint(self.maxRank)  # New random rank
            # print(" - A oneRhoRandSticky player {} saw a collision, so she had to select a new random rank : {} ...".format(self, self.rank))  # DEBUG

    def getReward(self, arm, reward):
        """ Pass the call to self.mother._getReward_one(playerId, arm, reward) with the player's ID number.

        - Additionally, if the current rank was good enough to not bring any collision during the last stickyTime time steps, the player "sits" on that rank.
        """
        super(oneRhoRandSticky, self).getReward(arm, reward)
        self.stepsWithoutCollisions += 1
        if not self.sitted and self.stepsWithoutCollisions >= self.stickyTime:
            # print(" - A oneRhoRandSticky player {} had rank = {}, without any collision from the last {} steps, so he is now sitted on this rank, and will not change ...".format(self, self.rank, self.stepsWithoutCollisions))  # DEBUG
            self.sitted = True


# --- Class rhoRandSticky

class rhoRandSticky(rhoRand):
    """ rhoRandSticky: implementation of a variant of the multi-player policy rhoRand from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 stickyTime=STICKY_TIME, maxRank=None, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - stickyTime: given to the oneRhoRandSticky objects (see above).
        - maxRank: maximum rank allowed by the rhoRandSticky child (default to nbPlayers, but for instance if there is 2 × rhoRandSticky[UCB] + 2 × rhoRandSticky[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = rhoRandSticky(nbPlayers, Thompson, nbArms, stickyTime)

        - To get a list of usable players, use ``s.children``.

        .. warning:: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRandSticky class has to be > 0."
        if maxRank is None:
            maxRank = nbPlayers
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.stickyTime = stickyTime  #: Number of time steps needed without collisions before sitting (never changing rank again)
        self.nbPlayers = nbPlayers  #: Number of players
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRhoRandSticky(maxRank, stickyTime, self, playerId)

    def __str__(self):
        return "rhoRandSticky({} x {}{}{})".format(self.nbPlayers, str(self._players[0]), "$T_0:{}$".format(self.stickyTime) if self.stickyTime > 0 else "")
