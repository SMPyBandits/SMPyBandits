# -*- coding: utf-8 -*-
""" rhoRand: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

FIXME explain
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy.random as rn

from .ChildPointer import ChildPointer


# --- Class oneRhoRand

class oneRhoRand(ChildPointer):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.
    """

    def __init__(self, nbPlayers, *args, **kwargs):
        super(oneRhoRand, self).__init__(*args, **kwargs)
        self.nbPlayers = nbPlayers
        self.rank = 1  # Start with a rank = 1: assume she is alone.

    def __str__(self):   # Better to recompute it automatically
        return '#{}<rhoRand, {}, rank:{}>'.format(self.playerId + 1, self.mother._players[self.playerId], self.rank)

    def startGame(self):
        super(oneRhoRand, self).startGame()
        self.rank = 1  # Start with a rank = 1: assume she is alone.

    def handleCollision(self, arm):
        self.rank = 1 + rn.randint(self.nbPlayers)  # New random rank
        # print(" - A oneRhoRand player {} saw a collision, so she had to select a new random rank : {} ...".format(self, self.rank))  # DEBUG
        super(oneRhoRand, self).handleCollision(arm)

    def choice(self):
        result = super(oneRhoRand, self).choiceWithRank(self.rank)
        # print(" - A oneRhoRand player {} had to choose an arm among the best from rank {}, her choice was : {} ...".format(self, self.rank, result))  # DEBUG
        return result


# --- Class rhoRand

class rhoRand(object):
    """ rhoRand: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms, *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - *args, **kwargs: arguments, named arguments, given to playerAlgo.

        Examples:
        >>> s = rhoRand(NB_PLAYERS, Thompson, nbArms)

        - To get a list of usable players, use s.childs.
        - Warning: s._players is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRand class has to be > 0."
        self.nbPlayers = nbPlayers
        self._players = [None] * nbPlayers
        self.childs = [None] * nbPlayers
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, **kwargs)
            self.childs[playerId] = oneRhoRand(nbPlayers, self, playerId)
        self.nbArms = nbArms
        self.params = '{} x {}'.format(nbPlayers, str(self._players[0]))

    def __str__(self):
        return "rhoRand({})".format(self.params)

    # --- Proxy methods

    def _startGame_one(self, playerId):
        return self._players[playerId].startGame()

    def _getReward_one(self, playerId, arm, reward):
        return self._players[playerId].getReward(arm, reward)

    def _choiceWithRank_one(self, playerId, rank):
        return self._players[playerId].choiceWithRank(rank)

    def _handleCollision_one(self, playerId, arm):
        player = self._players[playerId]
        if hasattr(player, 'handleCollision'):
            # player.handleCollision(arm) is called to inform the user that there were a collision
            player.handleCollision(arm)
        else:
            # And if it does not have this method, call players[j].getReward() with a reward = 0 to change the internals memory of the player ?
            player.getReward(arm, 0)
            # FIXME Strong assumption on the model
