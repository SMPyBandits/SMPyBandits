# -*- coding: utf-8 -*-
""" Base class for any multi-players policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.3"


class BaseMPPolicy(object):
    """ Base class for any multi-players policy."""

    def __init__(self):
        """New policy"""
        pass

    def __str__(self):
        return self.__class__.__name__

    # --- Proxy methods

    def _startGame_one(self, playerId):
        """Forward the call to self._players[playerId]."""
        return self._players[playerId].startGame()

    def _getReward_one(self, playerId, arm, reward):
        """Forward the call to self._players[playerId]."""
        return self._players[playerId].getReward(arm, reward)

    def _choice_one(self, playerId):
        """Forward the call to self._players[playerId]."""
        return self._players[playerId].choice()

    def _choiceWithRank_one(self, playerId, rank=1):
        """Forward the call to self._players[playerId]."""
        return self._players[playerId].choiceWithRank(rank)

    def _choiceFromSubSet_one(self, playerId, availableArms='all'):
        """Forward the call to self._players[playerId]."""
        return self._players[playerId].choiceFromSubSet(availableArms)

    def _choiceMultiple_one(self, playerId, nb=1):
        """Forward the call to self._players[playerId]."""
        return self._players[playerId].choiceMultiple(nb)

    def _choiceIMP_one(self, playerId, nb=1):
        """Forward the call to self._players[playerId]."""
        return self._players[playerId].choiceIMP(nb)

    def _estimatedOrder_one(self, playerId):
        """Forward the call to self._players[playerId]."""
        return self._players[playerId].estimatedOrder()

    def _estimatedBestArms_one(self, playerId, M=1):
        """Forward the call to self._players[playerId]."""
        return self._players[playerId].estimatedBestArms(M=M)
