# -*- coding: utf-8 -*-
""" Selfish: a multi-player policy where every player is selfish, playing on their side.

- without knowing how many players there is,
- and not even knowing that they should try to avoid collisions.
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np

from .ChildPointer import ChildPointer


class Selfish(object):
    """ Selfish: a multi-player policy where every player is selfish, playing on their side (without.

    - nowing how many players there is, and
    - not even knowing that they should try to avoid collisions.
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms, *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - *args, **kwargs: arguments, named arguments, given to playerAlgo.

        Examples:
        >>> s = Selfish(10, TakeFixedArm, 14)
        >>> s = Selfish(NB_PLAYERS, Softmax, nbArms, temperature=TEMPERATURE)

        - To get a list of usable players, use s.childs.
        - Warning: s._players is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for Selfish class has to be > 0."
        self.nbPlayers = nbPlayers
        self._players = [None] * nbPlayers
        self.childs = [None] * nbPlayers
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, **kwargs)
            self.childs[playerId] = ChildPointer(self, playerId)
        self.nbArms = nbArms
        self.params = '{} x {}'.format(nbPlayers, str(self._players[0]))

    def __str__(self):
        return "Selfish({})".format(self.params)

    def startGame(self):
        # XXX Not used right now!
        for player in self._players:
            player.startGame()

    def getReward(self, arm, reward):
        # XXX Not used right now!
        for player in self._players:
            player.getReward(arm, reward)()

    def choice(self):
        # XXX Not used right now!
        choices = np.zeros(self.nbPlayers)
        for i, player in enumerate(self._players):
            choices[i] = player.choice()
        return choices  # XXX What to do with this ?

    def _startGame_one(self, playerId):
        return self._players[playerId].startGame()

    def _getReward_one(self, playerId, arm, reward):
        return self._players[playerId].getReward(arm, reward)

    def _choice_one(self, playerId):
        return self._players[playerId].choice()
