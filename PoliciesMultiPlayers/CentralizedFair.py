# -*- coding: utf-8 -*-
""" CentralizedFair: a multi-player policy which uses a centralize intelligence to affect users an offset, each one take an orthogonal arm based on (offset + t) % nbArms.

- It allows to have absolutely *no* collision, if there is more channels than users (always assumed).
- And it is perfectly fair on every run: the best arm is played successively by each player.
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np

from .ChildPointer import ChildPointer


class Cycling(object):
    """ Cycling: select an arm as (offset + t) % nbArms, with offset being decided by the CentralizedFair multi-player policy.
    """

    def __init__(self, nbArms, offset):
        self.nbArms = nbArms
        self.offset = offset
        self.params = str(offset)
        self._t = -1

    def __str__(self):
        return "Cycling({})".format(self.params)

    def startGame(self):
        self._t = 0

    def getReward(self, arm, reward):
        pass

    def choice(self):
        c = (self.offset + self._t) % self.nbArms
        self._t += 1
        return c


class CentralizedFair(object):
    """ CentralizedFair: a multi-player policy which uses a centralize intelligence to affect users an offset, each one take an orthogonal arm based on (offset + t) % nbArms.
    """

    def __init__(self, nbPlayers, nbArms):
        """
        - nbPlayers: number of players to create (in self._players).
        - nbArms: number of arms, given as first argument to playerAlgo.

        Examples:
        >>> s = CentralizedFair(10, 14)

        - To get a list of usable players, use s.childs.
        - Warning: s._players is for internal use
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for CentralizedFair class has to be > 0."
        assert nbPlayers <= nbArms, "Error, the PoliciesMultiPlayers.CentralizedFair class is not YET ready to deal with the case nbPlayers > nbArms."
        # Attributes
        self.nbPlayers = nbPlayers
        self.nbArms = nbArms
        # Internal vectorial memory
        if nbPlayers <= nbArms:
            self._offsets = np.random.choice(nbArms, size=nbPlayers, replace=False)
        # FIXME check the general case
        else:
            self._offsets = np.zeros(nbPlayers)
            # Try to minimize the number of doubled offsets
            self._offsets[:nbArms] = np.random.choice(nbArms, size=nbArms, replace=False)
            self._offsets[nbArms:] = np.random.choice(nbArms, size=nbArms, replace=True)
        # Shuffle it once, just to be fair in average
        np.random.shuffle(self._offsets)
        print("CentralizedFair: initialized with {} arms and {} players ...".format(nbArms, nbPlayers))  # DEBUG
        print("It decided to use this affectation of arms :")  # DEBUG
        # Internal object memory
        self._players = [None] * nbPlayers
        self.childs = [None] * nbPlayers
        for playerId in range(nbPlayers):
            print(" - Player number {} will always choose the arm number {} ...".format(playerId + 1, self._offsets[playerId]))  # DEBUG
            self._players[playerId] = Cycling(nbArms, self._offsets[playerId])
            self.childs[playerId] = ChildPointer(self, playerId)
        self._printNbCollisions()  # DEBUG
        self.params = '{} x {}'.format(nbPlayers, str(self._players[0]))

    def __str__(self):
        return "CentralizedFair({})".format(self.params)

    def _printNbCollisions(self):
        """ Print number of collisions. """
        nbPlayersAlone = np.size(np.unique(self._offsets))
        if nbPlayersAlone != self.nbPlayers:
            print("\n==> This affectation will bring collisions! Exactly {} at each step...".format(self.nbPlayers - nbPlayersAlone))
            for armId in range(self.nbArms):
                nbAffected = np.count_nonzero(self._offsets == armId)
                if nbAffected > 1:
                    print(" - For arm number {}, there is {} different child player affected on this arm ...".format(armId + 1, nbAffected))

    def startGame(self):
        for player in self._players:
            player.startGame()

    def getReward(self, arm, reward):
        for player in self._players:
            player.getReward(arm, reward)()

    def choice(self):
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
