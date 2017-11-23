# -*- coding: utf-8 -*-
""" CentralizedCycling: a multi-player policy which uses a centralized intelligence to affect users an offset, each one take an orthogonal arm based on (offset + t) % nbArms.

- It allows to have absolutely *no* collision, if there is more channels than users (always assumed).
- And it is perfectly fair on every run: each chosen arm is played successively by each player.
- Note that it is NOT affecting players on the best arms: it has no knowledge of the means of the arms, only of the number of arms nbArms.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np

from .BaseMPPolicy import BaseMPPolicy
from .BaseCentralizedPolicy import BaseCentralizedPolicy
from .ChildPointer import ChildPointer


class Cycling(BaseCentralizedPolicy):
    """ Cycling: select an arm as (offset + t) % nbArms, with offset being decided by the CentralizedCycling multi-player policy.
    """

    def __init__(self, nbArms, offset):
        """Cycling with an offset."""
        self.nbArms = nbArms  #: Number of arms
        self.offset = offset  #: Offset
        self.t = -1  #: Internal time

    def __str__(self):
        return "Cycling({})".format(self.offset)

    def startGame(self):
        """Nothing to do."""
        pass

    def getReward(self, arm, reward):
        """Nothing to do."""
        pass

    def choice(self):
        """Chose cycling arm."""
        self.t += 1
        return (self.offset + self.t) % self.nbArms


class CentralizedCycling(BaseMPPolicy):
    """ CentralizedCycling: a multi-player policy which uses a centralize intelligence to affect users an offset, each one take an orthogonal arm based on (offset + t) % nbArms.
    """

    def __init__(self, nbPlayers, nbArms, lower=0., amplitude=1.):
        """
        - nbPlayers: number of players to create (in self._players).
        - nbArms: number of arms.

        Examples:

        >>> s = CentralizedCycling(10, 14)

        - To get a list of usable players, use ``s.children``.
        - Warning: ``s._players`` is for internal use
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for CentralizedCycling class has to be > 0."
        if nbPlayers > nbArms:
            print("Warning, there is more users than arms ... (nbPlayers > nbArms)")  # XXX
        # Attributes
        self.nbPlayers = nbPlayers  #: Number of players
        self.nbArms = nbArms  #: Number of arms
        # Internal vectorial memory
        if nbPlayers <= nbArms:
            self._offsets = np.random.choice(nbArms, size=nbPlayers, replace=False)  # Random offsets
        else:
            self._offsets = np.zeros(nbPlayers, dtype=int)
            self._offsets[:nbArms] = np.random.permutation(nbArms)
            # Try to minimize the number of doubled offsets, so all the other players are affected to the *same* arm
            # 1. first option : chose a random offset, everyone else uses it. Plus: minimize collisions
            trashArm = np.random.choice(nbArms)
            self._offsets[nbArms:] = trashArm
            # XXX this "trash" arm with max number of collision will cycle: that's the best we can do!
        # Shuffle it once, just to be even more fair in average (by repetitions)
        np.random.shuffle(self._offsets)
        print("CentralizedCycling: initialized with {} arms and {} players ...".format(nbArms, nbPlayers))  # DEBUG
        print("It decided to use this affectation of arms :")  # DEBUG
        # Internal object memory
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        for playerId in range(nbPlayers):
            print(" - Player number {} will use an offset of {} ...".format(playerId + 1, self._offsets[playerId]))  # DEBUG
            self._players[playerId] = Cycling(nbArms, self._offsets[playerId])
            self.children[playerId] = ChildPointer(self, playerId)
        self._printNbCollisions()  # DEBUG

    def __str__(self):
        return "CentralizedCycling({} x {})".format(self.nbPlayers, str(self._players[0]))

    def _printNbCollisions(self):
        """ Print number of collisions. """
        nbPlayersAlone = len(set(self._offsets))
        if nbPlayersAlone != self.nbPlayers:
            print("\n==> This affectation will bring collisions! Exactly {} at each step...".format(self.nbPlayers - nbPlayersAlone))
            for armId in range(self.nbArms):
                nbAffected = np.count_nonzero(self._offsets == armId)
                if nbAffected > 1:
                    print(" - For arm number {}, there is {} different child player affected on this arm ...".format(armId + 1, nbAffected))
