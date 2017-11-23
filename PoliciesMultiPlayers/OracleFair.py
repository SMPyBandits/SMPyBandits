# -*- coding: utf-8 -*-
""" OracleFair: a multi-player policy which uses a centralized intelligence to affect users an offset, each one take an orthogonal arm based on (offset + t) % nbBestArms, among the best arms.

- It allows to have absolutely *no* collision, if there is more channels than users (always assumed).
- And it is perfectly fair on every run: each chosen arm is played successively by each player.
- Note that it IS affecting players on the best arms: it requires full knowledge of the means of the arms, not simply the number of arms.

- Note that they need a perfect knowledge on the arms, even this is not physically plausible.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np

from .BaseMPPolicy import BaseMPPolicy
from .BaseCentralizedPolicy import BaseCentralizedPolicy
from .ChildPointer import ChildPointer


class CyclingBest(BaseCentralizedPolicy):
    """ CyclingBest: select an arm in the best ones (bestArms) as (offset + t) % (len(bestArms)), with offset being decided by the OracleFair multi-player policy.
    """

    def __init__(self, nbArms, offset, bestArms=None):
        """Cycling with an offset."""
        self.nbArms = nbArms  #: Number of arms
        self.offset = offset  #: Offset
        if bestArms is None:
            bestArms = list(range(nbArms))
        self.bestArms = bestArms  #: List of index of the best arms to play
        self.nb_bestArms = len(bestArms)  #: Number of best arms
        self.t = -1  #: Internal time

    def __str__(self):
        return "CyclingBest({}, {})".format(self.offset, self.bestArms)

    def startGame(self):
        """Nothing to do."""
        pass

    def getReward(self, arm, reward):
        """Nothing to do."""
        pass

    def choice(self):
        """Chose cycling arm."""
        self.t += 1
        return self.bestArms[(self.offset + self.t) % self.nb_bestArms]


class OracleFair(BaseMPPolicy):
    """ OracleFair: a multi-player policy which uses a centralize intelligence to affect users an offset, each one take an orthogonal arm based on (offset + t) % nbArms.
    """

    def __init__(self, nbPlayers, armsMAB, lower=0., amplitude=1.):
        """
        - nbPlayers: number of players to create (in self._players).
        - armsMAB: MAB object that represents the arms.

        Examples:

        >>> s = OracleFair(10, MAB({'arm_type': Bernoulli, 'params': [0.1, 0.5, 0.9]}))

        - To get a list of usable players, use ``s.children``.
        - Warning: ``s._players`` is for internal use
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for OracleFair class has to be > 0."
        nbArms = armsMAB.nbArms
        if nbPlayers > nbArms:
            print("Warning, there is more users than arms ... (nbPlayers > nbArms)")  # XXX
        # Attributes
        self.nbPlayers = nbPlayers  #: Number of players
        self.nbArms = nbArms  #: Number of arms
        # Internal vectorial memory
        means = np.array([arm.mean() for arm in armsMAB.arms])
        bestArms = np.argsort(means)[-min(nbPlayers, nbArms):]
        print("bestArms =", bestArms)  # DEBUG
        if nbPlayers <= nbArms:
            self._offsets = np.argsort(means)[-nbPlayers:]  # Decide the offsets of the centralized players
        else:
            self._offsets = np.zeros(nbPlayers, dtype=int)
            self._offsets[:nbArms] = np.random.permutation(nbArms)
            # Try to minimize the number of doubled offsets, so all the other players are affected to the *same* arm
            worseArm = np.argmin(means)
            self._offsets[nbArms:] = worseArm
            # FIXME improve this, when there is more player than arms, this is not optimal
            # indeed, all the collisions will first be in worseArm, but they all cycle! That's bad
            # XXX this "trash" arm with max number of collision will cycle: that's the best we can do!
            # self._offsets[nbArms:] = np.random.choice(nbArms, size=nbPlayers - nbArms, replace=True)
        # Shuffle it once, just to be fair in average
        np.random.shuffle(self._offsets)
        print("OracleFair: initialized with {} arms and {} players ...".format(nbArms, nbPlayers))  # DEBUG
        print("It decided to use this affectation of arms :")  # DEBUG
        # Internal object memory
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        for playerId in range(nbPlayers):
            print(" - Player number {} will use an offset of {} ...".format(playerId + 1, self._offsets[playerId]))  # DEBUG
            self._players[playerId] = CyclingBest(nbArms, self._offsets[playerId], bestArms)
            self.children[playerId] = ChildPointer(self, playerId)
        self._printNbCollisions()  # DEBUG

    def __str__(self):
        return "OracleFair({} x {})".format(self.nbPlayers, str(self._players[0]))

    def _printNbCollisions(self):
        """ Print number of collisions. """
        nbPlayersAlone = len(set(self._offsets))
        if nbPlayersAlone != self.nbPlayers:
            print("\n==> This affectation will bring collisions! Exactly {} at each step...".format(self.nbPlayers - nbPlayersAlone))
            for armId in range(self.nbArms):
                nbAffected = np.count_nonzero(self._offsets == armId)
                if nbAffected > 1:
                    print(" - For arm number {}, there is {} different child player affected on this arm ...".format(armId + 1, nbAffected))
