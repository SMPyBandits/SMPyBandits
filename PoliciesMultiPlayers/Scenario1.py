# -*- coding: utf-8 -*-
""" Scenario1: make a set of M experts with the following behavior, for K = 2 arms: at every round, one of them is chosen uniformly to predict arm 0, and the rest predict 1.

- Reference: Beygelzimer, A., Langford, J., Li, L., Reyzin, L., & Schapire, R. E. (2011, April). Contextual Bandit Algorithms with Supervised Learning Guarantees. In AISTATS (pp. 19-26).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np

from .BaseMPPolicy import BaseMPPolicy
from .ChildPointer import ChildPointer


# --- Class for a child player

class OneScenario1(ChildPointer):
    """ OneScenario1: at every round, one of them is chosen uniformly to predict arm 0, and the rest predict 1.
    """
    def __init__(self, mother, playerId):
        super(OneScenario1, self).__init__(mother, playerId)

    def __str__(self):
        return "#{}<OneScenario1>".format(self.playerId + 1)

    def __repr__(self):
        return "OneScenario1"


# --- Class for the mother

class Scenario1(BaseMPPolicy):
    """ Scenario1: make a set of M experts with the following behavior, for K = 2 arms: at every round, one of them is chosen uniformly to predict arm 0, and the rest predict 1.

    - Reference: Beygelzimer, A., Langford, J., Li, L., Reyzin, L., & Schapire, R. E. (2011, April). Contextual Bandit Algorithms with Supervised Learning Guarantees. In AISTATS (pp. 19-26).
    """

    def __init__(self, nbPlayers, nbArms, lower=0., amplitude=1.):
        """
        - nbPlayers: number of players to create (in self._players).

        Examples:

        >>> s = Scenario1(10)

        - To get a list of usable players, use ``s.children``.
        - Warning: ``s._players`` is for internal use
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for Scenario1 class has to be > 0."
        assert nbArms >= 2, "Error, the parameter 'nbArms' for Scenario1 class can only be >= 2."  # DEBUG
        # Attributes
        self.nbPlayers = nbPlayers
        self.nbArms = nbArms
        self.chosenOne = None
        # Internal object memory
        self.children = [None] * nbPlayers
        for playerId in range(nbPlayers):
            self.children[playerId] = OneScenario1(self, playerId)
            # print(" - One new child, of index {}, and class {} ...".format(playerId, self.children[playerId]))  # DEBUG

    def __str__(self):
        return "Scenario1({})".format(self.nbPlayers)

    def _startGame_one(self, playerId):
        self.chosenOne = np.random.randint(self.nbPlayers)  # New random choice

    def _getReward_one(self, playerId, arm, reward):
        pass

    def _choice_one(self, playerId):
        if playerId == 0:  # For the first player, chose a new chosenOne
            self.chosenOne = np.random.randint(self.nbPlayers)  # New random choice
        # print("  Currently, the only sub-player that can pull arm #0 is", self.chosenOne, "and playerId =", playerId)  # DEBUG
        if self.chosenOne == playerId:
            return 0  # Choose worse arm
        else:
            if self.nbArms > 2:
                return np.random.randint(low=1, high=1 + self.nbArms)  # to be general for nbArms > 2 setting
            else:
                return 1  # Choose best arm
