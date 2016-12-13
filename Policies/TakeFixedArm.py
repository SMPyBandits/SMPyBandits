# -*- coding: utf-8 -*-
""" TakeFixedArm: always select a fixed arm.
This is the perfect static policy if armIndex = bestArmIndex (not realistic, for test only).
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from .BasePolicy import BasePolicy


class TakeFixedArm(BasePolicy):
    """ TakeFixedArm: always select a fixed arm.
    This is the perfect static policy if armIndex = bestArmIndex (not realistic, for test only).
    """

    def __init__(self, nbArms, armIndex, lower=0., amplitude=1.):
        self.nbArms = nbArms
        self.armIndex = armIndex

    def __str__(self):
        return "TakeFixedArm({})".format(self.armIndex)

    def startGame(self):
        pass

    def getReward(self, arm, reward):
        pass

    def choice(self):
        return self.armIndex

    def choiceWithRank(self, rank=1):
        """ Ignore the rank."""
        return self.choice()
