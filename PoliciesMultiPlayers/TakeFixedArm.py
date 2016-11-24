# -*- coding: utf-8 -*-
""" TakeFixedArm: always select a fixed arm.
This is the perfect static policy if armIndex = bestArmIndex (not realistic, for test only).
"""

__author__ = "Lilian Besson"
__version__ = "0.1"


class TakeFixedArm():
    """ TakeFixedArm: always select a fixed arm.
    This is the perfect static policy if armIndex = bestArmIndex (not realistic, for test only).
    """

    def __init__(self, nbArms, armIndex):
        self.nbArms = nbArms
        self.armIndex = armIndex
        self.params = str(armIndex)

    def __str__(self):
        return "TakeFixedArm({})".format(self.params)

    def startGame(self):
        pass

    def getReward(self, arm, reward):
        pass

    def choice(self):
        return self.armIndex
