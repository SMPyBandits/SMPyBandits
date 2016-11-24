# -*- coding: utf-8 -*-
""" Perfect knowledge policy: always select the best arm (not realistic, for test only).
"""

__author__ = "Lilian Besson"
__version__ = "0.1"


class TakeBestArm():
    """ Perfect knowledge policy: always select the best arm (not realistic, for test only).
    """

    def __init__(self, nbArms, bestArmIndex, nbUsers=None):
        self.nbArms = nbArms
        self.nbUsers = nbUsers
        self.bestArmIndex = bestArmIndex
        self.params = ''

    def __str__(self):
        return "TakeBestArm"

    def startGame(self):
        pass

    def getReward(self, arm, reward):
        pass

    def handleCollision(self, arm):
        pass

    def choice(self):
        return self.bestArmIndex
