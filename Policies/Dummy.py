# -*- coding: utf-8 -*-
""" The fully uniform policy : selects randomly (uniformly) an arm at each step (stupid).
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from random import randint


class Dummy():
    """ The fully uniform policy : selects randomly (uniformly) an arm at each step (stupid).
    """

    def __init__(self, nbArms):
        self.nbArms = nbArms
        self.params = ''

    def __str__(self):
        return "Dummy"

    def startGame(self):
        pass

    def getReward(self, arm, reward):
        pass

    def computeIndex(self, arm):
        pass

    def choice(self):
        return randint(0, self.nbArms - 1)
