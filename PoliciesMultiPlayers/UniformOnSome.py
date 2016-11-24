# -*- coding: utf-8 -*-
""" UniformOnSome: a fully uniform policy: selects randomly (uniformly) an arm among a fix set, at each step (stupid).
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from numpy.random import choice


class UniformOnSome():
    """ UniformOnSome: a fully uniform policy: selects randomly (uniformly) an arm among a fix set, at each step (stupid).
    """

    def __init__(self, nbArms, armIndexes=None):
        self.nbArms = nbArms
        if armIndexes is None:
            armIndexes = list(range(nbArms))
        self.armIndexes = armIndexes
        self.params = repr(armIndexes)

    def __str__(self):
        return "UniformOnSome({})".format(self.params)

    def startGame(self):
        pass

    def getReward(self, arm, reward):
        pass

    def handleCollision(self, arm):
        pass

    def choice(self):
        return choice(self.armIndexes)
