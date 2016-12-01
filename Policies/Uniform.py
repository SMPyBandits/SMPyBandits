# -*- coding: utf-8 -*-
""" Uniform: the fully uniform policy who selects randomly (uniformly) an arm at each step (stupid).
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from random import randint


class Uniform(object):
    """ Uniform: the fully uniform policy who selects randomly (uniformly) an arm at each step (stupid).
    """

    def __init__(self, nbArms):
        self.nbArms = nbArms

    def __str__(self):
        return "Uniform"

    def startGame(self):
        pass

    def getReward(self, arm, reward):
        pass

    def choice(self):
        return randint(0, self.nbArms - 1)
