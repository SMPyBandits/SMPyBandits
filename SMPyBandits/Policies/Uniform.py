# -*- coding: utf-8 -*-
""" Uniform: the fully uniform policy who selects randomly (uniformly) an arm at each step (stupid).
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from random import randint
from .BasePolicy import BasePolicy


class Uniform(BasePolicy):
    """ Uniform: the fully uniform policy who selects randomly (uniformly) an arm at each step (stupid).
    """

    def __init__(self, nbArms, lower=0., amplitude=1.):
        """Nothing to do."""
        self.nbArms = nbArms  #: Number of arms

    def __str__(self):
        return "U(1..{})".format(self.nbArms)

    def startGame(self):
        """Nothing to do."""
        pass

    def getReward(self, arm, reward):
        """Nothing to do."""
        pass

    def choice(self):
        """Uniform random choice between 0 and nbArms - 1 (included)."""
        return randint(0, self.nbArms - 1)

    def choiceWithRank(self, rank=1):
        """Ignore the rank!"""
        return self.choice()
