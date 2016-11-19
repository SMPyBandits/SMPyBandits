# -*- coding: utf-8 -*-
""" Generic index policy.
"""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.5 $"


import random


class IndexPolicy:
    """ Class that implements a generic index policy."""

    def __init__(self):
        self.nbArms = 0

    def computeIndex(self, arm):
        pass

    def choice(self):
        """ In an index policy, choose uniformly at random an arm with maximal index."""
        index = dict()
        for arm in range(self.nbArms):
            index[arm] = self.computeIndex(arm)
        maxIndex = max(index.values())
        bestArms = [arm for arm in index.keys() if index[arm] == maxIndex]
        return random.choice(bestArms)  # Uniform choice among the best arms

    def __str__(self):
        return self.__class__.__name__
