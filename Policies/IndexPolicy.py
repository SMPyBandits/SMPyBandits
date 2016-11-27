# -*- coding: utf-8 -*-
""" Generic index policy.
"""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.5 $"


import numpy as np


class IndexPolicy(object):
    """ Class that implements a generic index policy."""

    def __init__(self, nbArms):
        self.nbArms = nbArms
        self._index = np.zeros(self.nbArms)
        self.pulls = np.zeros(nbArms)
        self.rewards = np.zeros(nbArms)
        self.t = -1
        self.params = ''

    def computeIndex(self, arm):
        raise NotImplementedError("This method computeIndex(arm) has to be implemented in the child class inheriting from IndexPolicy.")

    def startGame(self):
        self.t = 0
        self.pulls = np.zeros(self.nbArms)
        self.rewards = np.zeros(self.nbArms)

    def choice(self):
        """ In an index policy, choose uniformly at random an arm with maximal index."""
        for arm in range(self.nbArms):
            self._index[arm] = self.computeIndex(arm)
        maxIndex = np.max(self._index)
        # bestArms = self._index[self._index == maxIndex]
        # return rn.choice(bestArms)  # FIXED choice as to be an integer
        # Uniform choice among the best arms
        return np.random.choice(np.where(self._index == maxIndex)[0])

    def getReward(self, arm, reward):
        self.t += 1
        self.pulls[arm] += 1
        self.rewards[arm] += reward

    def __str__(self):
        return self.__class__.__name__
