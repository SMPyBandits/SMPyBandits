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
        self.pulls = np.zeros(nbArms, dtype=int)
        self.rewards = np.zeros(nbArms)
        self.t = -1
        self.params = ''

    def computeIndex(self, arm):
        raise NotImplementedError("This method computeIndex(arm) has to be implemented in the child class inheriting from IndexPolicy.")

    def startGame(self):
        self.t = 0
        self._index.fill(0)
        self.pulls.fill(0)
        self.rewards.fill(0)

    def choice(self):
        """ In an index policy, choose uniformly at random an arm with maximal index."""
        for arm in range(self.nbArms):
            self._index[arm] = self.computeIndex(arm)
        maxIndex = np.max(self._index)
        # FIXED Uniform choice among the best arms
        return np.random.choice(np.where(self._index == maxIndex)[0])

    def choiceWithRank(self, rank=1):
        """ In an index policy, choose uniformly at random an arm with index is the (1+rank)-th best.

        - For instance, if rank is 1, the best arm is chosen (the 1-st best).
        - If rank is 4, the 4-th best arm is chosen.

        - Note: this method is *required* for the rhoRand policy.
        """
        assert rank >= 1, "Error: for IndexPolicy = {}, in choiceWithRank(rank={}) rank has to be >= 1.".format(self, rank)
        for arm in range(self.nbArms):
            self._index[arm] = self.computeIndex(arm)
        # FIXME be more efficient
        try:
            values = np.sort(np.unique(self._index))  # XXX Should we do a np.unique here ??
            chosenIndex = values[-rank]
        except IndexError:
            values = np.sort(self._index)
            chosenIndex = values[-rank]
        # FIXED Uniform choice among the arms of same index
        return np.random.choice(np.where(self._index == chosenIndex)[0])

    def getReward(self, arm, reward):
        self.t += 1
        self.pulls[arm] += 1
        self.rewards[arm] += reward

    def __str__(self):
        return self.__class__.__name__
