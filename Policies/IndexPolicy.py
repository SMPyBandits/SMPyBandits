# -*- coding: utf-8 -*-
""" Generic index policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np

from .BasePolicy import BasePolicy


class IndexPolicy(BasePolicy):
    """ Class that implements a generic index policy."""

    def __init__(self, nbArms, lower=0., amplitude=1.):
        super(IndexPolicy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.index = np.zeros(nbArms)

    def startGame(self):
        super(IndexPolicy, self).startGame()
        self.rewards.fill(0)

    def computeIndex(self, arm):
        raise NotImplementedError("This method computeIndex(arm) has to be implemented in the child class inheriting from IndexPolicy.")

    def choice(self):
        """ In an index policy, choose uniformly at random an arm with maximal index."""
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)
        # FIXED Uniform choice among the best arms
        return np.random.choice(np.where(self.index == np.max(self.index))[0])

    def choiceWithRank(self, rank=1):
        """ In an index policy, choose uniformly at random an arm with index is the (1+rank)-th best.

        - For instance, if rank is 1, the best arm is chosen (the 1-st best).
        - If rank is 4, the 4-th best arm is chosen.

        - Note: this method is *required* for the rhoRand policy.
        """
        assert rank >= 1, "Error: for IndexPolicy = {}, in choiceWithRank(rank={}) rank has to be >= 1.".format(self, rank)
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)
        # FIXME be more efficient
        try:
            uniqueValues = np.sort(np.unique(self.index))  # XXX Should we do a np.unique here ??
            chosenIndex = uniqueValues[-rank]
        except IndexError:
            values = np.sort(self.index)  # XXX What happens here if two arms has the same index, being the max?
            chosenIndex = values[-rank]
        # FIXED Uniform choice among the rank-th best arms
        return np.random.choice(np.where(self.index == chosenIndex)[0])
