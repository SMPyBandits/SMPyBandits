# -*- coding: utf-8 -*-
""" Generic index policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""

__author__ = "Lilian Besson"
__version__ = "0.3"

import numpy as np

from .BasePolicy import BasePolicy


class IndexPolicy(BasePolicy):
    """ Class that implements a generic index policy."""

    def __init__(self, nbArms, lower=0., amplitude=1.):
        super(IndexPolicy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.index = np.zeros(nbArms)

    # --- Start game, and receive rewards

    def startGame(self):
        super(IndexPolicy, self).startGame()
        self.index.fill(0)

    def computeIndex(self, arm):
        raise NotImplementedError("This method computeIndex(arm) has to be implemented in the child class inheriting from IndexPolicy.")

    # --- Basic choice() method

    def choice(self):
        """ In an index policy, choose uniformly at random an arm with maximal index."""
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)
        # Uniform choice among the best arms
        return np.random.choice(np.nonzero(self.index == np.max(self.index)))

    # --- Others choice...() methods

    def choiceWithRank(self, rank=1):
        """ In an index policy, choose uniformly at random an arm with index is the (1+rank)-th best.

        - For instance, if rank is 1, the best arm is chosen (the 1-st best).
        - If rank is 4, the 4-th best arm is chosen.

        - Note: this method is *required* for the rhoRand policy.
        """
        if rank == 1:
            return self.choice()
        else:
            assert rank >= 1, "Error: for IndexPolicy = {}, in choiceWithRank(rank={}) rank has to be >= 1.".format(self, rank)
            for arm in range(self.nbArms):
                self.index[arm] = self.computeIndex(arm)
            # FIXME be more efficient?
            # try:
            #     sortedUniqueRewards = np.sort(np.unique(self.index))  # XXX Should we do a np.unique here ??
            #     chosenIndex = sortedUniqueRewards[-rank]
            # except IndexError:
            sortedRewards = np.sort(self.index)  # XXX What happens here if two arms has the same index, being the max?
            chosenIndex = sortedRewards[-rank]
            # Uniform choice among the rank-th best arms
            return np.random.choice(np.nonzero(self.index == chosenIndex))

    def choiceFromSubSet(self, availableArms='all'):
        if availableArms == 'all':
            return self.choice()
        else:
            for arm in availableArms:
                self.index[arm] = self.computeIndex(arm)
            # Uniform choice among the best arms
            return np.random.choice(np.nonzero(self.index[availableArms] == np.max(self.index[availableArms])))

    def choiceMultiple(self, nb=1):
        """ In an index policy, choose uniformly at random nb arms with maximal indexes."""
        if nb == 1:
            return self.choice()
        else:
            for arm in range(self.nbArms):
                self.index[arm] = self.computeIndex(arm)
            sortedRewards = np.sort(self.index)
            # Uniform choice of nb different arms among the best arms
            return np.random.choice(np.nonzero(self.index >= sortedRewards[-nb])[0], size=nb, replace=False)
