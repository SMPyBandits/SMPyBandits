# -*- coding: utf-8 -*-
""" Generic index policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""

__author__ = "Lilian Besson"
__version__ = "0.3"

from warnings import warn
import numpy as np

from .BasePolicy import BasePolicy


class IndexPolicy(BasePolicy):
    """ Class that implements a generic index policy."""

    def __init__(self, nbArms, lower=0., amplitude=1.):
        """ New generic index policy.

        - nbArms: the number of arms,
        - lower, amplitude: lower value and known amplitude of the rewards.
        """
        super(IndexPolicy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.index = np.zeros(nbArms)

    # --- Start game, and receive rewards

    def startGame(self):
        """ Initialize the policy for a new game."""
        super(IndexPolicy, self).startGame()
        self.index.fill(0)

    def computeIndex(self, arm):
        """ Compute the current index of arm 'arm'."""
        raise NotImplementedError("This method computeIndex(arm) has to be implemented in the child class inheriting from IndexPolicy.")

    # --- Basic choice() method

    def choice(self):
        """ In an index policy, choose an arm with maximal index (uniformly at random)."""
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)
        # Uniform choice among the best arms
        return np.random.choice(np.nonzero(self.index == np.max(self.index))[0])

    # --- Others choice...() methods

    def choiceWithRank(self, rank=1):
        """ In an index policy, choose an arm with index is the (1+rank)-th best (uniformly at random).

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
            sortedRewards = np.sort(self.index)
            # Question: What happens here if two arms has the same index, being the max?
            # Then it is fair to chose a random arm with best index, instead of aiming at an arm with index being ranked rank
            chosenIndex = sortedRewards[-rank]
            # Uniform choice among the rank-th best arms
            return np.random.choice(np.nonzero(self.index == chosenIndex)[0])

    def choiceFromSubSet(self, availableArms='all'):
        """ In an index policy, choose the best arm from sub-set availableArms (uniformly at random)."""
        if isinstance(availableArms, str) and isinstance == 'all':
            return self.choice()
        # If availableArms are all arms?
        elif len(availableArms) == self.nbArms:
            return self.choice()
        elif len(availableArms) == 0:
            warn("IndexPolicy.choiceFromSubSet({}): the argument availableArms of type {} should not be empty.".format(availableArms, type(availableArms)), RuntimeWarning)
            # FIXME if no arms are tagged as available, what to do ? choose an arm at random, or call choice() as if available == 'all'
            return self.choice()
            # return np.random.randint(self.nbArms)
        else:
            for arm in availableArms:
                self.index[arm] = self.computeIndex(arm)
            # Uniform choice among the best arms
            return availableArms[np.random.choice(np.nonzero(self.index[availableArms] == np.max(self.index[availableArms]))[0])]

    def choiceMultiple(self, nb=1):
        """ In an index policy, choose nb arms with maximal indexes (uniformly at random)."""
        if nb == 1:
            return self.choice()
        else:
            for arm in range(self.nbArms):
                self.index[arm] = self.computeIndex(arm)
            sortedIndexes = np.sort(self.index)
            # Uniform choice of nb different arms among the best arms
            # FIXED sort it then apply affectation_order, to fix its order ==> will have a fixed nb of switches for CentralizedMultiplePlay
            return np.random.choice(np.nonzero(self.index >= sortedIndexes[-nb])[0], size=nb, replace=False)

    def estimatedOrder(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means."""
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)
        return np.argsort(self.index)
