# -*- coding: utf-8 -*-
""" Generic index policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

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
        self.index = np.zeros(nbArms)  #: Numerical index for each arms

    # --- Start game, and receive rewards

    def startGame(self):
        """ Initialize the policy for a new game."""
        super(IndexPolicy, self).startGame()
        self.index.fill(0)

    def computeIndex(self, arm):
        """ Compute the current index of arm 'arm'."""
        raise NotImplementedError("This method computeIndex(arm) has to be implemented in the child class inheriting from IndexPolicy.")

    def computeAllIndex(self):
        """ Compute the current indexes for all arms. Possibly vectorized, by default it can *not* be vectorized automatically."""
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)

    # --- Basic choice() method

    def choice(self):
        r""" In an index policy, choose an arm with maximal index (uniformly at random):

        .. math:: A(t) \sim U(\arg\max_{1 \leq k \leq K} I_k(t)).
        """
        self.computeAllIndex()
        # Uniform choice among the best arms
        try:
            return np.random.choice(np.nonzero(self.index == np.max(self.index))[0])
        except ValueError:
            if not np.all(np.isnan(self.index)):
                raise ValueError("Error: unknown error in IndexPolicy.choice(): the indexes were {} but couldn't be used to select an arm.".format(self.index))
            return np.random.randint(self.nbArms)

    # --- Others choice...() methods

    def choiceWithRank(self, rank=1):
        """ In an index policy, choose an arm with index is the (1+rank)-th best (uniformly at random).

        - For instance, if rank is 1, the best arm is chosen (the 1-st best).
        - If rank is 4, the 4-th best arm is chosen.


        .. note:: This method is *required* for the rhoRand policy.

        """
        if rank == 1:
            return self.choice()
        else:
            assert rank >= 1, "Error: for IndexPolicy = {}, in choiceWithRank(rank={}) rank has to be >= 1.".format(self, rank)
            self.computeAllIndex()
            sortedRewards = np.sort(self.index)
            # Question: What happens here if two arms has the same index, being the max?
            # Then it is fair to chose a random arm with best index, instead of aiming at an arm with index being ranked rank
            chosenIndex = sortedRewards[-rank]
            # Uniform choice among the rank-th best arms
            return np.random.choice(np.nonzero(self.index == chosenIndex)[0])

    def choiceFromSubSet(self, availableArms='all'):
        """ In an index policy, choose the best arm from sub-set availableArms (uniformly at random)."""
        if isinstance(availableArms, str) and availableArms == 'all':
            return self.choice()
        # If availableArms are all arms?
        # elif len(availableArms) == self.nbArms:
        #     return self.choice()
        elif len(availableArms) == 0:
            warn("IndexPolicy.choiceFromSubSet({}): the argument availableArms of type {} should not be empty.".format(availableArms, type(availableArms)), RuntimeWarning)
            # FIXME if no arms are tagged as available, what to do ? choose an arm at random, or call choice() as if available == 'all'
            return self.choice()
        else:
            for arm in availableArms:
                self.index[arm] = self.computeIndex(arm)
            # Uniform choice among the best arms
            return availableArms[np.random.choice(np.nonzero(self.index[availableArms] == np.max(self.index[availableArms]))[0])]

    def choiceMultiple(self, nb=1):
        """ In an index policy, choose nb arms with maximal indexes (uniformly at random)."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            self.computeAllIndex()
            sortedIndexes = np.sort(self.index)
            # Uniform choice of nb different arms among the best arms
            # FIXED sort it then apply affectation_order, to fix its order ==> will have a fixed nb of switches for CentralizedMultiplePlay
            return np.random.choice(np.nonzero(self.index >= sortedIndexes[-nb])[0], size=nb, replace=False)

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        """ In an index policy, the IMP strategy is hybrid: choose nb-1 arms with maximal empirical averages, then 1 arm with maximal index. Cf. algorithm IMP-TS [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779]."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            # For first exploration steps, do pure exploration
            if startWithChoiceMultiple:
                if np.min(self.pulls) < 1:
                    return self.choiceMultiple(nb=nb)
                else:
                    empiricalMeans = self.rewards / self.pulls
            else:
                empiricalMeans = self.rewards / self.pulls
                empiricalMeans[self.pulls < 1] = float('inf')
            # First choose nb-1 arms, from rewards
            sortedEmpiricalMeans = np.sort(empiricalMeans)
            exploitations = np.random.choice(np.nonzero(empiricalMeans >= sortedEmpiricalMeans[-nb])[0], size=nb - 1, replace=False)
            # Then choose 1 arm, from index now
            availableArms = np.setdiff1d(np.arange(self.nbArms), exploitations)
            exploration = self.choiceFromSubSet(availableArms)
            # Affect a random location to is exploratory arm
            choices = np.insert(exploitations, np.random.randint(np.size(exploitations) + 1), exploration)
            return choices  # XXX remove this useless variable

    def estimatedOrder(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means."""
        self.computeAllIndex()
        return np.argsort(self.index)

    def estimatedBestArms(self, M=1):
        """ Return a (non-necessarily sorted) list of the indexes of the M-best arms. Identify the set M-best."""
        assert 1 <= M <= self.nbArms, "Error: the parameter 'M' has to be between 1 and K = {}, but it was {} ...".format(self.nbArms, M)  # DEBUG
        # # FIXME this slows down everything, but maybe the only way to make this correct?
        # if np.any(np.isinf(self.index)) and set(self.index) == {np.inf}:
        #     # Initial guess: random estimate of the set Mbest
        #     choice = np.random.choice(self.nbArms, size=M, replace=False)
        #     print("Warning: estimatedBestArms() for self = {} was called with M = {} but all indexes are +inf, so using a random estimate = {} of Mbest instead of the biased [K-M,...,K-1] ...".format(self, M, choice))  # DEBUG
        #     return choice
        # else:
        order = self.estimatedOrder()
        return order[-M:]
