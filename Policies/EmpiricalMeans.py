# -*- coding: utf-8 -*-
""" The naive Empirical Means policy for bounded bandits: like UCB but without a bias correction term. Note that it is equal to UCBalpha with alpha=0, only quicker."""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np
np.seterr(divide='ignore', invalid='ignore')  # XXX dangerous in general, controlled here!

from .IndexPolicy import IndexPolicy


class EmpiricalMeans(IndexPolicy):
    """ The naive Empirical Means policy for bounded bandits: like UCB but without a bias correction term. Note that it is equal to UCBalpha with alpha=0, only quicker."""

    def computeIndex(self, arm):
        """ Compute the current index for this arm."""
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return self.rewards[arm] / self.pulls[arm]

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = self.rewards / self.pulls
        indexes[self.pulls < 1] = float('+inf')
        self.index = indexes
