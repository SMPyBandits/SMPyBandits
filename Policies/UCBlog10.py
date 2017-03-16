# -*- coding: utf-8 -*-
""" The UCB policy for bounded bandits, using log10(t) and not ln(t) for UCB index.
Reference: [Lai & Robbins, 1985].
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import sqrt, log10
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .IndexPolicy import IndexPolicy


class UCBlog10(IndexPolicy):
    """ The UCB policy for bounded bandits, using log10(t) and not ln(t) for UCB index.
    Reference: [Lai & Robbins, 1985].
    """

    def computeIndex(self, arm):
        """ Compute the current index for this arm."""
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt((2 * log10(self.t)) / self.pulls[arm])

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.pulls) + np.sqrt((2 * np.log10(self.t)) / self.pulls)
        indexes[self.pulls < 1] = float('+inf')
        self.index = indexes
