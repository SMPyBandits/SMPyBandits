# -*- coding: utf-8 -*-
""" The UCB policy for bounded bandits, using log10(t) and not ln(t) for UCB index.
Reference: [Lai & Robbins, 1985].
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import sqrt, log10

from .IndexPolicy import IndexPolicy


class UCBlog10(IndexPolicy):
    """ The UCB policy for bounded bandits, using log10(t) and not ln(t) for UCB index.
    Reference: [Lai & Robbins, 1985].
    """

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            mean = self.rewards[arm] / self.pulls[arm]   # Mean estimate
            return mean + sqrt((2 * log10(self.t)) / self.pulls[arm])
