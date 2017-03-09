# -*- coding: utf-8 -*-
""" The UCB policy for bounded bandits.
Reference: [Lai & Robbins, 1985].
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import sqrt, log

from .IndexPolicy import IndexPolicy


class UCB(IndexPolicy):
    """ The UCB policy for bounded bandits.
    Reference: [Lai & Robbins, 1985].
    """

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            return self.rewards[arm] / self.pulls[arm] + sqrt((2 * log(self.t)) / self.pulls[arm])
