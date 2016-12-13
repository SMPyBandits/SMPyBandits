# -*- coding: utf-8 -*-
""" The UCB-Tuned policy for bounded bandits, with a tuned variance correction term.
Reference: [Auer et al. 02].
"""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.7 $"

from math import sqrt, log

from .UCBV import UCBV


class UCBtuned(UCBV):
    """ The UCB-Tuned policy for bounded bandits, with a tuned variance correction term.
    Reference: [Auer et al. 02].
    """

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            mean = self.rewards[arm] / self.pulls[arm]   # Mean estimate
            variance = (self.rewardsSquared[arm] / self.pulls[arm]) - mean ** 2  # Variance estimate
            # Correct variance estimate
            variance += sqrt(2.0 * log(self.t) / self.pulls[arm])
            return mean + sqrt(log(self.t) * variance / self.pulls[arm])
