# -*- coding: utf-8 -*-
""" The MOSS policy for bounded bandits.
Reference: [Audibert & Bubeck, 2010].
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from numpy import sqrt, log

from .IndexPolicy import IndexPolicy


class MOSS(IndexPolicy):
    """ The MOSS policy for bounded bandits.
    Reference: [Audibert & Bubeck, 2010].
    """

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            return self.rewards[arm] / self.pulls[arm] + sqrt(max(0, log(self.t / (self.nbArms * self.pulls[arm]))) / self.pulls[arm])
