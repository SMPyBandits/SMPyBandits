# -*- coding: utf-8 -*-
""" The MOSS policy for bounded bandits.
Reference: [Audibert & Bubeck, 2010].
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from numpy import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .IndexPolicy import IndexPolicy


class MOSS(IndexPolicy):
    """ The MOSS policy for bounded bandits.
    Reference: [Audibert & Bubeck, 2010].
    """

    def computeIndex(self, arm):
        """ Compute the current index for this arm."""
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt(max(0, log(self.t / (self.nbArms * self.pulls[arm]))) / self.pulls[arm])

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        return (self.rewards / self.pulls) + np.sqrt(np.max(0., np.log(self.t / (self.nbArms * self.pulls))) / self.pulls)
