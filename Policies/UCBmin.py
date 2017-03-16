# -*- coding: utf-8 -*-
""" The UCB-min policy for bounded bandits, with a min(1, sqrt(...)) term.
Reference: [Anandkumar et al., 2010].
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .UCB import UCB


class UCBmin(UCB):
    """ The UCB-min policy for bounded bandits, with a min(1, sqrt(...)) term.
    Reference: [Anandkumar et al., 2010].
    """

    def computeIndex(self, arm):
        """ Compute the current index for this arm."""
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + min(1., sqrt(log(self.t) / (2 * self.pulls[arm])))

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.pulls) + np.minimum(1., np.sqrt((2 * np.log10(self.t)) / self.pulls))
        indexes[self.pulls < 1] = float('+inf')
        self.index = indexes
