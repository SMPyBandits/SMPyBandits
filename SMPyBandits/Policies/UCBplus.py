# -*- coding: utf-8 -*-
""" The UCB+ policy for bounded bandits, with a small trick on the index.
Reference: [Auer et al. 02].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .UCB import UCB


class UCBplus(UCB):
    """ The UCB+ policy for bounded bandits, with a small trick on the index.
    Reference: [Auer et al. 02].
    """

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\max\left(0, \frac{\log(t / N_k(t))}{2 N_k(t)}\right)}.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt(max(0., log(self.t / (self.pulls[arm]))) / (2 * self.pulls[arm]))

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.pulls) + np.sqrt(np.maximum(0., (2 * np.log10(self.t)) / self.pulls))
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
