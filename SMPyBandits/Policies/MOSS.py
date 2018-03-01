# -*- coding: utf-8 -*-
""" The MOSS policy for bounded bandits.
Reference: [Audibert & Bubeck, 2010](http://www.jmlr.org/papers/volume11/audibert10a/audibert10a.pdf).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .IndexPolicy import IndexPolicy


class MOSS(IndexPolicy):
    """ The MOSS policy for bounded bandits.
    Reference: [Audibert & Bubeck, 2010](http://www.jmlr.org/papers/volume11/audibert10a/audibert10a.pdf).
    """

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, if there is K arms:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\max\left(0, \frac{\log\left(\frac{t}{K N_k(t)}\right)}{N_k(t)}\right)}.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + np.sqrt(max(0, np.log(self.t / (self.nbArms * self.pulls[arm]))) / self.pulls[arm])

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.pulls) + np.sqrt(np.maximum(0., np.log(self.t / (self.nbArms * self.pulls))) / self.pulls)
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
