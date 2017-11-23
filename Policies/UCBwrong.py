# -*- coding: utf-8 -*-
r""" The UCBwrong policy for bounded bandits, like UCB but with a typo on the estimator of means:
:math:`\frac{X_k(t)}{t}` is used instead of :math:`\frac{X_k(t)}{N_k(t)}`.

One paper of W.Jouini, C.Moy and J.Palicot from 2009 contained this typo, I reimplemented it just to check that:

- its performance is worse than simple UCB,
- but not that bad...
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .IndexPolicy import IndexPolicy


class UCBwrong(IndexPolicy):
    """ The UCBwrong policy for bounded bandits, like UCB but with a typo on the estimator of means.

    One paper of W.Jouini, C.Moy and J.Palicot from 2009 contained this typo, I reimplemented it just to check that:

    - its performance is worse than simple UCB
    - but not that bad...
    """

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math:: I_k(t) = \frac{X_k(t)}{t} + \sqrt{\frac{2 \log(t)}{N_k(t)}}.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX Volontary typo, wrong mean estimate
            return (self.rewards[arm] / self.t) + sqrt((2 * log(self.t)) / self.pulls[arm])

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.t) + np.sqrt((2 * np.log(self.t)) / self.pulls)
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
