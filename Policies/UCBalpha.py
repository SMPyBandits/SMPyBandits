# -*- coding: utf-8 -*-
""" The UCB1 (UCB-alpha) index policy, modified to take a random permutation order for the initial exploration of each arm (reduce collisions in the multi-players setting).
Reference: [Auer et al. 02].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from math import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .UCB import UCB

#: Default parameter for alpha
ALPHA = 1
ALPHA = 4


class UCBalpha(UCB):
    """ The UCB1 (UCB-alpha) index policy, modified to take a random permutation order for the initial exploration of each arm (reduce collisions in the multi-players setting).
    Reference: [Auer et al. 02].
    """

    def __init__(self, nbArms, alpha=ALPHA, lower=0., amplitude=1.):
        super(UCBalpha, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert alpha >= 0, "Error: the alpha parameter for UCBalpha class has to be >= 0."  # DEBUG
        self.alpha = alpha  #: Parameter alpha

    def __str__(self):
        return r"UCB($\alpha={:.3g}$)".format(self.alpha)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{\alpha \log(t)}{2 N_k(t)}}.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt((self.alpha * log(self.t)) / (2 * self.pulls[arm]))

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.pulls) + np.sqrt((self.alpha * np.log(self.t)) / (2 * self.pulls))
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
