# -*- coding: utf-8 -*-
""" The MOSS-Anytime policy for bounded bandits, without knowing the horizon (and no doubling trick).
Reference: [Degenne & Perchet, 2016](http://proceedings.mlr.press/v48/degenne16.pdf).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .MOSS import MOSS


#: Default value for the parameter :math:`\alpha` for the MOSS-Anytime algorithm.
ALPHA = 1.0


class MOSSAnytime(MOSS):
    """ The MOSS-Anytime policy for bounded bandits, without knowing the horizon (and no doubling trick).
    Reference: [Degenne & Perchet, 2016](http://proceedings.mlr.press/v48/degenne16.pdf).
    """

    def __init__(self, nbArms, alpha=ALPHA, lower=0., amplitude=1.):
        super(MOSSAnytime, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.alpha = alpha  #: Parameter :math:`\alpha \geq 0` for the computations of the index. Optimal value seems to be :math:`1.35`.

    def __str__(self):
        return r"MOSS-Anytime($\alpha={}$)".format(self.alpha)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, if there is K arms:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\left(\frac{1+\alpha}{2}\right) \max\left(0, \frac{\log\left(\frac{t}{K N_k(t)}\right)}{N_k(t)}\right)}.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + np.sqrt(((1. + self.alpha) / 2.) * max(0, np.log(self.t / (self.nbArms * self.pulls[arm]))) / self.pulls[arm])

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.pulls) + np.sqrt(((1. + self.alpha) / 2.) * np.maximum(0., np.log(self.t / (self.nbArms * self.pulls))) / self.pulls)
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
