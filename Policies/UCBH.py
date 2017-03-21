# -*- coding: utf-8 -*-
""" The UCB-H policy for bounded bandits, with knowing the horizon.
Reference: [Audibert et al. 09].
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

from numpy import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .UCBalpha import UCBalpha, ALPHA


class UCBH(UCBalpha):
    """ The UCB-H policy for bounded bandits, with knowing the horizon.
    Reference: [Audibert et al. 09].
    """

    def __init__(self, nbArms, horizon=None, alpha=ALPHA, lower=0., amplitude=1.):
        super(UCBH, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self._horizon = horizon
        self.alpha = alpha

    def __str__(self):
        return r"UCB-H($\alpha={:.3g}$)".format(self.alpha)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def horizon(self):
        """ If the 'horizon' parameter was not provided, acts like the MOSS policy. """
        return self.t if self._horizon is None else self._horizon

    def computeIndex(self, arm):
        """ Compute the current index for this arm."""
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt((self.alpha * log(self.horizon)) / (2 * self.pulls[arm]))

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.pulls) + np.sqrt((self.alpha * np.log(self.horizon)) / (2 * self.pulls))
        indexes[self.pulls < 1] = float('+inf')
        self.index = indexes
