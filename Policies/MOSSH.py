# -*- coding: utf-8 -*-
""" The MOSS-H policy for bounded bandits, with knowing the horizon.
Reference: [Audibert & Bubeck, 2010](http://www.jmlr.org/papers/volume11/audibert10a/audibert10a.pdf).
"""

__author__ = "Lilian Besson"
__version__ = "0.5"

from numpy import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .MOSS import MOSS


class MOSSH(MOSS):
    """ The MOSS-H policy for bounded bandits, with knowing the horizon.
    Reference: [Audibert & Bubeck, 2010](http://www.jmlr.org/papers/volume11/audibert10a/audibert10a.pdf).
    """

    def __init__(self, nbArms, horizon=None, lower=0., amplitude=1.):
        super(MOSSH, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self._horizon = horizon

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def horizon(self):
        """ If the 'horizon' parameter was not provided, acts like the MOSS policy. """
        return self.t if self._horizon is None else self._horizon

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, if there is K arms:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\max\left(0, \frac{\log(\frac{T}{K N_k(t)})}{N_k(t)}\right)}.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt(max(0, log(self.horizon / (self.nbArms * self.pulls[arm]))) / self.pulls[arm])

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.pulls) + np.sqrt(np.maximum(0., np.log(self.horizon / (self.nbArms * self.pulls))) / self.pulls)
        indexes[self.pulls < 1] = float('+inf')
        self.index = indexes
