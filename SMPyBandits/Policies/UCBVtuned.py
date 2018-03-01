# -*- coding: utf-8 -*-
""" The UCBV-Tuned policy for bounded bandits, with a tuned variance correction term.
Reference: [Auer et al. 02].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.5"

from math import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .UCBV import UCBV


class UCBVtuned(UCBV):
    """ The UCBV-Tuned policy for bounded bandits, with a tuned variance correction term.
    Reference: [Auer et al. 02].
    """

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

           \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
           V_k(t) &= \frac{Z_k(t)}{N_k(t)} - \hat{\mu}_k(t)^2, \\
           V'_k(t) &= V_k(t) + \sqrt{\frac{2 \log(t)}{N_k(t)}}, \\
           I_k(t) &= \hat{\mu}_k(t) + \sqrt{\frac{\log(t) V'_k(t)}{N_k(t)}}.

        Where :math:`V'_k(t)` is an other estimator of the variance of rewards,
        obtained from :math:`X_k(t) = \sum_{\sigma=1}^{t} 1(A(\sigma) = k) r_k(\sigma)` is the sum of rewards from arm k,
        and :math:`Z_k(t) = \sum_{\sigma=1}^{t} 1(A(\sigma) = k) r_k(\sigma)^2` is the sum of rewards *squared*.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            mean = self.rewards[arm] / self.pulls[arm]   # Mean estimate
            variance = (self.rewardsSquared[arm] / self.pulls[arm]) - mean ** 2  # Variance estimate
            # Correct variance estimate
            variance += sqrt(2.0 * log(self.t) / self.pulls[arm])
            return mean + sqrt(log(self.t) * variance / self.pulls[arm])

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        means = self.rewards / self.pulls   # Mean estimate
        variances = (self.rewardsSquared / self.pulls) - means ** 2  # Variance estimate
        variances += np.sqrt(2.0 * np.log(self.t) / self.pulls)
        indexes = means + np.sqrt(np.log(self.t) * variances / self.pulls)
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
