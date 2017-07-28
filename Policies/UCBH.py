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
        self.horizon = int(horizon) if horizon is not None else None  #: Parameter :math:`T` = known horizon of the experiment.
        self.alpha = alpha  #: Parameter alpha

    def __str__(self):
        return r"UCB-H($T={}$, $\alpha={:.3g}$)".format(self.horizon, self.alpha)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{\alpha \log(T)}{2 N_k(t)}}.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt((self.alpha * log(self.horizon)) / (2 * self.pulls[arm]))

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.pulls) + np.sqrt((self.alpha * np.log(self.horizon)) / (2 * self.pulls))
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
