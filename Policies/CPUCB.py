# -*- coding: utf-8 -*-
""" The Copper-Pearson UCB policy for bounded bandits.
Reference: [Garivier & Cappé, COLT 2011](https://arxiv.org/pdf/1102.2490.pdf).
"""

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.6"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .UCB import UCB

#: Default value for the parameter c for CP-UCB
C = 1.01


class CPUCB(UCB):
    """ The Copper-Pearson UCB policy for bounded bandits.
    Reference: [Garivier & Cappé, COLT 2011].
    """

    def __init__(self, nbArms, c=C, lower=0., amplitude=1.):
        super(CPUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.c = c  #: Parameter c for the CP-UCB formula (see below)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{2 \log(t)}{N_k(t)}}.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # FIXME what is binofit in MATLAB ?
            index = binofit(self.rewards[arm], self.pulls[arm], 1. / (self.t ** self.c))[1]
            return index[:, 1]

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        # FIXME what is binofit in MATLAB ?
        # indexes = (self.rewards / self.pulls) + np.sqrt((2 * np.log(self.t)) / self.pulls)
        indexes = binofit(self.rewards, self.pulls, 1. / (self.t ** self.c))[1]
        indexes[self.pulls < 1] = float('+inf')
        self.index = indexes
