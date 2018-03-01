# -*- coding: utf-8 -*-
"""The Thompson (Bayesian) index policy, using an average of 20 index. By default, it uses a Beta posterior.
Reference: [Thompson - Biometrika, 1933].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
from .Thompson import Thompson
from .Posterior import Beta


#: Default value of how many indexes are computed by sampling the posterior
#: for the ThompsonRobust variant.
AVERAGEON = 10


class ThompsonRobust(Thompson):
    """The Thompson (Bayesian) index policy, using an average of 20 index. By default, it uses a Beta posterior.
    Reference: [Thompson - Biometrika, 1933].
    """

    def __init__(self, nbArms, posterior=Beta, averageOn=AVERAGEON, lower=0., amplitude=1.):
        super(ThompsonRobust, self).__init__(nbArms, posterior=posterior, lower=lower, amplitude=amplitude)
        assert averageOn >= 1, "Error: invalid value for 'averageOn' parameter for ThompsonRobust, should be >= 1."  # DEBUG
        self.averageOn = averageOn  #: How many indexes are computed before averaging

    def __str__(self):
        return "%s(averageOn = %i)" % (self.__class__.__name__, self.averageOn)

    def computeIndex(self, arm):
        r""" Compute the current index for this arm, by sampling averageOn times the posterior and returning the average index.

        At time t and after :math:`N_k(t)` pulls of arm k, giving :math:`S_k(t)` rewards of 1, by sampling from the Beta posterior and averaging:

        .. math::

           I_k(t) &= \frac{1}{\mathrm{averageOn}} \sum_{i=1}^{\mathrm{averageOn}} I_k^{(i)}(t), \\
           I_k^{(i)}(t) &\sim \mathrm{Beta}(1 + S_k(t), 1 + N_k(t) - S_k(t)).
        """
        return np.mean([self.posterior[arm].sample() for _ in range(self.averageOn)])
