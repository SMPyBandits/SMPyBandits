# -*- coding: utf-8 -*-
""" The improved kl-UCB-H policy, for one-parameter exponential distributions.
Reference: [Lai 87](https://projecteuclid.org/download/pdf_1/euclid.aos/1176350495)
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import log

from .kullback import klucbBern
from .klUCB import klUCB


class klUCBHPlus(klUCB):
    """ The improved kl-UCB-H policy, for one-parameter exponential distributions.
    Reference: [Lai 87](https://projecteuclid.org/download/pdf_1/euclid.aos/1176350495)
    """

    def __init__(self, nbArms, horizon=None, tolerance=1e-4, klucb=klucbBern, lower=0., amplitude=1.):
        super(klUCBHPlus, self).__init__(nbArms, tolerance=tolerance, klucb=klucb, lower=lower, amplitude=amplitude)
        self._horizon = horizon

    # This decorator @property makes this method an attributes, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def horizon(self):
        """ If the 'horizon' parameter was not provided, acts like the klUCBPlus policy. """
        return self.t if self._horizon is None else self._horizon

    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return self.klucb(self.rewards[arm] / self.pulls[arm], self.c * log(self.horizon / self.pulls[arm]) / self.pulls[arm], self.tolerance)
