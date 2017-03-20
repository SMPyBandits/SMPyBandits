# -*- coding: utf-8 -*-
""" The kl-UCB-H policy, for one-parameter exponential distributions.
Reference: [Lai 87](https://projecteuclid.org/download/pdf_1/euclid.aos/1176350495)
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .kullback import klucbBern
from .klUCB import klUCB, c


class klUCBH(klUCB):
    """ The kl-UCB-H policy, for one-parameter exponential distributions.
    Reference: [Lai 87](https://projecteuclid.org/download/pdf_1/euclid.aos/1176350495)
    """

    def __init__(self, nbArms, horizon=None, tolerance=1e-4, klucb=klucbBern, c=c, lower=0., amplitude=1.):
        super(klUCBH, self).__init__(nbArms, tolerance=tolerance, klucb=klucb, c=c, lower=lower, amplitude=amplitude)
        self._horizon = horizon

    def __str__(self):
        return r"KL-UCB-H({}{})".format("" if self.c == 1 else r"$c={:.3g}$".format(self.c), self.klucb.__name__[5:])

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def horizon(self):
        """ If the 'horizon' parameter was not provided, acts like the klUCBPlus policy. """
        return self.t if self._horizon is None else self._horizon

    def computeIndex(self, arm):
        """ Compute the current index for this arm."""
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return self.klucb(self.rewards[arm] / self.pulls[arm], self.c * log(self.horizon) / self.pulls[arm], self.tolerance)

    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     # FIXME klucb does not accept vectorial inputs, right?
    #     indexes = self.klucb(self.rewards / self.pulls, self.c * np.log(self.horizon) / self.pulls, self.tolerance)
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index = indexes
