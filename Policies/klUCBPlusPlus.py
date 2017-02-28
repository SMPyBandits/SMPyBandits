# -*- coding: utf-8 -*-
""" The improved kl-UCB++ policy, for one-parameter exponential distributions.
Reference: [Menard & Garivier, 2017](https://arxiv.org/abs/1702.07211)
"""

__author__ = "Lilian Besson"
__version__ = "0.5"

from math import log

from .kullback import klucbBern
from .klUCB import klUCB, c


# --- Numerical functions required for the function g(n) for kl-UCB++

def logplus(x):
    """logplus(x) = max(log(x), 0)"""
    return max(log(x), 0)


def g(n, T, K):
    """The exploration function g(n), as defined in page 3 of the reference paper."""
    y = T / float(K * n)
    return logplus(y * (1 + logplus(y) ** 2))



class klUCBPlusPlus(klUCB):
    """ The improved kl-UCB++ policy, for one-parameter exponential distributions.
    Reference: [Menard & Garivier, 2017](https://arxiv.org/abs/1702.07211)
    """

    def __init__(self, nbArms, horizon=None, tolerance=1e-4, klucb=klucbBern, c=c, lower=0., amplitude=1.):
        super(klUCBPlusPlus, self).__init__(nbArms, tolerance=tolerance, klucb=klucb, c=c, lower=lower, amplitude=amplitude)
        self._horizon = horizon

    def __str__(self):
        return r"KL-UCB++({}{})".format("" if self.c == 1 else r"$c={:.3g}$".format(self.c), self.klucb.__name__[5:])

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def horizon(self):
        """ If the 'horizon' parameter was not provided, acts like the klUCBPlus policy. """
        return self.t if self._horizon is None else self._horizon

    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return self.klucb(self.rewards[arm] / self.pulls[arm], self.c * g(self.pulls[arm], self.horizon, self.nbArms), self.tolerance)
