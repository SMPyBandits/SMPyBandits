# -*- coding: utf-8 -*-
""" The improved kl-UCB++ policy, for one-parameter exponential distributions.
Reference: [Menard & Garivier, 2017](https://arxiv.org/abs/1702.07211)
"""

__author__ = "Lilian Besson"
__version__ = "0.5"

from math import log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .kullback import klucbBern
from .klUCB import klUCB, c


# --- Numerical functions required for the function g(n) for kl-UCB++

def logplus(x):
    """logplus(x) = max(0, log(x))."""
    return max(0., log(x))


def g(n, T, K):
    """The exploration function g(n), as defined in page 3 of the reference paper."""
    y = T / (float(K) * n)
    return max(0., log(y * (1. + max(0., log(y)) ** 2)))


def np_g(n, T, K):
    """The exploration function g(n), as defined in page 3 of the reference paper, for numpy inputs."""
    y = T / (float(K) * n)
    return np.maximum(0., np.log(y * (1. + np.maximum(0., np.log(y)) ** 2)))


class klUCBPlusPlus(klUCB):
    """ The improved kl-UCB++ policy, for one-parameter exponential distributions.
    Reference: [Menard & Garivier, 2017](https://arxiv.org/abs/1702.07211)
    """

    def __init__(self, nbArms, horizon=None, tolerance=1e-4, klucb=klucbBern, c=c, lower=0., amplitude=1.):
        super(klUCBPlusPlus, self).__init__(nbArms, tolerance=tolerance, klucb=klucb, c=c, lower=lower, amplitude=amplitude)
        self.horizon = int(horizon)  #: Parameter :math:`T` = known horizon of the experiment.

    def __str__(self):
        return r"KL-UCB++($T={}$, {}{})".format(self.horizon, "" if self.c == 1 else r"$c={:.3g}$".format(self.c), self.klucb.__name__[5:])

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

           \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
           U_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c g(N_k(t), T, K)}{N_k(t)} \right\},\\
           I_k(t) &= U_k(t).

        If rewards are in :math:`[a, b]` (default to :math:`[0, 1]`) and :math:`\mathrm{kl}(x, y)` is the Kullback-Leibler divergence between two distributions of means x and y (see :mod:`Arms.kullback`),
        and c is the parameter (default to 1), and where :math:`g(n, T, K)` is this function:

        .. math::

           g(n, T, K) := \max\left(0, \log(\frac{T}{K n} (1 + \max\left(0, \log(\frac{T}{K n})\right)^2)) \right).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return self.klucb(self.rewards[arm] / self.pulls[arm], self.c * g(self.pulls[arm], self.horizon, self.nbArms) / self.pulls[arm], self.tolerance)

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = self.klucb(self.rewards / self.pulls, self.c * np_g(self.pulls, self.horizon, self.nbArms) / self.pulls, self.tolerance)
        indexes[self.pulls < 1] = float('+inf')
        self.index = indexes
