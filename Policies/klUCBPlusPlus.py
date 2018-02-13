# -*- coding: utf-8 -*-
""" The improved kl-UCB++ policy, for one-parameter exponential distributions.
Reference: [Menard & Garivier, ALT 2017](https://hal.inria.fr/hal-01475078)
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.5"

from math import log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .kullback import klucbBern
from .klUCB import klUCB, c

try:
    from .usenumba import jit  # Import numba.jit or a dummy jit(f)=f
except (ValueError, SystemError):
    from usenumba import jit  # Import numba.jit or a dummy jit(f)=f


# --- Numerical functions required for the function g(n) for kl-UCB++

# @jit
def logplus(x):
    """..math:: \log^+(x) := \max(0, \log(x))."""
    return max(0., log(x))


# @jit
def g(t, T, K):
    r"""The exploration function g(t) (for t current time, T horizon, K nb arms), as defined in page 3 of the reference paper.

    .. math::

        g(t, T, K) &:= \log^+(y (1 + \log^+(y)^2)),\\
        y &:= \frac{T}{K t}.
    """
    y = T / (K * t)
    return max(0., log(y * (1. + max(0., log(y)) ** 2)))


# @jit
def np_g(t, T, K):
    r"""The exploration function g(t) (for t current time, T horizon, K nb arms), as defined in page 3 of the reference paper, for numpy vectorized inputs.

    .. math::

        g(t, T, K) &:= \log^+(y (1 + \log^+(y)^2)),\\
        y &:= \frac{T}{K t}.
    """
    y = T / (K * t)
    return np.maximum(0., np.log(y * (1. + np.maximum(0., np.log(y)) ** 2)))


class klUCBPlusPlus(klUCB):
    """ The improved kl-UCB++ policy, for one-parameter exponential distributions.
    Reference: [Menard & Garivier, ALT 2017](https://hal.inria.fr/hal-01475078)
    """

    def __init__(self, nbArms, horizon=None, tolerance=1e-4, klucb=klucbBern, c=c, lower=0., amplitude=1.):
        super(klUCBPlusPlus, self).__init__(nbArms, tolerance=tolerance, klucb=klucb, c=c, lower=lower, amplitude=amplitude)
        self.nbArms = float(self.nbArms)  # Just speed up type casting by forcing it to be a float
        self.horizon = int(horizon) if horizon is not None else None  #: Parameter :math:`T` = known horizon of the experiment.

    def __str__(self):
        name = "" if self.klucb.__name__[5:] == "Bern" else ", " + self.klucb.__name__[5:]
        complement = "$T={}${}{}".format(self.horizon, name, "" if self.c == 1 else r", $c={:.3g}$".format(self.c))
        return r"KLUCB{}({})".format("$^{++}$", complement)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

           \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
           U_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c g(N_k(t), T, K)}{N_k(t)} \right\},\\
           I_k(t) &= U_k(t).

        If rewards are in :math:`[a, b]` (default to :math:`[0, 1]`) and :math:`\mathrm{kl}(x, y)` is the Kullback-Leibler divergence between two distributions of means x and y (see :mod:`Arms.kullback`),
        and c is the parameter (default to 1), and where :math:`g(t, T, K)` is this function:

        .. math::

            g(t, T, K) &:= \log^+(y (1 + \log^+(y)^2)),\\
            y &:= \frac{T}{K t}.
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
        self.index[:] = indexes
