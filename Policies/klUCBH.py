# -*- coding: utf-8 -*-
""" The kl-UCB-H policy, for one-parameter exponential distributions.
Reference: [Lai 87](https://projecteuclid.org/download/pdf_1/euclid.aos/1176350495)
"""
from __future__ import division, print_function  # Python 2 compatibility

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
        self.horizon = int(horizon) if horizon is not None else None  #: Parameter :math:`T` = known horizon of the experiment.

    def __str__(self):
        return r"KLUCB-H($T={}$, {}{})".format(self.horizon, "" if self.c == 1 else r"$c={:.3g}$".format(self.c), self.klucb.__name__[5:])

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

           \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
           U_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c \log(T)}{N_k(t)} \right\},\\
           I_k(t) &= U_k(t).

        If rewards are in :math:`[a, b]` (default to :math:`[0, 1]`) and :math:`\mathrm{kl}(x, y)` is the Kullback-Leibler divergence between two distributions of means x and y (see :mod:`Arms.kullback`),
        and c is the parameter (default to 1).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return self.klucb(self.rewards[arm] / self.pulls[arm], self.c * log(self.horizon) / self.pulls[arm], self.tolerance)

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = self.klucb(self.rewards / self.pulls, self.c * np.log(self.horizon) / self.pulls, self.tolerance)
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
