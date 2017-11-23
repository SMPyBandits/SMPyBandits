# -*- coding: utf-8 -*-
""" The Boltzmann-Gumbel Exploration (BGE) index policy, a different formulation of the :class:`Exp3` policy with an optimally tune decreasing sequence of temperature parameters :math:`\gamma_t`.

- Reference: Section 4 of [Boltzmann Exploration Done Right, N.Cesa-Bianchi & C.Gentile & G.Lugosi & G.Neu, arXiv 2017](https://arxiv.org/pdf/1705.10257.pdf).
- It is an index policy with indexes computed from the empirical mean estimators and a random sample from a Gumbel distribution.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
import numpy.random as rn

from .IndexPolicy import IndexPolicy


#: Default constant :math:`\sigma` assuming the arm distributions are :math:`\sigma^2`-subgaussian. 1 for Bernoulli arms.
SIGMA = 1

class BoltzmannGumbel(IndexPolicy):
    r""" The Boltzmann-Gumbel Exploration (BGE) index policy, a different formulation of the :class:`Exp3` policy with an optimally tune decreasing sequence of temperature parameters :math:`\gamma_t`.

    - Reference: Section 4 of [Boltzmann Exploration Done Right, N.Cesa-Bianchi & C.Gentile & G.Lugosi & G.Neu, arXiv 2017](https://arxiv.org/pdf/1705.10257.pdf).
    - It is an index policy with indexes computed from the empirical mean estimators and a random sample from a Gumbel distribution.
    """

    def __init__(self, nbArms, C=SIGMA, lower=0., amplitude=1.):
        super(BoltzmannGumbel, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert C > 0, "Error: the C parameter for BoltzmannGumbel class has to be > 0."
        self.C = C

    def __str__(self):
        return r"BoltzmannGumbel($\alpha={:.3g}$)".format(self.C)

    def computeIndex(self, arm):
        r""" Take a random index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

           I_k(t) &= \frac{X_k(t)}{N_k(t)} + \beta_k(t) Z_k(t), \\
           \text{where}\;\; \beta_k(t) &:= \sqrt{C^2 / N_k(t)}, \\
           \text{and}\;\; Z_k(t) &\sim \mathrm{Gumbel}(0, 1).

        Where :math:`\mathrm{Gumbel}(0, 1)` is the standard Gumbel distribution.
        See [Numpy documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.gumbel.html#numpy.random.gumbel) or [Wikipedia page](https://en.wikipedia.org/wiki/Gumbel_distribution) for more details.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            beta_k_t = np.sqrt(self.C ** 2 / self.pulls[arm])
            z_k_t = rn.gumbel(0, 1)
            return (self.rewards[arm] / self.pulls[arm]) + beta_k_t * z_k_t

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        beta_t = np.sqrt(self.C ** 2 / self.pulls)
        z_t = rn.gumbel(0, 1, self.nbArms)  # vector samples
        indexes = (self.rewards / self.pulls) + beta_t * z_t
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
