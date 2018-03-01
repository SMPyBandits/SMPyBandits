# -*- coding: utf-8 -*-
""" The Optimally Confident UCB (OC-UCB) policy for bounded stochastic bandits, with sub-Gaussian noise.

- Reference: [Lattimore, 2016](https://arxiv.org/pdf/1603.08661.pdf).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

from math import exp, sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .UCB import UCB

#: Default value for parameter :math:`\eta > 1` for OCUCB.
ETA = 2

#: Default value for parameter :math:`\rho \in (1/2, 1]` for OCUCB.
RHO = 1


class OCUCB(UCB):
    """ The Optimally Confident UCB (OC-UCB) policy for bounded stochastic bandits, with sub-Gaussian noise.

    - Reference: [Lattimore, 2016](https://arxiv.org/pdf/1603.08661.pdf).
    """

    def __init__(self, nbArms, eta=ETA, rho=RHO, lower=0., amplitude=1.):
        super(OCUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert eta > 1, "Error: parameter 'eta' for OCUCB algorithm has to be > 1."  # DEBUG
        self.eta = eta  #: Parameter :math:`\eta > 1`.
        assert 0.5 < rho <= 1, "Error: parameter 'rho' for OCUCB algorithm has to be in (1/2, 1]."  # DEBUG
        self.rho = rho  #: Parameter :math:`\rho \in (1/2, 1]`.

    def __str__(self):
        return r"OC-UCB($\eta:{:.3g}$, $\rho:{:.3g}$)".format(self.eta, self.rho)

    def _Bterm(self, k):
        r""" Compute the extra term :math:`B_k(t)` as follows:

        .. math::

           B_k(t) &= \max\Big\{ \exp(1), \log(t), t \log(t) / C_k(t) \Big\},\\
           \text{where}\; C_k(t) &= \sum_{j=1}^{K} \min\left\{ T_k(t), T_j(t)^{\rho} T_k(t)^{1 - \rho} \right\}
        """
        t = self.t
        T_ = self.pulls
        C_kt = sum(min(T_[k], (T_[j] ** self.rho) * (T_[k] ** (1. - self.rho))) for j in range(self.nbArms))
        return max([exp(1), log(t), t * log(t) / C_kt])

    def _Bterms(self):
        r""" Compute all the extra terms, :math:`B_k(t)` for each arm k, in a naive manner, not optimized to be vectorial, but it works."""
        return np.array([self._Bterm(k) for k in range(self.nbArms)])

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{2 \eta \log(B_k(t))}{N_k(t)}}.

        - Where :math:`\eta` is a parameter of the algorithm,
        - And :math:`B_k(t)` is the additional term defined above.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt(2 * self.eta * log(self._Bterm(arm)) / self.pulls[arm])

    # FIXME it does not work so far?! Why?!!
    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     indexes = (self.rewards / self.pulls) + np.sqrt(2 * self.eta * np.log(self._Bterms()) / self.pulls)
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index[:] = indexes
