# -*- coding: utf-8 -*-
""" The Optimally Confident UCB (OC-UCB) policy for bounded stochastic bandits. Initial version (horizon-dependent).

- Reference: [Lattimore, 2015](https://arxiv.org/pdf/1507.07880.pdf)
- There is also a horizon-independent version, :class:`OCUCB.OCUCB`, from  [Lattimore, 2016](https://arxiv.org/pdf/1603.08661.pdf).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from math import exp, sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .OCUCB import OCUCB
except ImportError:
    from OCUCB import OCUCB

#: Default value for parameter :math:`\psi \geq 2` for OCUCBH.
PSI = 2

#: Default value for parameter :math:`\alpha \geq 2` for OCUCBH.
ALPHA = 4


# --- OCUCBH


class OCUCBH(OCUCB):
    """ The Optimally Confident UCB (OC-UCB) policy for bounded stochastic bandits. Initial version (horizon-dependent).

    - Reference: [Lattimore, 2015](https://arxiv.org/pdf/1507.07880.pdf)
    """

    def __init__(self, nbArms, horizon=None, psi=PSI, alpha=ALPHA, lower=0., amplitude=1.):
        super(OCUCBH, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert psi >= 2, "Error: parameter 'psi' for OCUCBH algorithm has to be >= 2."  # DEBUG
        self.psi = psi  #: Parameter :math:`\psi \geq 2`.
        assert alpha >= 2, "Error: parameter 'alpha' for OCUCBH algorithm has to be in >= 2."  # DEBUG
        self.alpha = alpha  #: Parameter :math:`\alpha \geq 2`.
        assert horizon > 1, "Error: parameter 'psi' for OCUCBH algorithm has to be > 1."  # DEBUG
        self.horizon = int(horizon)  #: Horizon T.

    def __str__(self):
        return r"OC-UCB-H($\alpha={:.3g}$, $\psi={:.3g}$, $T={}$)".format(self.alpha, self.psi, self.horizon)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{\alpha}{N_k(t)} \log(\frac{\psi T}{t})}.

        - Where :math:`\alpha` and :math:`\psi` are two parameters of the algorithm.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt((self.alpha / self.pulls[arm]) * log(self.psi * self.horizon / self.t) )

    # XXX Error : division by zero ?
    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     indexes = (self.rewards / self.pulls) + np.sqrt((self.alpha / self.pulls) * np.log(self.psi * self.horizon / self.t) )
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index[:] = indexes


# --- AOCUCB

class AOCUCBH(OCUCBH):
    """ The Almost Optimally Confident UCB (OC-UCB) policy for bounded stochastic bandits. Initial version (horizon-dependent).

    - Reference: [Lattimore, 2015](https://arxiv.org/pdf/1507.07880.pdf)
    """

    def __init__(self, nbArms, horizon=None, lower=0., amplitude=1.):
        super(AOCUCBH, self).__init__(nbArms, horizon=horizon, psi=2, alpha=2, lower=lower, amplitude=amplitude)

    def __str__(self):
        return r"AOC-UCB-H($T={}$)".format(self.horizon)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{2}{N_k(t)} \log(\frac{T}{N_k(t)})}.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt((2 / self.pulls[arm]) * log(self.horizon / self.pulls[arm]) )

    # XXX Error : division by zero ?
    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     indexes = (self.rewards / self.pulls) + np.sqrt((2 / self.pulls) * np.log(self.horizon / self.pulls) )
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index[:] = indexes

