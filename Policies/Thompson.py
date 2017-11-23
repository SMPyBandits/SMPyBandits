# -*- coding: utf-8 -*-
""" The Thompson (Bayesian) index policy. By default, it uses a Beta posterior.
Reference: [Thompson - Biometrika, 1933].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann, Lilian Besson"
__version__ = "0.5"

from .BayesianIndexPolicy import BayesianIndexPolicy


class Thompson(BayesianIndexPolicy):
    """The Thompson (Bayesian) index policy. By default, it uses a Beta posterior.
    Reference: [Thompson - Biometrika, 1933].
    """

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, giving :math:`S_k(t)` rewards of 1, by sampling from the Beta posterior:

        .. math:: I_k(t) \sim \mathrm{Beta}(1 + S_k(t), 1 + N_k(t) - S_k(t)).
        """
        return self.posterior[arm].sample()
