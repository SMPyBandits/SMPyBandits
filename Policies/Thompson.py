# -*- coding: utf-8 -*-
""" The Thompson (Bayesian) index policy. By default, it uses a Beta posterior.
Reference: [Thompson - Biometrika, 1933].
"""

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann"
__version__ = "$Revision: 1.9 $"

from .BayesianIndexPolicy import BayesianIndexPolicy


class Thompson(BayesianIndexPolicy):
    """The Thompson (Bayesian) index policy. By default, it uses a Beta posterior.
    Reference: [Thompson - Biometrika, 1933].
    """

    def computeIndex(self, arm):
        return self.posterior[arm].sample()
