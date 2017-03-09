# -*- coding: utf-8 -*-
""" The Bayes-UCB policy. By default, it uses a Beta posterior.
Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012]
"""

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann, Lilian Besson"
__version__ = "0.5"

from .BayesianIndexPolicy import BayesianIndexPolicy


class BayesUCB(BayesianIndexPolicy):
    """ The Bayes-UCB policy. By default, it uses a Beta posterior.
      Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012].
    """

    def computeIndex(self, arm):
        return self.posterior[arm].quantile(1. - 1. / (1 + self.t))
