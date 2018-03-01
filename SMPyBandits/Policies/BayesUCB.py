# -*- coding: utf-8 -*-
""" The Bayes-UCB policy. By default, it uses a Beta posterior.
Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012]
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann, Lilian Besson"
__version__ = "0.5"

from .BayesianIndexPolicy import BayesianIndexPolicy


class BayesUCB(BayesianIndexPolicy):
    """ The Bayes-UCB policy. By default, it uses a Beta posterior.
      Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012].
    """

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, giving :math:`S_k(t)` rewards of 1, by taking the :math:`1 - \frac{1}{t}` quantile from the Beta posterior:

        .. math:: I_k(t) = \mathrm{Quantile}\left(\mathrm{Beta}(1 + S_k(t), 1 + N_k(t) - S_k(t)), 1 - \frac{1}{t}\right).
        """
        return self.posterior[arm].quantile(1. - 1. / (1 + self.t))
