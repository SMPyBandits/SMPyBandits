# -*- coding: utf-8 -*-
""" The generic kl-UCB policy for one-parameter exponential distributions.
By default, it uses a Beta posterior.
Note: using log10(t) and not ln(t) for UCB index.
Reference: [Garivier & Cappé - COLT, 2011].
"""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.15 $"

from math import log10

from .kullback import klucbBern
from .IndexPolicy import IndexPolicy


class klUCBlog10(IndexPolicy):
    """ The generic kl-UCB policy for one-parameter exponential distributions.
    By default, it uses a Beta posterior.
    Note: using log10(t) and not ln(t) for UCB index.
    Reference: [Garivier & Cappé - COLT, 2011].
    """

    def __init__(self, nbArms, tolerance=1e-4, klucb=klucbBern, lower=0., amplitude=1.):
        super(klUCBlog10, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.c = 1.
        self.klucb = klucb
        self.tolerance = tolerance

    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return self.klucb(self.rewards[arm] / self.pulls[arm], self.c * log10(self.t) / self.pulls[arm], self.tolerance)
