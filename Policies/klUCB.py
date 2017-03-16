# -*- coding: utf-8 -*-
""" The generic KL-UCB policy for one-parameter exponential distributions.
By default, it uses a Beta posterior and assumes Bernoulli arms.
Reference: [Garivier & Cappé - COLT, 2011](https://arxiv.org/pdf/1102.2490.pdf).
"""

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.5"

from math import log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .kullback import klucbBern
from .IndexPolicy import IndexPolicy

# Default value for the constant c used in the computation of KL-UCB index
# c = 3.  # as suggested in the Theorem 1 in https://arxiv.org/pdf/1102.2490.pdf
c = 1.  # default value, as it was in pymaBandits v1.0


class klUCB(IndexPolicy):
    """ The generic KL-UCB policy for one-parameter exponential distributions.
    By default, it uses a Beta posterior and assumes Bernoulli arms.
    Reference: [Garivier & Cappé - COLT, 2011](https://arxiv.org/pdf/1102.2490.pdf).
    """

    def __init__(self, nbArms, tolerance=1e-4, klucb=klucbBern, c=c, lower=0., amplitude=1.):
        super(klUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.c = c
        self.klucb = klucb
        self.tolerance = tolerance

    def __str__(self):
        return r"KL-UCB({}{})".format("" if self.c == 1 else r"$c={:.3g}$".format(self.c), self.klucb.__name__[5:])

    def computeIndex(self, arm):
        """ Compute the current index for this arm."""
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return self.klucb(self.rewards[arm] / self.pulls[arm], self.c * log(self.t) / self.pulls[arm], self.tolerance)

    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     # FIXME klucb does not accept vectorial inputs, right?
    #     indexes = self.klucb(self.rewards / self.pulls, self.c * np.log(self.t) / self.pulls, self.tolerance)
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index = indexes
