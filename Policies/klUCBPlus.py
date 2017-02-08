# -*- coding: utf-8 -*-
""" The improved kl-UCB policy, for one-parameter exponential distributions.
Reference: [Cappé et al. 13](https://arxiv.org/pdf/1210.1136.pdf)
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import log

from .klUCB import klUCB


class klUCBPlus(klUCB):
    """ The improved kl-UCB policy, for one-parameter exponential distributions.
    Reference: [Cappé et al. 13](https://arxiv.org/pdf/1210.1136.pdf)
    """

    def __str__(self):
        return r"KL-UCB+({}{})".format("" if self.c == 1 else r"$c={:.3g}$, ".format(self.c), self.klucb.__name__[5:])

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return self.klucb(self.rewards[arm] / self.pulls[arm], self.c * log(self.t / self.pulls[arm]) / self.pulls[arm], self.tolerance)
