# -*- coding: utf-8 -*-
""" The generic kl-UCB policy for one-parameter exponential distributions.
By default, it uses a Beta posterior and assumes Bernoulli arms.
Note: using log(t) + c log(log(t)) for the KL-UCB index of just log(t)
Reference: [Garivier & Cappé - COLT, 2011].
"""

__author__ = "Lilian Besson"
__version__ = "0.5"

from math import log

from .klUCB import klUCB


class klUCBloglog(klUCB):
    """ The generic kl-UCB policy for one-parameter exponential distributions.
    By default, it uses a Beta posterior and assumes Bernoulli arms.
    Note: using log(t) + c log(log(t)) for the KL-UCB index of just log(t)
    Reference: [Garivier & Cappé - COLT, 2011].
    """

    def __str__(self):
        return r"KL-UCB({}{}{})".format("" if self.c == 1 else r"$c={:.3g}$, ".format(self.c), r"$\log\log$", self.klucb.__name__[5:])

    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return self.klucb(self.rewards[arm] / self.pulls[arm], (log(self.t) + self.c * log(max(1, log(self.t)))) / self.pulls[arm], self.tolerance)
