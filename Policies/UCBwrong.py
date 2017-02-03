# -*- coding: utf-8 -*-
""" The UCBwrong policy for bounded bandits, like UCB but with a typo on the estimator of means.

One paper of W.Jouini, C.Moy and J.Palicot from 2009 contained this typo, I reimplemented it just to check that:

- its performance is worse than simple UCB
- but not that bad...
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import sqrt, log

from .IndexPolicy import IndexPolicy


class UCBwrong(IndexPolicy):
    """ The UCBwrong policy for bounded bandits, like UCB but with a typo on the estimator of means.

    One paper of W.Jouini, C.Moy and J.Palicot from 2009 contained this typo, I reimplemented it just to check that:

    - its performance is worse than simple UCB
    - but not that bad...
    """

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            mean = self.rewards[arm] / self.t   # XXX Volontary typo, wrong mean estimate
            return mean + sqrt((2 * log(self.t)) / self.pulls[arm])
