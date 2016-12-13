# -*- coding: utf-8 -*-
""" The UCB-opt policy for bounded bandits, with a min(1, sqrt(...)) term.
Reference: [Anandkumar et al., 2010].
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import sqrt, log

from .UCB import UCB


class UCBopt(UCB):
    """ The UCB-opt policy for bounded bandits, with a min(1, sqrt(...)) term.
    Reference: [Anandkumar et al., 2010].
    """

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            mean = self.rewards[arm] / self.pulls[arm]   # Mean estimate
            return mean + min(1, sqrt(log(self.t) / (2 * self.pulls[arm])))
