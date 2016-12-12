# -*- coding: utf-8 -*-
""" The UCB-+ policy for bounded bandits, with a small trick on the index.
Reference: [Auer et al. 02].
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import sqrt, log
import numpy as np

from .UCBV import UCBV


class UCBPlus(UCBV):
    """ The UCB-+ policy for bounded bandits, with a small trick on the index.
    Reference: [Auer et al. 02].
    """

    def __init__(self, nbArms):
        super(UCBPlus, self).__init__(nbArms)

    def __str__(self):
        return "UCBPlus"

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            mean = self.rewards[arm] / self.pulls[arm]   # Mean estimate
            return mean + sqrt(max(0, log(self.t / (self.pulls[arm]))) / (2 * self.pulls[arm]))
