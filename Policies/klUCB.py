# -*- coding: utf-8 -*-
""" The generic kl-UCB policy for one-parameter exponential distributions.
Reference: [Garivier & Cappé - COLT, 2011].
"""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.15 $"

from math import log
import numpy as np

from .kullback import klucbBern
from .IndexPolicy import IndexPolicy


class klUCB(IndexPolicy):
    """ The generic kl-UCB policy for one-parameter exponential distributions.
    Reference: [Garivier & cappé - COLT, 2011].
    """

    def __init__(self, nbArms,
                 amplitude=1., lower=0., tolerance=1e-4,
                 klucb=klucbBern):
        super(klUCB, self).__init__(nbArms)
        self.c = 1.
        self.amplitude = float(amplitude)
        self.lower = lower
        self.klucb = klucb
        self.tolerance = tolerance
        self.params = 'amplitude:' + repr(self.amplitude) + \
                      ', lower:' + repr(self.lower)

    def __str__(self):
        return "klUCB"

    def startGame(self):
        self.t = 0
        self.pulls = np.zeros(self.nbArms)
        self.rewards = np.zeros(self.nbArms)

    def computeIndex(self, arm):
        if self.pulls[arm] == 0:
            return float('+infinity')
        else:
            # Could adapt tolerance to the value of self.t
            return self.klucb(self.rewards[arm] / float(self.pulls[arm]), self.c * log(self.t) / float(self.pulls[arm]), self.tolerance)

    def getReward(self, arm, reward):
        self.t += 1
        self.pulls[arm] += 1
        self.rewards[arm] += (reward - self.lower) / self.amplitude
