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
        self.c = 1.
        self.nbArms = nbArms
        self.amplitude = float(amplitude)
        self.lower = lower
        self.nbpulls = np.zeros(nbArms)
        self.cumReward = np.zeros(nbArms)
        self.klucb = klucb
        self.tolerance = tolerance
        self.params = 'amplitude:' + repr(self.amplitude) + \
                      ', lower:' + repr(self.lower)
        self.t = -1

    def __str__(self):
        return "klUCB"

    def startGame(self):
        self.t = 0
        self.nbpulls = np.zeros(self.nbArms)
        self.cumReward = np.zeros(self.nbArms)

    def computeIndex(self, arm):
        if self.nbpulls[arm] == 0:
            return float('+infinity')
        else:
            # Could adapt tolerance to the value of self.t
            return self.klucb(self.cumReward[arm] / float(self.nbpulls[arm]), self.c * log(self.t) / float(self.nbpulls[arm]), self.tolerance)

    def getReward(self, arm, reward):
        self.t += 1
        self.nbpulls[arm] += 1
        self.cumReward[arm] += (reward - self.lower) / self.amplitude
