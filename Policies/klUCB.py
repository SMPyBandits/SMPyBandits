# -*- coding: utf-8 -*-
""" The generic kl-UCB policy for one-parameter exponential distributions.
Reference: [Garivier & cappé - COLT, 2011].
"""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.15 $"

from math import log

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
        self.nbDraws = dict()
        self.cumReward = dict()
        self.klucb = klucb
        self.tolerance = tolerance
        self.params = 'amplitude:' + repr(self.amplitude) + \
                      ', lower:' + repr(self.lower)
        self.t = -1

    def __str__(self):
        return "klUCB"

    def startGame(self):
        self.t = 1
        for arm in range(self.nbArms):
            self.nbDraws[arm] = 0
            self.cumReward[arm] = 0.0

    def computeIndex(self, arm):
        if self.nbDraws[arm] == 0:
            return float('+infinity')
        else:
            # Could adapt tolerance to the value of self.t
            return self.klucb(self.cumReward[arm] / float(self.nbDraws[arm]), self.c * log(self.t) / float(self.nbDraws[arm]), self.tolerance)

    def getReward(self, arm, reward):
        self.nbDraws[arm] += 1
        self.cumReward[arm] += (reward - self.lower) / self.amplitude
        self.t += 1
