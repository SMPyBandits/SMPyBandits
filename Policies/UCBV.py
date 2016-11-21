# -*- coding: utf-8 -*-
""" The UCB-V policy for bounded bandits.
Reference: [Audibert, Munos, & Szepesvári - Theoret. Comput. Sci., 2009].
"""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.7 $"

from math import sqrt, log
import numpy as np
from .IndexPolicy import IndexPolicy


class UCBV(IndexPolicy):
    """ The UCB-V policy for bounded bandits.
    Reference: [Audibert, Munos, & Szepesvári - Theoret. Comput. Sci., 2009].
    """

    def __init__(self, nbArms, amplitude=1., lower=0.):
        self.nbArms = nbArms
        self.amplitude = amplitude
        self.lower = lower
        self.nbpulls = np.zeros(nbArms)
        self.cumReward = np.zeros(nbArms)
        self.cumReward2 = np.zeros(nbArms)
        self.t = -1
        self.params = 'amplitude: ' + repr(amplitude) + ', lower: ' + repr(lower)

    def __str__(self):
        return "UCBV"

    def startGame(self):
        self.t = 0
        self.nbpulls = np.zeros(self.nbArms)
        self.cumReward = np.zeros(self.nbArms)
        self.cumReward2 = np.zeros(self.nbArms)

    def computeIndex(self, arm):
        if self.nbpulls[arm] < 2:
            return float('+infinity')
        else:
            m = self.cumReward[arm] / self.nbpulls[arm]
            v = self.cumReward2[arm] / self.nbpulls[arm] - m * m
            return m + sqrt(2 * log(self.t) * v / self.nbpulls[arm]) + 3 * self.amplitude * log(self.t) / self.nbpulls[arm]

    def getReward(self, arm, reward):
        self.t += 1
        self.nbpulls[arm] += 1
        self.cumReward[arm] += reward
        self.cumReward2[arm] += reward**2
