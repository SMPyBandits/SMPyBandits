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
        self.pulls = np.zeros(nbArms)
        self.rewards = np.zeros(nbArms)
        self.rewardsCorrected = np.zeros(nbArms)
        self.t = -1
        self.params = 'amplitude: ' + repr(amplitude) + ', lower: ' + repr(lower)

    def __str__(self):
        return "UCBV"

    def startGame(self):
        self.t = 0
        self.pulls = np.zeros(self.nbArms)
        self.rewards = np.zeros(self.nbArms)
        self.rewardsCorrected = np.zeros(self.nbArms)

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+infinity')
        else:
            m = self.rewards[arm] / self.pulls[arm]
            v = self.rewardsCorrected[arm] / self.pulls[arm] - m * m
            return m + sqrt(2 * log(self.t) * v / self.pulls[arm]) + 3 * self.amplitude * log(self.t) / self.pulls[arm]

    def getReward(self, arm, reward):
        self.t += 1
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        self.rewardsCorrected[arm] += reward**2
