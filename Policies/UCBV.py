# -*- coding: utf-8 -*-
""" The UCB-V policy for bounded bandits, with a variance correction term.
Reference: [Audibert, Munos, & Szepesvári - Theoret. Comput. Sci., 2009].
"""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.7 $"

from math import sqrt, log
import numpy as np

from .IndexPolicy import IndexPolicy


class UCBV(IndexPolicy):
    """ The UCB-V policy for bounded bandits, with a variance correction term.
    Reference: [Audibert, Munos, & Szepesvári - Theoret. Comput. Sci., 2009].
    """

    def __init__(self, nbArms, amplitude=1.):
        super(UCBV, self).__init__(nbArms)
        self.amplitude = amplitude
        self.rewardsSquared = np.zeros(nbArms)
        self.params = 'amplitude: {}'.format(amplitude)
        self.t = -1
        self.pulls = np.zeros(self.nbArms, dtype=int)
        self.rewards = np.zeros(self.nbArms)
        self.rewardsSquared = np.zeros(self.nbArms)

    def __str__(self):
        return "UCBV"

    def startGame(self):
        self.t = 0
        self.pulls.fill(0)
        self.rewards.fill(0)
        self.rewardsSquared.fill(0)

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            mean = self.rewards[arm] / self.pulls[arm]   # Mean estimate
            variance = (self.rewardsSquared[arm] / self.pulls[arm]) - mean ** 2  # Variance estimate
            return mean + sqrt(2.0 * log(self.t) * variance / self.pulls[arm]) + 3.0 * self.amplitude * log(self.t) / self.pulls[arm]

    def getReward(self, arm, reward):
        self.t += 1
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        self.rewardsSquared[arm] += reward ** 2
