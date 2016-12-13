# -*- coding: utf-8 -*-
""" The UCB-V policy for bounded bandits, with a variance correction term.
Reference: [Audibert, Munos, & Szepesvári - Theoret. Comput. Sci., 2009].
"""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.7 $"

from math import sqrt, log
import numpy as np

from .UCB import UCB


class UCBV(UCB):
    """ The UCB-V policy for bounded bandits, with a variance correction term.
    Reference: [Audibert, Munos, & Szepesvári - Theoret. Comput. Sci., 2009].
    """

    def __init__(self, nbArms, lower=0., amplitude=1.):
        super(UCBV, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.rewardsSquared = np.zeros(self.nbArms)

    def startGame(self):
        super(UCBV, self).startGame()
        self.rewardsSquared.fill(0)

    def getReward(self, arm, reward):
        super(UCBV, self).getReward(arm, reward)
        self.rewardsSquared[arm] += ((reward - self.lower) / self.amplitude) ** 2

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            mean = self.rewards[arm] / self.pulls[arm]   # Mean estimate
            variance = (self.rewardsSquared[arm] / self.pulls[arm]) - mean ** 2  # Variance estimate
            return mean + sqrt(2.0 * log(self.t) * variance / self.pulls[arm]) + 3.0 * self.amplitude * log(self.t) / self.pulls[arm]
