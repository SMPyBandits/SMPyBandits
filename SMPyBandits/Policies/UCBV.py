# -*- coding: utf-8 -*-
""" The UCB-V policy for bounded bandits, with a variance correction term.
Reference: [Audibert, Munos, & Szepesvári - Theoret. Comput. Sci., 2009].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.5"

from math import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .UCB import UCB


class UCBV(UCB):
    """ The UCB-V policy for bounded bandits, with a variance correction term.
    Reference: [Audibert, Munos, & Szepesvári - Theoret. Comput. Sci., 2009].
    """

    def __init__(self, nbArms, lower=0., amplitude=1.):
        super(UCBV, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.rewardsSquared = np.zeros(self.nbArms)  #: Keep track of squared of rewards, to compute an empirical variance

    def startGame(self):
        super(UCBV, self).startGame()
        self.rewardsSquared.fill(0)

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and of rewards squared for that arm (normalized in [0, 1])."""
        super(UCBV, self).getReward(arm, reward)
        self.rewardsSquared[arm] += ((reward - self.lower) / self.amplitude) ** 2

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

           \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
           V_k(t) &= \frac{Z_k(t)}{N_k(t)} - \hat{\mu}_k(t)^2, \\
           I_k(t) &= \hat{\mu}_k(t) + \sqrt{\frac{2 \log(t) V_k(t)}{N_k(t)}} + 3 (b - a) \frac{\log(t)}{N_k(t)}.

        Where rewards are in :math:`[a, b]`, and :math:`V_k(t)` is an estimator of the variance of rewards,
        obtained from :math:`X_k(t) = \sum_{\sigma=1}^{t} 1(A(\sigma) = k) r_k(\sigma)` is the sum of rewards from arm k,
        and :math:`Z_k(t) = \sum_{\sigma=1}^{t} 1(A(\sigma) = k) r_k(\sigma)^2` is the sum of rewards *squared*.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            mean = self.rewards[arm] / self.pulls[arm]   # Mean estimate
            variance = (self.rewardsSquared[arm] / self.pulls[arm]) - mean ** 2  # Variance estimate
            return mean + sqrt(2.0 * log(self.t) * variance / self.pulls[arm]) + 3.0 * self.amplitude * log(self.t) / self.pulls[arm]

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        means = self.rewards / self.pulls   # Mean estimate
        variances = (self.rewardsSquared / self.pulls) - means ** 2  # Variance estimate
        indexes = means + np.sqrt(2.0 * np.log(self.t) * variances / self.pulls) + 3.0 * self.amplitude * np.log(self.t) / self.pulls
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
