# -*- coding: utf-8 -*-
""" The Thompson (Bayesian) index policy.
Reference: [Thompson - Biometrika, 1933].
"""

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann"
__version__ = "$Revision: 1.9 $"

from .IndexPolicy import IndexPolicy
from .Beta import Beta


class Thompson(IndexPolicy):
    """The Thompson (Bayesian) index policy.
    Reference: [Thompson - Biometrika, 1933].
    """

    def __init__(self, nbArms, posterior=Beta):
        super(Thompson, self).__init__(nbArms)
        self.posterior = [None] * nbArms  # List instead of dict, quicker access
        for arm in range(self.nbArms):
            self.posterior[arm] = posterior()

    def __str__(self):
        return "Thompson"

    def startGame(self):
        self.t = 0
        for arm in range(self.nbArms):
            self.posterior[arm].reset()

    def getReward(self, arm, reward):
        self.posterior[arm].update(reward)
        self.t += 1

    def computeIndex(self, arm):
        return self.posterior[arm].sample()
