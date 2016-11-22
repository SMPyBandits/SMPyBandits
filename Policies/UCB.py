# -*- coding: utf-8 -*-
""" The UCB index policy.
Reference: [Lai & Robbins, 1985].
"""

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann"
__version__ = "$Revision: 1.9 $"

import numpy as np
from .IndexPolicy import IndexPolicy


class UCB(IndexPolicy):
    """ The UCB index policy.
    Reference: [Lai & Robbins, 1985].
    """

    def __init__(self, nbArms):
        # self.arms = arms
        self.nbArms = nbArms
        # self.budgets = np.asarray([arm.budget for arm in arms])
        self.pulls = np.zeros(nbArms)
        self.rewards = np.zeros(nbArms)
        self.t = -1
        self.params = ''

    def __str__(self):
        return "UCB"

    def startGame(self):
        self.t = 0
        # self.budgets = np.asarray([arm.budget for arm in self.arms])
        self.pulls = np.zeros(self.nbArms)
        self.rewards = np.zeros(self.nbArms)

    def choice(self):
        if self.t < self.nbArms:  # Force to first visit each arm
            arm = self.t % self.nbArms
            self.pulls[arm] += 1
        else:
            # print(self.rewards, self.pulls, self.t)
            arm = np.argmax(self.rewards / self.pulls + np.sqrt((2 * np.log(self.t)) / self.pulls))
            # XXX should be uniformly chosen if more than one arm has the highest index
        return arm

    def getReward(self, arm, reward):
        self.t += 1
        # self.pulls[arm] += 1  # XXX why is it not here?
        self.rewards[arm] += reward
