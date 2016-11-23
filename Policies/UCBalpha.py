# -*- coding: utf-8 -*-
""" The UCB1 (UCB-alpha) index policy.
Reference: [Auer et al. 02].
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np


class UCBalpha():
    """ The UCB1 (UCB-alpha) index policy.
    Reference: [Auer et al. 02].
    """

    def __init__(self, nbArms, alpha=4):
        self.nbArms = nbArms
        assert alpha > 0, "Error: the alpha parameter for UCBalpha class has to be > 0."
        self.alpha = alpha
        self.pulls = np.zeros(nbArms)
        self.rewards = np.zeros(nbArms)
        self.t = -1
        self.params = 'alpha:' + repr(alpha)

    def __str__(self):
        return "UCB1 (" + self.params + ")"

    def startGame(self):
        self.t = 0
        self.pulls = np.zeros(self.nbArms)
        self.rewards = np.zeros(self.nbArms)

    def choice(self):
        if self.t < self.nbArms:
            arm = self.t % self.nbArms
        else:
            # print(self.rewards, self.pulls, self.t)
            arm = np.argmax(self.rewards / self.pulls + np.sqrt((self.alpha * np.log(self.t)) / (2 * self.pulls)))
            # XXX should be uniformly chosen if more than one arm has the highest index
        self.pulls[arm] += 1
        return arm

    def getReward(self, arm, reward):
        self.t += 1
        # self.pulls[arm] += 1  # XXX why is it not here?
        self.rewards[arm] += reward
