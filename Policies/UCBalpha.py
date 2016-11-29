# -*- coding: utf-8 -*-
""" The UCB1 (UCB-alpha) index policy.
Reference: [Auer et al. 02].
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np


class UCBalpha(object):
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
        # XXX trying to randomize the order of the initial visit to each arm; as this determinism breaks its habitility to play efficiently in multi-players games
        self._random_offset = np.random.randint(nbArms)  # Exploration starts with this arm
        self.params = 'alpha: {}, offset: {}'.format(self.alpha, self._random_offset)

    def __str__(self):
        return "UCB1 (" + self.params + ")"

    def startGame(self):
        self.t = 0
        self.pulls = np.zeros(self.nbArms)
        self.rewards = np.zeros(self.nbArms)

    def choice(self):
        if self.t < self.nbArms:  # Force to first visit each arm
            arm = (self.t + self._random_offset) % self.nbArms
        else:
            # print(self.rewards, self.pulls, self.t)  # DEBUG
            arm = np.argmax(self.rewards / self.pulls + np.sqrt((self.alpha * np.log(self.t)) / (2 * self.pulls)))
            # TODO should be uniformly chosen if more than one arm has the highest index, but that's unlikely
        self.pulls[arm] += 1
        return arm

    def getReward(self, arm, reward):
        self.t += 1
        # self.pulls[arm] += 1  # XXX why is it not here?
        self.rewards[arm] += reward
