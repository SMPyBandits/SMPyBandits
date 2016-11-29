# -*- coding: utf-8 -*-
""" The UCB index policy.
Reference: [Lai & Robbins, 1985].
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np


class UCB(object):
    """ The UCB index policy.
    Reference: [Lai & Robbins, 1985].
    """

    def __init__(self, nbArms):
        self.nbArms = nbArms
        self.pulls = np.zeros(nbArms)
        self.rewards = np.zeros(nbArms)
        self.t = -1
        self.params = ''
        # XXX trying to randomize the order of the initial visit to each arm; as this determinism breaks its habitility to play efficiently in multi-players games
        self._random_offset = np.random.randint(nbArms)  # Exploration starts with this arm
        # self.params = 'offset: {}'.format(self._random_offset)

    def __str__(self):
        return "UCB"

    def startGame(self):
        self.t = 0
        self.pulls = np.zeros(self.nbArms)
        self.rewards = np.zeros(self.nbArms)

    def choice(self):
        if self.t < self.nbArms:  # Force to first visit each arm
            arm = (self.t + self._random_offset) % self.nbArms
        else:
            # print(self.rewards, self.pulls, self.t)  # DEBUG
            arm = np.argmax(self.rewards / self.pulls + np.sqrt((2 * np.log(self.t)) / self.pulls))
            # TODO should be uniformly chosen if more than one arm has the highest index, but that's unlikely
        self.pulls[arm] += 1
        return arm

    def getReward(self, arm, reward):
        self.t += 1
        # self.pulls[arm] += 1  # XXX why is it not here?
        self.rewards[arm] += reward
