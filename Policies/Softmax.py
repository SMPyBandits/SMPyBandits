# -*- coding: utf-8 -*-
""" The Boltzmann Exploration (Softmax) index policy.
Reference: [http://www.cs.mcgill.ca/~vkules/bandits.pdf §2.1].
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np

temperature = 1


class Softmax(object):
    """The Boltzmann Exploration (Softmax) index policy.
    Reference: [http://www.cs.mcgill.ca/~vkules/bandits.pdf §2.1].
    """

    def __init__(self, nbArms, temperature=temperature):
        self.nbArms = nbArms
        assert temperature > 0, "Error: the temperature parameter for Softmax class has to be > 0."
        self.temperature = temperature
        self.params = "temperature:" + repr(temperature)
        self.rewards = np.zeros(nbArms)
        self.pulls = np.zeros(nbArms, dtype=int)
        self.t = -1

    def __str__(self):
        return "Softmax ({})".format(self.params)

    def startGame(self):
        self.t = 0
        self.rewards.fill(0)
        self.pulls.fill(0)

    def choice(self):
        # Force to first visit each arm once in the first steps
        if self.t < self.nbArms:
            arm = self.t % self.nbArms
            self.pulls[arm] += 1
        else:
            trusts = np.exp(self.rewards / (self.temperature * self.pulls))
            trusts /= np.sum(trusts)
            arm = np.random.choice(self.nbArms, p=trusts)
            self.pulls[arm] += 1  # XXX why is it not here?
        return arm

    def getReward(self, arm, reward):
        self.rewards[arm] += reward
        # self.pulls[arm] += 1  # XXX why is it not here?
        self.t += 1
