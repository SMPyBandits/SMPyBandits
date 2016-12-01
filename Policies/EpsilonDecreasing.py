# -*- coding: utf-8 -*-
""" The epsilon-decreasing random policy.
Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np
import random

epsilon = 0.1
decreasingRate = 1e-6


class EpsilonDecreasing(object):
    """ The epsilon-decreasing random policy.
    Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, epsilon=epsilon, decreasingRate=decreasingRate):
        self.nbArms = nbArms
        assert 0 <= epsilon <= 1, "Error: the epsilon parameter for EpsilonDecreasing class has to be in [0, 1]."
        self.epsilon = epsilon
        assert decreasingRate > 0, "Error: the decreasingRate parameter for EpsilonDecreasing class has to be > 0."
        self.decreasingRate = decreasingRate
        self.rewards = np.zeros(nbArms)
        self.params = ''
        self.t = -1

    def __str__(self):
        return "EpsilonDecreasing"

    def startGame(self):
        self.t = 0
        self.rewards.fill(0)

    def choice(self):
        if random.random() < self.epsilon * np.exp(- self.t * decreasingRate):
            # Proba epsilon : explore
            arm = random.randint(0, self.nbArms - 1)
        else:
            # Proba 1-epsilon : exploit
            arm = np.argmax(self.rewards)
        return arm

    def getReward(self, arm, reward):
        self.rewards[arm] += reward
        self.t += 1
