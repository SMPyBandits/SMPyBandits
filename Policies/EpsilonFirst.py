# -*- coding: utf-8 -*-
""" The epsilon-first random policy.
Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np
import random

epsilon = 0.1


class EpsilonFirst(object):
    """ The epsilon-first random policy.
    Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, horizon, epsilon=epsilon):
        self.nbArms = nbArms
        self.horizon = horizon
        assert 0 <= epsilon <= 1, "Error: the epsilon parameter for EpsilonFirst class has to be in [0, 1]."
        self.epsilon = epsilon
        self.rewards = np.zeros(nbArms)
        self.params = ''
        self.t = -1

    def __str__(self):
        return "EpsilonFirst"

    def startGame(self):
        self.t = 0
        self.rewards = np.zeros(self.nbArms)

    def choice(self):
        if self.t <= self.epsilon * self.horizon:
            # First phase: randomly explore!
            arm = random.randint(0, self.nbArms - 1)
        else:
            # Second phase: just exploit!
            arm = np.argmax(self.rewards)
        return arm

    def getReward(self, arm, reward):
        self.rewards[arm] += reward
        self.t += 1
