# -*- coding: utf-8 -*-
""" The epsilon-greedy random policy.
Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np
import random
from .IndexPolicy import IndexPolicy


epsilon = 0.1


class EpsilonGreedy(IndexPolicy):
    """ The epsilon-greedy random policy.
    Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, epsilon=epsilon):
        self.nbArms = nbArms
        assert 0 <= epsilon <= 1, "Error: the epsilon parameter for EpsilonGreedy class has to be in [0, 1]."
        self.epsilon = epsilon
        self.rewards = np.zeros(nbArms)
        self.params = ''

    def __str__(self):
        return "EpsilonGreedy"

    def startGame(self):
        self.rewards = np.zeros(self.nbArms)

    def choice(self):
        if random.random() < self.epsilon:  # Proba epsilon : explore
            arm = random.randint(0, self.nbArms - 1)
        else:  # Proba 1-epsilon : exploit
            arm = np.argmax(self.rewards)
        return arm

    def getReward(self, arm, reward):
        self.rewards[arm] += reward
