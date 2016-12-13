# -*- coding: utf-8 -*-
""" The epsilon-decreasing random policy.
Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import random
import numpy as np
from .BasePolicy import BasePolicy

EPSILON = 0.1
DECREASINGRATE = 1e-6


class EpsilonDecreasing(BasePolicy):
    """ The epsilon-decreasing random policy.
    Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, epsilon=EPSILON, decreasingRate=DECREASINGRATE, lower=0., amplitude=1.):
        super(EpsilonDecreasing, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert 0 <= epsilon <= 1, "Error: the epsilon parameter for EpsilonDecreasing class has to be in [0, 1]."
        self.epsilon = epsilon
        assert decreasingRate > 0, "Error: the decreasingRate parameter for EpsilonDecreasing class has to be > 0."
        self.decreasingRate = decreasingRate

    def __str__(self):
        return "EpsilonDecreasing({})".format(self.decreaseRate)

    def choice(self):
        if random.random() < self.epsilon * np.exp(- self.t * self.decreasingRate):
            # Proba epsilon : explore
            arm = random.randint(0, self.nbArms - 1)
        else:
            # Proba 1-epsilon : exploit
            arm = np.argmax(self.rewards)
        return arm
