# -*- coding: utf-8 -*-
""" The epsilon-first random policy.
Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import random
import numpy as np
from .BasePolicy import BasePolicy

EPSILON = 0.1


class EpsilonFirst(BasePolicy):
    """ The epsilon-first random policy.
    Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, horizon, epsilon=EPSILON, lower=0., amplitude=1.):
        super(EpsilonFirst, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.horizon = horizon
        assert 0 <= epsilon <= 1, "Error: the epsilon parameter for EpsilonFirst class has to be in [0, 1]."
        self.epsilon = epsilon

    def __str__(self):
        return "EpsilonFirst({})".format(self.epsilon)

    def choice(self):
        if self.t <= self.epsilon * self.horizon:
            # First phase: randomly explore!
            arm = random.randint(0, self.nbArms - 1)
        else:
            # Second phase: just exploit!
            arm = np.argmax(self.rewards)
        return arm
