# -*- coding: utf-8 -*-
""" The epsilon-greedy random policy.
Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import random
import numpy as np
from .BasePolicy import BasePolicy

EPSILON = 0.1


class EpsilonGreedy(BasePolicy):
    """ The epsilon-greedy random policy.
    Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, epsilon=EPSILON, lower=0., amplitude=1.):
        super(EpsilonGreedy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert 0 <= epsilon <= 1, "Error: the epsilon parameter for EpsilonGreedy class has to be in [0, 1]."
        self.epsilon = epsilon

    def __str__(self):
        return "EpsilonGreedy({})".format(self.epsilon)

    def choice(self):
        if random.random() < self.epsilon:  # Proba epsilon : explore
            arm = random.randint(0, self.nbArms - 1)
        else:  # Proba 1-epsilon : exploit
            # FIXED Uniform choice among the best arms
            arm = np.random.choice(np.nonzero(self.rewards == np.max(self.rewards)))
        return arm

    def choiceWithRank(self, rank=1):
        if random.random() < self.epsilon:  # Proba epsilon : explore
            arm = random.randint(0, self.nbArms - 1)
        else:  # Proba 1-epsilon : exploit
            # FIXME be more efficient
            try:
                uniqueValues = np.sort(np.unique(self.rewards))  # XXX Should we do a np.unique here ??
                chosenIndex = uniqueValues[-rank]
            except IndexError:
                values = np.sort(self.rewards)  # XXX What happens here if two arms has the same index, being the max?
                chosenIndex = values[-rank]
            # FIXED Uniform choice among the rank-th best arms
            arm = np.random.choice(np.nonzero(self.index == chosenIndex))
        return arm
