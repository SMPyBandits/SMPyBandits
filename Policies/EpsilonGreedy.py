# -*- coding: utf-8 -*-
""" The epsilon-greedy random policy.
Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np

from .BasePolicy import BasePolicy

EPSILON = 0.1


class EpsilonGreedy(BasePolicy):
    """ The epsilon-greedy random policy.
    Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, epsilon=EPSILON, lower=0., amplitude=1.):
        super(EpsilonGreedy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for EpsilonGreedy class has to be in [0, 1]."
        self._epsilon = epsilon

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):  # Allow child classes to use time-dependent epsilon coef
        return self._epsilon

    def __str__(self):
        return "EpsilonGreedy({})".format(self.epsilon)

    def choice(self):
        if np.random.random() < self.epsilon:  # Proba epsilon : explore
            return np.random.randint(0, self.nbArms - 1)
        else:  # Proba 1 - epsilon : exploit
            # Uniform choice among the best arms
            return np.random.choice(np.nonzero(self.rewards == np.max(self.rewards))[0])

    def choiceWithRank(self, rank=1):
        if rank == 1:
            return self.choice()
        else:
            if np.random.random() < self.epsilon:  # Proba epsilon : explore
                return np.random.randint(0, self.nbArms - 1)
            else:  # Proba 1 - epsilon : exploit
                sortedRewards = np.sort(self.rewards)  # XXX What happens here if two arms has the same index, being the max?
                chosenIndex = sortedRewards[-rank]
                # Uniform choice among the rank-th best arms
                return np.random.choice(np.nonzero(self.index == chosenIndex)[0])

    def choiceFromSubSet(self, availableArms='all'):
        if (availableArms == 'all') or (len(availableArms) == self.nbArms):
            return self.choice()
        else:
            if np.random.random() < self.epsilon:  # Proba epsilon : explore
                return np.random.choice(availableArms)
            else:  # Proba 1 - epsilon : exploit
                # Uniform choice among the best arms
                return np.random.choice(np.nonzero(self.rewards[availableArms] == np.max(self.rewards[availableArms]))[0])

    def choiceMultiple(self, nb=1):
        if nb == 1:
            return self.choice()
        else:
            if np.random.random() < self.epsilon:  # Proba epsilon : explore
                return np.random.choice(self.nbArms, size=nb, replace=False)
            else:  # Proba 1 - epsilon : exploit
                sortedRewards = np.sort(self.rewards)
                # Uniform choice among the best arms
                return np.random.choice(np.nonzero(self.index >= sortedRewards[-nb])[0], size=nb, replace=False)
