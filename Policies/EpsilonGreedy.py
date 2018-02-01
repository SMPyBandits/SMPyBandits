# -*- coding: utf-8 -*-
r""" The epsilon-greedy random policy.

- At every time step, a fully uniform random exploration has probability :math:`\varepsilon(t)` to happen, otherwise an exploitation is done on accumulated rewards (not means).
- Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.2"

from warnings import warn
from random import random
import numpy as np
import numpy.random as rn

from .BasePolicy import BasePolicy

#: Default value for epsilon
EPSILON = 0.1


class EpsilonGreedy(BasePolicy):
    r""" The epsilon-greedy random policy.

    - At every time step, a fully uniform random exploration has probability :math:`\varepsilon(t)` to happen, otherwise an exploitation is done on accumulated rewards (not means).
    - Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, epsilon=EPSILON, lower=0., amplitude=1.):
        super(EpsilonGreedy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for EpsilonGreedy class has to be in [0, 1]."  # DEBUG
        self._epsilon = epsilon

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):  # Allow child classes to use time-dependent epsilon coef
        return self._epsilon

    def __str__(self):
        return "EpsilonGreedy({})".format(self.epsilon)

    def choice(self):
        """With a probability of epsilon, explore (uniform choice), otherwhise exploit based on just accumulated *rewards* (not empirical mean rewards)."""
        if random() < self.epsilon:  # Proba epsilon : explore
            return rn.randint(0, self.nbArms - 1)
        else:  # Proba 1 - epsilon : exploit
            # Uniform choice among the best arms
            return rn.choice(np.nonzero(self.rewards == np.max(self.rewards))[0])

    def choiceWithRank(self, rank=1):
        """With a probability of epsilon, explore (uniform choice), otherwhise exploit with the rank, based on just accumulated *rewards* (not empirical mean rewards)."""
        if rank == 1:
            return self.choice()
        else:
            if random() < self.epsilon:  # Proba epsilon : explore
                return rn.randint(0, self.nbArms - 1)
            else:  # Proba 1 - epsilon : exploit
                sortedRewards = np.sort(self.rewards)
                chosenIndex = sortedRewards[-rank]
                # Uniform choice among the rank-th best arms
                return rn.choice(np.nonzero(self.rewards == chosenIndex)[0])

    def choiceFromSubSet(self, availableArms='all'):
        if (availableArms == 'all') or (len(availableArms) == self.nbArms):
            return self.choice()
        elif len(availableArms) == 0:
            warn("EpsilonGreedy.choiceFromSubSet({}): the argument availableArms of type {} should not be empty.".format(availableArms, type(availableArms)), RuntimeWarning)
            # FIXME if no arms are tagged as available, what to do ? choose an arm at random, or call choice() as if available == 'all'
            return self.choice()
            # return np.random.randint(self.nbArms)
        else:
            if random() < self.epsilon:  # Proba epsilon : explore
                return rn.choice(availableArms)
            else:  # Proba 1 - epsilon : exploit
                # Uniform choice among the best arms
                return rn.choice(np.nonzero(self.rewards[availableArms] == np.max(self.rewards[availableArms]))[0])

    def choiceMultiple(self, nb=1):
        if nb == 1:
            return np.array([self.choice()])
        else:
            # FIXME the explore/exploit balancy should be for each choice, right?
            if random() < self.epsilon:  # Proba epsilon : Explore
                return rn.choice(self.nbArms, size=nb, replace=False)
            else:  # Proba 1 - epsilon : exploit
                sortedRewards = np.sort(self.rewards)
                # Uniform choice among the best arms
                return rn.choice(np.nonzero(self.rewards >= sortedRewards[-nb])[0], size=nb, replace=False)
