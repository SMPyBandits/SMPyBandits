# -*- coding: utf-8 -*-
""" The epsilon-decreasing random policy.

- epsilon(t) = epsilon0 / t
- Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

from .EpsilonGreedy import EpsilonGreedy

EPSILON = 0.1


class EpsilonDecreasing(EpsilonGreedy):
    """ The epsilon-decreasing random policy.

    - epsilon(t) = epsilon0 / t
    - Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, epsilon=EPSILON, lower=0., amplitude=1.):
        super(EpsilonDecreasing, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for EpsilonDecreasing class has to be in [0, 1]."
        self._epsilon = epsilon

    def __str__(self):
        return "EpsilonDecreasing(e:{})".format(self._epsilon)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):
        return min(1, self._epsilon / max(1, self.t))
