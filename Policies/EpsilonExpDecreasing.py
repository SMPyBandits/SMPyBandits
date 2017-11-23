# -*- coding: utf-8 -*-
r""" The epsilon exp-decreasing random policy.

- :math:`\varepsilon(t) = \varepsilon_0 \exp(-t \mathrm{decreasingRate})`.
- Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np

from .EpsilonGreedy import EpsilonGreedy

#: Default value for epsilon
EPSILON = 0.1

#: Default value for the constant for the decreasing rate
DECREASINGRATE = 1e-6


class EpsilonExpDecreasing(EpsilonGreedy):
    r""" The epsilon exp-decreasing random policy.

    - :math:`\varepsilon(t) = \varepsilon_0 \exp(-t \mathrm{decreasingRate})`.
    - Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, epsilon=EPSILON, decreasingRate=DECREASINGRATE, lower=0., amplitude=1.):
        super(EpsilonExpDecreasing, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for EpsilonExpDecreasing class has to be in [0, 1]."  # DEBUG
        self._epsilon = epsilon
        assert decreasingRate > 0, "Error: the 'decreasingRate' parameter for EpsilonExpDecreasing class has to be > 0."
        self._decreasingRate = decreasingRate

    def __str__(self):
        return "EpsilonExpDecreasing(e:{}, r:{})".format(self._epsilon, self._decreasingRate)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):
        return min(1, self._epsilon * np.exp(- self.t * self._decreasingRate))
