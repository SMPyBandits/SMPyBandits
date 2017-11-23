# -*- coding: utf-8 -*-
r""" The epsilon-decreasing random policy, using MEGA's heuristic for a good choice of epsilon0 value.

- :math:`\varepsilon(t) = \min(1, \varepsilon_0 / \max(1, t))`
- :math:`\varepsilon_0 = \frac{c K^2}{d^2 (K - 1)}`
- Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.2"

from .EpsilonGreedy import EpsilonGreedy

C = 0.1  #: Constant C in the MEGA formula
D = 0.5  #: Constant C in the MEGA formula


def epsilon0(c, d, nbArms):
    r"""MEGA heuristic:

    .. math:: \varepsilon_0 = \frac{c K^2}{d^2 (K - 1)}.
    """
    return (c * nbArms**2) / (d**2 * (nbArms - 1))


class EpsilonDecreasingMEGA(EpsilonGreedy):
    r""" The epsilon-decreasing random policy, using MEGA's heuristic for a good choice of epsilon0 value.

    - :math:`\varepsilon(t) = \min(1, \varepsilon_0 / \max(1, t))`
    - :math:`\varepsilon_0 = \frac{c K^2}{d^2 (K - 1)}`
    - Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, c=C, d=D, lower=0., amplitude=1.):
        super(EpsilonDecreasingMEGA, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self._epsilon = epsilon0(c, d, nbArms)

    def __str__(self):
        return r"EpsilonDecreasingMEGA($\varepsilon=%.3g$)" % self._epsilon

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):
        r"""Decreasing :math:`\varepsilon(t) = \min(1, \varepsilon_0 / \max(1, t))`."""
        return min(1, self._epsilon / max(1, self.t))
