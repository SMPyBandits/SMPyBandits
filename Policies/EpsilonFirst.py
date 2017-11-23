# -*- coding: utf-8 -*-
""" The epsilon-first random policy.
Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.2"

from .EpsilonGreedy import EpsilonGreedy

#: Default value for epsilon
EPSILON = 0.01


class EpsilonFirst(EpsilonGreedy):
    """ The epsilon-first random policy.
    Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, horizon, epsilon=EPSILON, lower=0., amplitude=1.):
        super(EpsilonFirst, self).__init__(nbArms, epsilon=epsilon, lower=lower, amplitude=amplitude)
        assert horizon > 0, "Error: the 'horizon' parameter for EpsilonFirst class has to be > 0."
        self.horizon = int(horizon) if horizon is not None else None  #: Parameter :math:`T` = known horizon of the experiment.
        assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for EpsilonFirst class has to be in [0, 1]."  # DEBUG
        self._epsilon = epsilon

    def __str__(self):
        return r"EpsilonFirst($T={}$, $\varepsilon={:.3g}$)".format(self.horizon, self._epsilon)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):
        r"""1 while :math:`t \leq \varepsilon_0 T`, 0 after."""
        if self.t <= self._epsilon * self.horizon:
            # First phase: randomly explore!
            return 1
        else:
            # Second phase: just exploit!
            return 0
