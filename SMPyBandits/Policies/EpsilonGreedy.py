# -*- coding: utf-8 -*-
r""" The epsilon-greedy random policies, with the naive one and some variants.

- At every time step, a fully uniform random exploration has probability :math:`\varepsilon(t)` to happen, otherwise an exploitation is done on accumulated rewards (not means).
- Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies

.. warning:: Except if :math:`\varepsilon(t)` is optimally tuned for a specific problem, none of these policies can hope to be efficient.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from random import random
import numpy as np
import numpy.random as rn

try:
    from .BasePolicy import BasePolicy
    from .with_proba import with_proba
except ImportError:
    from BasePolicy import BasePolicy
    from with_proba import with_proba


#: Default value for epsilon for :class:`EpsilonGreedy`
EPSILON = 0.1


class EpsilonGreedy(BasePolicy):
    r""" The epsilon-greedy random policy.

    - At every time step, a fully uniform random exploration has probability :math:`\varepsilon(t)` to happen, otherwise an exploitation is done on accumulated rewards (not means).
    - Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, epsilon=EPSILON, lower=0., amplitude=1.):
        super(EpsilonGreedy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for EpsilonGreedy class has to be in [0, 1]."  # DEBUG
        self._epsilon = epsilon

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def epsilon(self):  # Allow child classes to use time-dependent epsilon coef
        return self._epsilon

    def __str__(self):
        return r"EpsilonGreedy($\varepsilon={:.3g}$)".format(self.epsilon)

    def choice(self):
        """With a probability of epsilon, explore (uniform choice), otherwhise exploit based on just accumulated *rewards* (not empirical mean rewards)."""
        if with_proba(self.epsilon):  # Proba epsilon : explore
            return rn.randint(0, self.nbArms - 1)
        else:  # Proba 1 - epsilon : exploit
            # Uniform choice among the best arms
            biased_means = self.rewards / (1 + self.pulls)
            # return rn.choice(np.nonzero(biased_means == np.max(biased_means))[0])
            # WARNING why max on rewards and not mean rewards?
            return rn.choice(np.nonzero(self.rewards == np.max(self.rewards))[0])

    def choiceWithRank(self, rank=1):
        """With a probability of epsilon, explore (uniform choice), otherwhise exploit with the rank, based on just accumulated *rewards* (not empirical mean rewards)."""
        if rank == 1:
            return self.choice()
        else:
            if with_proba(self.epsilon):  # Proba epsilon : explore
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
            print("WARNING: EpsilonGreedy.choiceFromSubSet({}): the argument availableArms of type {} should not be empty.".format(availableArms, type(availableArms)))  # DEBUG
            # WARNING if no arms are tagged as available, what to do ? choose an arm at random, or call choice() as if available == 'all'
            return self.choice()
            # return np.random.randint(self.nbArms)
        else:
            if with_proba(self.epsilon):  # Proba epsilon : explore
                return rn.choice(availableArms)
            else:  # Proba 1 - epsilon : exploit
                # Uniform choice among the best arms
                return rn.choice(np.nonzero(self.rewards[availableArms] == np.max(self.rewards[availableArms]))[0])

    def choiceMultiple(self, nb=1):
        if nb == 1:
            return np.array([self.choice()])
        else:
            # FIXME the explore/exploit balance should be for each choice, right?
            if with_proba(self.epsilon):  # Proba epsilon : Explore
                return rn.choice(self.nbArms, size=nb, replace=False)
            else:  # Proba 1 - epsilon : exploit
                sortedRewards = np.sort(self.rewards)
                # Uniform choice among the best arms
                return rn.choice(np.nonzero(self.rewards >= sortedRewards[-nb])[0], size=nb, replace=False)


# --- Epsilon-Decreasing

#: Default value for epsilon for :class:`EpsilonDecreasing`
EPSILON = 0.1


class EpsilonDecreasing(EpsilonGreedy):
    r""" The epsilon-decreasing random policy.

    - :math:`\varepsilon(t) = \min(1, \varepsilon_0 / \max(1, t))`
    - Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, epsilon=EPSILON, lower=0., amplitude=1.):
        super(EpsilonDecreasing, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # assert 0. <= epsilon <= 1., "Error: the 'epsilon' parameter for EpsilonDecreasing class has to be in [0, 1]."  # DEBUG
        self._epsilon = epsilon

    def __str__(self):
        return r"EpsilonDecreasing($\varepsilon_0={:.3g}$)".format(self._epsilon)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def epsilon(self):
        r"""Decreasing :math:`\varepsilon(t) = \min(1, \varepsilon_0 / \max(1, t))`."""
        return min(1, self._epsilon / max(1, self.t))


#  --- EpsilonDecreasingMEGA

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

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def epsilon(self):
        r"""Decreasing :math:`\varepsilon(t) = \min(1, \varepsilon_0 / \max(1, t))`."""
        return min(1, self._epsilon / max(1, self.t))


# --- Epsilon-First

#: Default value for epsilon for :class:`EpsilonFirst`
EPSILON = 0.01


class EpsilonFirst(EpsilonGreedy):
    """ The epsilon-first random policy.
    Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, horizon, epsilon=EPSILON, lower=0., amplitude=1.):
        super(EpsilonFirst, self).__init__(nbArms, epsilon=epsilon, lower=lower, amplitude=amplitude)
        assert horizon > 0, "Error: the 'horizon' parameter for EpsilonFirst class has to be > 0."
        self.horizon = int(horizon)  #: Parameter :math:`T` = known horizon of the experiment.
        # assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for EpsilonFirst class has to be in [0, 1]."  # DEBUG
        self._epsilon = epsilon

    def __str__(self):
        return r"EpsilonFirst($T={}$, $\varepsilon={:.3g}$)".format(self.horizon, self._epsilon)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def epsilon(self):
        r"""1 while :math:`t \leq \varepsilon_0 T`, 0 after."""
        if self.t <= self._epsilon * self.horizon:
            # First phase: randomly explore!
            return 1
        else:
            # Second phase: just exploit!
            return 0


#: Default value for epsilon for :class:`EpsilonDecreasing`
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
        # assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for EpsilonExpDecreasing class has to be in [0, 1]."  # DEBUG
        self._epsilon = epsilon
        assert decreasingRate > 0, "Error: the 'decreasingRate' parameter for EpsilonExpDecreasing class has to be > 0."
        self._decreasingRate = decreasingRate

    def __str__(self):
        return "EpsilonExpDecreasing(e:{}, r:{})".format(self._epsilon, self._decreasingRate)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def epsilon(self):
        r"""Decreasing :math:`\varepsilon(t) = \min(1, \varepsilon_0 \exp(- t \tau))`."""
        return min(1, self._epsilon * np.exp(- self.t * self._decreasingRate))
