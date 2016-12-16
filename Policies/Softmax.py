# -*- coding: utf-8 -*-
""" The Boltzmann Exploration (Softmax) index policy.
Reference: [http://www.cs.mcgill.ca/~vkules/bandits.pdf §2.1].

Very similar to Exp3 but in the bandit setting and not the full information setting.
Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi, §3.1](http://research.microsoft.com/en-us/um/people/sebubeck/SurveyBCB12.pdf)
"""

__author__ = "Lilian Besson"
__version__ = "0.3"

import numpy as np
from .BasePolicy import BasePolicy

# self.unbiased is a flag to know if the rewards are used as biased estimator,
# ie just r_t, or unbiased estimators, r_t / p_t
UNBIASED = True
UNBIASED = False


class Softmax(BasePolicy):
    """The Boltzmann Exploration (Softmax) index policy.
    Reference: [http://www.cs.mcgill.ca/~vkules/bandits.pdf §2.1].

    Very similar to Exp3 but in the bandit setting and not the full information setting.
    Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi, §3.1](http://research.microsoft.com/en-us/um/people/sebubeck/SurveyBCB12.pdf)
    """

    def __init__(self, nbArms, temperature=None, unbiased=UNBIASED, lower=0., amplitude=1.):
        super(Softmax, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        if temperature is None:
            temperature = np.sqrt(np.log(nbArms) / nbArms)
        assert temperature > 0, "Error: the temperature parameter for Softmax class has to be > 0."
        self._temperature = temperature
        self.unbiased = unbiased
        self.trusts = np.ones(nbArms) / nbArms

    def startGame(self):
        super(Softmax, self).startGame()
        self.trusts.fill(1. / self.nbArms)

    def __str__(self):
        return "Softmax(temp: {})".format(self.temperature)

    # This decorator @property makes this method an attributes, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def temperature(self):
        return self._temperature

    def choice(self):
        rewards = (self.rewards - self.lower) / self.amplitude
        # Force to first visit each arm once in the first steps
        if self.t < self.nbArms:
            arm = self.t % self.nbArms  # TODO? random permutation instead of deterministic order!
        else:
            if self.unbiased:
                # FIXME we should divide by the proba p_t of selecting actions, not by the trusts !
                estimator_rewards = rewards / self.trusts
            else:
                estimator_rewards = rewards
            trusts = np.exp(estimator_rewards / (self.temperature * self.pulls))
            self.trusts = trusts / np.sum(trusts)
            arm = np.random.choice(self.nbArms, p=self.trusts)
        return arm

    def choiceWithRank(self, rank=1):
        rewards = (self.rewards - self.lower) / self.amplitude
        # Force to first visit each arm once in the first steps
        if self.t < self.nbArms:
            arm = self.t % self.nbArms  # TODO? random permutation instead of deterministic order!
        else:
            if self.unbiased:
                # FIXME we should divide by the proba p_t of selecting actions, not by the trusts !
                estimator_rewards = rewards / self.trusts
            else:
                estimator_rewards = rewards
            trusts = np.exp(estimator_rewards / (self.temperature * self.pulls))
            self.trusts = trusts / np.sum(trusts)
            arm = np.random.choice(self.nbArms, size=rank, replace=False, p=self.trusts)[rank - 1]
        return arm


# --- Two special cases

class SoftmaxDecreasing(Softmax):
    """ Softmax / Exp3 with decreasing temperature eta_t."""

    def __str__(self):
        return "Softmax(decreasing)"

    # This decorator @property makes this method an attributes, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def temperature(self):
        """ Decreasing temperature with the time: sqrt(log(K) / (t * K)).

        - Cf. Theorem 3.1 case #2 of [Bubeck & Cesa-Bianchi, 2012].
        """
        return np.sqrt(np.log(self.nbArms) / (self.t * self.nbArms))


class SoftmaxWithHorizon(Softmax):
    """ Softmax / Exp3 with fixed temperature eta chosen with a knowledge of the horizon."""

    def __init__(self, nbArms, horizon, lower=0., amplitude=1.):
        super(SoftmaxWithHorizon, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.horizon = horizon

    def __str__(self):
        return "Softmax(horizon: {})".format(self.horizon)

    # This decorator @property makes this method an attributes, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def temperature(self):
        """ Fixed temperature, small, knowing the horizon: sqrt(2 * log(K) / (horizon * K)).

        - Cf. Theorem 3.1 case #1 of [Bubeck & Cesa-Bianchi, 2012].
        """
        return np.sqrt(2 * np.log(self.nbArms) / (self.horizon * self.nbArms))
