# -*- coding: utf-8 -*-
""" The Exp3 index policy.
Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi, §3.1](http://research.microsoft.com/en-us/um/people/sebubeck/SurveyBCB12.pdf)
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np
import numpy.random as rn
from .BasePolicy import BasePolicy

#: self.unbiased is a flag to know if the rewards are used as biased estimator,
#: ie just r_t, or unbiased estimators, r_t / trusts_t
UNBIASED = True
UNBIASED = False

#: Default gamma parameter
GAMMA = 0.01


class Exp3(BasePolicy):
    """ The Exp3 index policy.
    Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi, §3.1](http://research.microsoft.com/en-us/um/people/sebubeck/SurveyBCB12.pdf)
    """

    def __init__(self, nbArms, gamma=GAMMA,
                 unbiased=UNBIASED, lower=0., amplitude=1.):
        super(Exp3, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        if gamma is None:  # Use a default value for the gamma parameter
            gamma = np.sqrt(np.log(nbArms) / nbArms)
        assert 0 < gamma <= 1, "Error: the 'gamma' parameter for Exp3 class has to be in (0, 1]."
        self._gamma = gamma
        self.unbiased = unbiased
        # Internal memory
        self.weights = np.ones(nbArms) / nbArms
        # trying to randomize the order of the initial visit to each arm; as this determinism breaks its habitility to play efficiently in multi-players games
        # XXX do even more randomized, take a random permutation of the arm ?
        self._initial_exploration = rn.permutation(nbArms)
        # The proba that another player has the same is nbPlayers / factorial(nbArms) : should be SMALL !

    def startGame(self):
        super(Exp3, self).startGame()
        self.weights.fill(1. / self.nbArms)

    def __str__(self):
        return "Exp3(gamma: {:.3g})".format(self.gamma)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        return self._gamma

    @property
    def trusts(self):
        # Mixture between the weights and the uniform distribution
        p_t = ((1 - self.gamma) * self.weights) + (self.gamma / self.nbArms)
        return p_t / np.sum(p_t)

    def getReward(self, arm, reward):
        super(Exp3, self).getReward(arm, reward)  # XXX Call to BasePolicy
        # Update weight of THIS arm, with this biased or unbiased reward
        if self.unbiased:
            reward /= self.trusts[arm]
        # Multiplicative weights
        self.weights[arm] *= np.exp(reward * (self.gamma / self.nbArms))
        # Renormalize weights at each step
        self.weights /= np.sum(self.weights)

    # --- Choice methods

    def choice(self):
        # Force to first visit each arm once in the first steps
        if self.t < self.nbArms:
            # DONE we could use a random permutation instead of deterministic order!
            return self._initial_exploration[self.t]
        else:
            return rn.choice(self.nbArms, p=self.trusts)

    def choiceWithRank(self, rank=1):
        if (self.t < self.nbArms) or (rank == 1):
            return self.choice()
        else:
            return rn.choice(self.nbArms, size=rank, replace=False, p=self.trusts)[rank - 1]

    def choiceFromSubSet(self, availableArms='all'):
        if (self.t < self.nbArms) or (availableArms == 'all') or (len(availableArms) == self.nbArms):
            return self.choice()
        else:
            return rn.choice(availableArms, p=self.trusts[availableArms])

    def choiceMultiple(self, nb=1):
        if (self.t < self.nbArms) or (nb == 1):
            return np.array([self.choice() for _ in range(nb)])  # good size if nb > 1 but t < nbArms
        else:
            return rn.choice(self.nbArms, size=nb, replace=False, p=self.trusts)


# --- Two special cases

class Exp3Decreasing(Exp3):
    """ Exp3 with decreasing gamma eta_t."""

    def __str__(self):
        return "Exp3(decreasing)"

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        """ Decreasing gamma with the time: sqrt(log(K) / (t * K)).

        - Cf. Theorem 3.1 case #2 of [Bubeck & Cesa-Bianchi, 2012].
        """
        return np.sqrt(np.log(self.nbArms) / (self.t * self.nbArms))


class Exp3SoftMix(Exp3):
    """ Another Exp3 with decreasing gamma eta_t."""

    def __str__(self):
        return "Exp3(SoftMix)"

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        """ Decreasing gamma with the time: c * log(t) / t.

        - Cf. [Cesa-Bianchi & Fisher, 1998].
        - Default value for c = sqrt(log(K) / K).
        """
        c = np.sqrt(np.log(self.nbArms) / self.nbArms)
        return c * np.log(self.t) / self.t


class Exp3WithHorizon(Exp3):
    """ Exp3 with fixed gamma eta chosen with a knowledge of the horizon."""

    def __init__(self, nbArms, horizon, lower=0., amplitude=1.):
        super(Exp3WithHorizon, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert horizon > 0, "Error: the 'horizon' parameter for SoftmaxWithHorizon class has to be > 0."
        self.horizon = horizon

    def __str__(self):
        return "Exp3(horizon: {})".format(self.horizon)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        """ Fixed gamma, small, knowing the horizon: sqrt(2 * log(K) / (horizon * K)).

        - Cf. Theorem 3.1 case #1 of [Bubeck & Cesa-Bianchi, 2012].
        """
        return np.sqrt(2 * np.log(self.nbArms) / (self.horizon * self.nbArms))
