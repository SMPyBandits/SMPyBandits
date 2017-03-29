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
        self.unbiased = unbiased  #: Unbiased estimators ?
        # Internal memory
        self.weights = np.ones(nbArms) / nbArms  #: Weights on the arms
        # trying to randomize the order of the initial visit to each arm; as this determinism breaks its habitility to play efficiently in multi-players games
        # XXX do even more randomized, take a random permutation of the arm ?
        self._initial_exploration = rn.permutation(nbArms)
        # The proba that another player has the same is nbPlayers / factorial(nbArms) : should be SMALL !

    def startGame(self):
        """Start with uniform weights."""
        super(Exp3, self).startGame()
        self.weights.fill(1. / self.nbArms)

    def __str__(self):
        return "Exp3(gamma: {:.3g})".format(self.gamma)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        r"""Constant :math:`\gamma_t = \gamma`."""
        return self._gamma

    @property
    def trusts(self):
        r"""Update the trusts probabilities according to Exp3 formula, and the parameter :math:`\gamma_t`.

        .. math::

           \mathrm{trusts}'_k(t+1) &= (1 - \gamma_t) w_k(t) + \gamma_t \frac{1}{K},
           \mathrm{trusts}(t+1) &= \mathrm{trusts}'(t+1) / \sum_{k=1}^{K} \mathrm{trusts}'_k(t+1).

        If :math:`w_k(t)` is the current weight from arm k.
        """
        # Mixture between the weights and the uniform distribution
        p_t = ((1 - self.gamma) * self.weights) + (self.gamma / self.nbArms)
        return p_t / np.sum(p_t)

    def getReward(self, arm, reward):
        r"""Give a reward: accumulate rewards on that arm k, then update the weight :math:`w_k(t)` and renormalize the weights.

        - With unbiased estimators, devide by the trust on that arm k, ie the probability of observing arm k: :math:`\tilde{r}_k(t) = \frac{r_k(t)}{\mathrm{trusts}_k(t)}`.
        - But with a biased estimators, :math:`\tilde{r}_k(t) = r_k(t)`.

        .. math::

           w'_k(t+1) &= w_k(t) \times \exp\left( \frac{\tilde{r}_k(t)}{\gamma_t N_k(t)} \right) \\
           w(t+1) &= w'(t+1) / \sum_{k=1}^{K} w'_k(t+1).
        """
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
        """One random selection, with probabilities = trusts, thank to :func:`numpy.random.choice`."""
        # Force to first visit each arm once in the first steps
        if self.t < self.nbArms:
            # DONE we could use a random permutation instead of deterministic order!
            return self._initial_exploration[self.t]
        else:
            return rn.choice(self.nbArms, p=self.trusts)

    def choiceWithRank(self, rank=1):
        """Multiple (rank >= 1) random selection, with probabilities = trusts, thank to :func:`numpy.random.choice`, and select the last one (less probable)."""
        if (self.t < self.nbArms) or (rank == 1):
            return self.choice()
        else:
            return rn.choice(self.nbArms, size=rank, replace=False, p=self.trusts)[rank - 1]

    def choiceFromSubSet(self, availableArms='all'):
        """One random selection, from availableArms, with probabilities = trusts, thank to :func:`numpy.random.choice`."""
        if (self.t < self.nbArms) or (availableArms == 'all') or (len(availableArms) == self.nbArms):
            return self.choice()
        else:
            return rn.choice(availableArms, p=self.trusts[availableArms])

    def choiceMultiple(self, nb=1):
        """Multiple (nb >= 1) random selection, with probabilities = trusts, thank to :func:`numpy.random.choice`."""
        if (self.t < self.nbArms) or (nb == 1):
            return np.array([self.choice() for _ in range(nb)])  # good size if nb > 1 but t < nbArms
        else:
            return rn.choice(self.nbArms, size=nb, replace=False, p=self.trusts)


# --- Three special cases

class Exp3WithHorizon(Exp3):
    """ Exp3 with fixed gamma, :math:`\gamma_t = \gamma_0`, chosen with a knowledge of the horizon."""

    def __init__(self, nbArms, horizon, lower=0., amplitude=1.):
        super(Exp3WithHorizon, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert horizon > 0, "Error: the 'horizon' parameter for SoftmaxWithHorizon class has to be > 0."
        self.horizon = horizon

    def __str__(self):
        return "Exp3(horizon: {})".format(self.horizon)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        r""" Fixed temperature, small, knowing the horizon: :math:`\gamma_t = \sqrt(\frac{2 \log(K)}{T K})` (*heuristic*).

        - Cf. Theorem 3.1 case #1 of [Bubeck & Cesa-Bianchi, 2012].
        """
        return np.sqrt(2 * np.log(self.nbArms) / (self.horizon * self.nbArms))


class Exp3Decreasing(Exp3):
    r""" Exp3 with decreasing parameter :math:`\gamma_t`."""

    def __str__(self):
        return "Exp3(decreasing)"

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        r""" Decreasing gamma with the time: :math:`\gamma_t = \sqrt(\frac{\log(K)}{t K})` (*heuristic*).

        - Cf. Theorem 3.1 case #2 of [Bubeck & Cesa-Bianchi, 2012].
        """
        return np.sqrt(np.log(self.nbArms) / (self.t * self.nbArms))


class Exp3SoftMix(Exp3):
    r""" Another Exp3 with decreasing parameter :math:`\gamma_t`."""

    def __str__(self):
        return "Exp3(SoftMix)"

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        r""" Decreasing gamma parameter with the time: :math:`\gamma_t = c \frac{\log(t)}{t}` (*heuristic*).

        - Cf. [Cesa-Bianchi & Fisher, 1998].
        - Default value for is :math:`c = \sqrt(\frac{\log(K)}{K})`.
        """
        c = np.sqrt(np.log(self.nbArms) / self.nbArms)
        return c * np.log(self.t) / self.t
