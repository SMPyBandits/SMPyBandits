# -*- coding: utf-8 -*-
""" The Exp3 randomized index policy.

Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi, §3.1](http://research.microsoft.com/en-us/um/people/sebubeck/SurveyBCB12.pdf)

See also [Evaluation and Analysis of the Performance of the EXP3 Algorithm in Stochastic Environments, Y. Seldin & C. Szepasvari & P. Auer & Y. Abbasi-Adkori, 2012](http://proceedings.mlr.press/v24/seldin12a/seldin12a.pdf).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
import numpy.random as rn
from .BasePolicy import BasePolicy

#: self.unbiased is a flag to know if the rewards are used as biased estimator,
#: i.e., just :math:`r_t`, or unbiased estimators, :math:`r_t / trusts_t`.
UNBIASED = False
UNBIASED = True

#: Default :math:`\gamma` parameter.
GAMMA = 0.01


class Exp3(BasePolicy):
    """ The Exp3 randomized index policy.

    Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi, §3.1](http://research.microsoft.com/en-us/um/people/sebubeck/SurveyBCB12.pdf)

    See also [Evaluation and Analysis of the Performance of the EXP3 Algorithm in Stochastic Environments, Y. Seldin & C. Szepasvari & P. Auer & Y. Abbasi-Adkori, 2012](http://proceedings.mlr.press/v24/seldin12a/seldin12a.pdf).
    """

    def __init__(self, nbArms, gamma=GAMMA,
                 unbiased=UNBIASED, lower=0., amplitude=1.):
        super(Exp3, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        if gamma is None:  # Use a default value for the gamma parameter
            gamma = np.sqrt(np.log(nbArms) / nbArms)
        assert 0 < gamma <= 1, "Error: the 'gamma' parameter for Exp3 class has to be in (0, 1]."  # DEBUG
        self._gamma = gamma
        self.unbiased = unbiased  #: Unbiased estimators ?
        # Internal memory
        self.weights = np.full(nbArms, 1. / nbArms)  #: Weights on the arms
        # trying to randomize the order of the initial visit to each arm; as this determinism breaks its habitility to play efficiently in multi-players games
        # XXX do even more randomized, take a random permutation of the arm ?
        self._initial_exploration = rn.permutation(nbArms)
        # The proba that another player has the same is nbPlayers / factorial(nbArms) : should be SMALL !

    def startGame(self):
        """Start with uniform weights."""
        super(Exp3, self).startGame()
        self.weights.fill(1. / self.nbArms)

    def __str__(self):
        return r"Exp3($\gamma: {:.3g}$)".format(self.gamma)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        r"""Constant :math:`\gamma_t = \gamma`."""
        return self._gamma

    @property
    def trusts(self):
        r"""Update the trusts probabilities according to Exp3 formula, and the parameter :math:`\gamma_t`.

        .. math::

           \mathrm{trusts}'_k(t+1) &= (1 - \gamma_t) w_k(t) + \gamma_t \frac{1}{K}, \\
           \mathrm{trusts}(t+1) &= \mathrm{trusts}'(t+1) / \sum_{k=1}^{K} \mathrm{trusts}'_k(t+1).

        If :math:`w_k(t)` is the current weight from arm k.
        """
        # Mixture between the weights and the uniform distribution
        trusts = ((1 - self.gamma) * self.weights) + (self.gamma / self.nbArms)
        # XXX Handle weird cases, slow down everything but safer!
        if not np.all(np.isfinite(trusts)):
            # XXX some value has non-finite trust, probably on the first steps
            # 1st case: all values are non-finite (nan): set trusts to 1/N uniform choice
            if np.all(~np.isfinite(trusts)):
                trusts = np.full(self.nbArms, 1. / self.nbArms)
            # 2nd case: only few values are non-finite: set them to 0
            else:
                trusts[~np.isfinite(trusts)] = 0
        # Bad case, where the sum is so small that it's only rounding errors
        if np.isclose(np.sum(trusts), 0):
                trusts = np.full(self.nbArms, 1. / self.nbArms)
        # Normalize it and return it
        return trusts / np.sum(trusts)

    def getReward(self, arm, reward):
        r"""Give a reward: accumulate rewards on that arm k, then update the weight :math:`w_k(t)` and renormalize the weights.

        - With unbiased estimators, divide by the trust on that arm k, i.e., the probability of observing arm k: :math:`\tilde{r}_k(t) = \frac{r_k(t)}{\mathrm{trusts}_k(t)}`.
        - But with a biased estimators, :math:`\tilde{r}_k(t) = r_k(t)`.

        .. math::

           w'_k(t+1) &= w_k(t) \times \exp\left( \frac{\tilde{r}_k(t)}{\gamma_t N_k(t)} \right) \\
           w(t+1) &= w'(t+1) / \sum_{k=1}^{K} w'_k(t+1).
        """
        super(Exp3, self).getReward(arm, reward)  # XXX Call to BasePolicy
        # Update weight of THIS arm, with this biased or unbiased reward
        if self.unbiased:
            reward = reward / self.trusts[arm]
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
        """Multiple (rank >= 1) random selection, with probabilities = trusts, thank to :func:`numpy.random.choice`, and select the last one (less probable).

        - Note that if not enough entries in the trust vector are non-zero, then :func:`choice` is called instead (rank is ignored).
        """
        if (self.t < self.nbArms) or (rank == 1) or np.sum(~np.isclose(self.trusts, 0)) < rank:
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
        return rn.choice(self.nbArms, size=nb, replace=False, p=self.trusts)

    # --- Other methods

    def estimatedOrder(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing trust probabilities."""
        return np.argsort(self.trusts)

    def estimatedBestArms(self, M=1):
        """ Return a (non-necessarily sorted) list of the indexes of the M-best arms. Identify the set M-best."""
        assert 1 <= M <= self.nbArms, "Error: the parameter 'M' has to be between 1 and K = {}, but it was {} ...".format(self.nbArms, M)  # DEBUG
        order = self.estimatedOrder()
        return order[-M:]


# --- Three special cases

class Exp3WithHorizon(Exp3):
    r""" Exp3 with fixed gamma, :math:`\gamma_t = \gamma_0`, chosen with a knowledge of the horizon."""

    def __init__(self, nbArms, horizon, unbiased=UNBIASED, lower=0., amplitude=1.):
        super(Exp3WithHorizon, self).__init__(nbArms, unbiased=unbiased, lower=lower, amplitude=amplitude)
        assert horizon > 0, "Error: the 'horizon' parameter for SoftmaxWithHorizon class has to be > 0."
        self.horizon = int(horizon) if horizon is not None else None  #: Parameter :math:`T` = known horizon of the experiment.

    def __str__(self):
        return r"Exp3($T={}$)".format(self.horizon)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        r""" Fixed temperature, small, knowing the horizon: :math:`\gamma_t = \sqrt(\frac{2 \log(K)}{T K})` (*heuristic*).

        - Cf. Theorem 3.1 case #1 of [Bubeck & Cesa-Bianchi, 2012](http://sbubeck.com/SurveyBCB12.pdf).
        """
        return np.sqrt(2 * np.log(self.nbArms) / (self.horizon * self.nbArms))


class Exp3Decreasing(Exp3):
    r""" Exp3 with decreasing parameter :math:`\gamma_t`."""

    def __str__(self):
        return "Exp3(decreasing)"

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        r""" Decreasing gamma with the time: :math:`\gamma_t = \min(\frac{1}{K}, \sqrt(\frac{\log(K)}{t K}))` (*heuristic*).

        - Cf. Theorem 3.1 case #2 of [Bubeck & Cesa-Bianchi, 2012](http://sbubeck.com/SurveyBCB12.pdf).
        """
        return min(1. / self.nbArms, np.sqrt(np.log(self.nbArms) / (self.t * self.nbArms)))


class Exp3SoftMix(Exp3):
    r""" Another Exp3 with decreasing parameter :math:`\gamma_t`."""

    def __str__(self):
        return "Exp3(SoftMix)"

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        r""" Decreasing gamma parameter with the time: :math:`\gamma_t = c \frac{\log(t)}{t}` (*heuristic*).

        - Cf. [Cesa-Bianchi & Fisher, 1998](http://dl.acm.org/citation.cfm?id=657473).
        - Default value for is :math:`c = \sqrt(\frac{\log(K)}{K})`.
        """
        c = np.sqrt(np.log(self.nbArms) / self.nbArms)
        return c * np.log(self.t) / self.t


# --- Other variants


DELTA = 0.01  #: Default value for the confidence parameter delta


class Exp3ELM(Exp3):
    r""" A variant of Exp3, apparently designed to work better in stochastic environments.

    - Reference: [Evaluation and Analysis of the Performance of the EXP3 Algorithm in Stochastic Environments, Y. Seldin & C. Szepasvari & P. Auer & Y. Abbasi-Adkori, 2012](http://proceedings.mlr.press/v24/seldin12a/seldin12a.pdf).
    """

    def __init__(self, nbArms, delta=DELTA, unbiased=True, lower=0., amplitude=1.):
        super(Exp3ELM, self).__init__(nbArms, unbiased=unbiased, lower=lower, amplitude=amplitude)
        assert delta > 0, "Error: the 'delta' parameter for Exp3ELM class has to be > 0."
        self.delta = delta  #: Confidence parameter, given in input
        self.B = 4 * (np.exp(2) - 2.) * (2 * np.log(nbArms) + np.log(2. / delta))  #: Constant B given by :math:`B = 4 (e - 2) (2 \log K + \log(2 / \delta))`.
        self.availableArms = np.arange(nbArms)  #: Set of available arms, starting from all arms, and it can get reduced at each step.
        self.varianceTerm = np.zeros(nbArms)  #: Estimated variance term, for each arm.

    def __str__(self):
        return r"Exp3ELM($\delta={:.3g}$)".format(self.delta)

    def choice(self):
        """ Choose among the remaining arms."""
        # Force to first visit each arm once in the first steps
        if self.t < self.nbArms:
            return self._initial_exploration[self.t]
        else:
            p = self.trusts[self.availableArms]
            return rn.choice(self.availableArms, p=p / np.sum(p))

    def getReward(self, arm, reward):
        r""" Get reward and update the weights, as in Exp3, but also update the variance term :math:`V_k(t)` for all arms, and the set of available arms :math:`\mathcal{A}(t)`, by removing arms whose empirical accumulated reward and variance term satisfy a certain inequality.

        .. math::

           a^*(t+1) &= \arg\max_a \hat{R}_{a}(t+1), \\
           V_k(t+1) &= V_k(t) + \frac{1}{\mathrm{trusts}_k(t+1)}, \\
           \mathcal{A}(t+1) &= \mathcal{A}(t) \setminus \left\{ a : \hat{R}_{a^*(t+1)}(t+1) - \hat{R}_{a}(t+1) > \sqrt{B (V_{a^*(t+1)}(t+1) + V_{a}(t+1))} \right\}.
        """
        assert arm in self.availableArms, "Error: at time {}, the arm {} was played by Exp3ELM but it is not in the set of remaining arms {}...".format(self.t, arm, self.availableArms)  # DEBUG
        # First, use the reward to update the weights
        self.t += 1
        self.pulls[arm] += 1

        reward = (reward - self.lower) / self.amplitude
        # Update weight of THIS arm, with this biased or unbiased reward
        if self.unbiased:
            reward = reward / self.trusts[arm]
        self.rewards[arm] += reward

        # Multiplicative weights
        self.weights[arm] *= np.exp(reward * self.gamma)
        # Renormalize weights at each step
        self.weights[self.availableArms] /= np.sum(self.weights[self.availableArms])

        # Then update the variance
        self.varianceTerm[self.availableArms] += 1. / self.trusts[self.availableArms]

        # And update the set of available arms
        a_star = np.argmax(self.rewards[self.availableArms])
        # print("- Exp3ELM identified the arm of best accumulated rewards to be {}, at time {} ...".format(a_star, self.t))  # DEBUG
        test = (self.rewards[a_star] - self.rewards[self.availableArms]) > np.sqrt(self.B * (self.varianceTerm[a_star] + self.varianceTerm[self.availableArms]))
        badArms = np.where(test)[0]
        # Do we have bad arms ? If yes, remove them
        if len(badArms) > 0:
            print("- Exp3ELM identified these arms to be bad at time {} : {}, removing them from the set of available arms ...".format(self.t, badArms))  # DEBUG
            self.availableArms = np.setdiff1d(self.availableArms, badArms)

        # # DEBUG
        # print("- Exp3ELM at time {} as this internal memory:\n  - B = {} and delta = {}\n  - Pulls {}\n  - Rewards {}\n  - Weights {}\n  - Variance {}\n  - Trusts {}\n  - a_star {}\n  - Left part of test {}\n  - Right part of test {}\n  - test {}\n  - Bad arms {}\n  - Available arms {}".format(self.t, self.B, self.delta, self.pulls, self.rewards, self.weights, self.varianceTerm, self.trusts, a_star, (self.rewards[a_star] - self.rewards[self.availableArms]), np.sqrt(self.B * (self.varianceTerm[a_star] + self.varianceTerm[self.availableArms])), test, badArms, self.availableArms))  # DEBUG
        # print(input("[Enter to keep going on]"))  # DEBUG

    # --- Trusts and gamma coefficient

    @property
    def trusts(self):
        r""" Update the trusts probabilities according to Exp3ELM formula, and the parameter :math:`\gamma_t`.

        .. math::

           \mathrm{trusts}'_k(t+1) &= (1 - |\mathcal{A}_t| \gamma_t) w_k(t) + \gamma_t, \\
           \mathrm{trusts}(t+1) &= \mathrm{trusts}'(t+1) / \sum_{k=1}^{K} \mathrm{trusts}'_k(t+1).

        If :math:`w_k(t)` is the current weight from arm k.
        """
        # Mixture between the weights and the uniform distribution
        trusts = ((1 - self.gamma * len(self.availableArms)) * self.weights[self.availableArms]) + self.gamma
        # XXX Handle weird cases, slow down everything but safer!
        if not np.all(np.isfinite(trusts)):
            # XXX some value has non-finite trust, probably on the first steps
            # 1st case: all values are non-finite (nan): set trusts to 1/N uniform choice
            if np.all(~np.isfinite(trusts)):
                trusts = np.full(len(self.availableArms), 1. / len(self.availableArms))
            # 2nd case: only few values are non-finite: set them to 0
            else:
                trusts[~np.isfinite(trusts)] = 0
        # Bad case, where the sum is so small that it's only rounding errors
        if np.isclose(np.sum(trusts), 0):
                trusts = np.full(len(self.availableArms), 1. / len(self.availableArms))
        # Normalize it and return it
        return trusts
        # return trusts / np.sum(trusts[self.availableArms])

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def gamma(self):
        r""" Decreasing gamma with the time: :math:`\gamma_t = \min(\frac{1}{K}, \sqrt(\frac{\log(K)}{t K}))` (*heuristic*).

        - Cf. Theorem 3.1 case #2 of [Bubeck & Cesa-Bianchi, 2012](http://sbubeck.com/SurveyBCB12.pdf).
        """
        return min(1. / self.nbArms, np.sqrt(np.log(self.nbArms) / (self.t * self.nbArms)))
