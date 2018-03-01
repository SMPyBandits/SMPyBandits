# -*- coding: utf-8 -*-
""" The Boltzmann Exploration (Softmax) index policy.

- Reference: [Algorithms for the multi-armed bandit problem, V.Kuleshov & D.Precup, JMLR, 2008, §2.1](http://www.cs.mcgill.ca/~vkules/bandits.pdf) and [Boltzmann Exploration Done Right, N.Cesa-Bianchi & C.Gentile & G.Lugosi & G.Neu, arXiv 2017](https://arxiv.org/pdf/1705.10257.pdf).

- Very similar to Exp3 but uses a Boltzmann distribution.
  Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi, §3.1](http://sbubeck.com/SurveyBCB12.pdf)
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
import numpy.random as rn
from .BasePolicy import BasePolicy

#: self.unbiased is a flag to know if the rewards are used as biased estimator,
#: i.e., just :math:`r_t`, or unbiased estimators, :math:`r_t / trusts_t`.
UNBIASED = True
UNBIASED = False


class Softmax(BasePolicy):
    r"""The Boltzmann Exploration (Softmax) index policy, with a constant temperature :math:`\eta_t`.

    - Reference: [Algorithms for the multi-armed bandit problem, V.Kuleshov & D.Precup, JMLR, 2008, §2.1](http://www.cs.mcgill.ca/~vkules/bandits.pdf) and [Boltzmann Exploration Done Right, N.Cesa-Bianchi & C.Gentile & G.Lugosi & G.Neu, arXiv 2017](https://arxiv.org/pdf/1705.10257.pdf).

    - Very similar to Exp3 but uses a Boltzmann distribution.
      Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi, §3.1](http://sbubeck.com/SurveyBCB12.pdf)
    """

    def __init__(self, nbArms, temperature=None, unbiased=UNBIASED, lower=0., amplitude=1.):
        super(Softmax, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        if temperature is None:  # Use a default value for the temperature
            temperature = np.sqrt(np.log(nbArms) / nbArms)
        assert temperature > 0, "Error: the temperature parameter for Softmax class has to be > 0."
        self._temperature = temperature
        self.unbiased = unbiased  #: Flag
        # Trying to randomize the order of the initial visit to each arm; as this determinism breaks its habitility to play efficiently in multi-players games
        # Even more randomized, take a random permutation of the arm ?
        self._initial_exploration = rn.permutation(nbArms)
        # The proba that another player has the same is nbPlayers / factorial(nbArms) : should be SMALL !

    def startGame(self):
        """Nothing special to do."""
        super(Softmax, self).startGame()

    def __str__(self):
        return "Softmax(temp: {})".format(self.temperature)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def temperature(self):
        r"""Constant temperature, :math:`\eta_t`."""
        return self._temperature

    @property
    def trusts(self):
        r"""Update the trusts probabilities according to the Softmax (ie Boltzmann) distribution on accumulated rewards, and with the temperature :math:`\eta_t`.

        .. math::

           \mathrm{trusts}'_k(t+1) &= \exp\left( \frac{X_k(t)}{\eta_t N_k(t)} \right) \\
           \mathrm{trusts}(t+1) &= \mathrm{trusts}'(t+1) / \sum_{k=1}^{K} \mathrm{trusts}'_k(t+1).

        If :math:`X_k(t) = \sum_{\sigma=1}^{t} 1(A(\sigma) = k) r_k(\sigma)` is the sum of rewards from arm k.
        """
        # rewards = (self.rewards - self.lower) / self.amplitude  # XXX we don't need this, the BasePolicy.getReward does it already
        rewards = self.rewards
        trusts = np.exp(rewards / (self.temperature * (1 + self.pulls)))  # 1 + pulls to prevent division by 0
        if self.unbiased:
            rewards = self.rewards / trusts
            trusts = np.exp(rewards / (self.temperature * (1 + self.pulls)))  # 1 + pulls to prevent division by 0
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

    # --- Choice methods

    def choice(self):
        """One random selection, with probabilities = trusts, thank to :func:`numpy.random.choice`."""
        # Force to first visit each arm once in the first steps
        if self.t < self.nbArms:
            # return self.t  # random permutation instead of deterministic order!
            return self._initial_exploration[self.t]  # DONE
        else:
            return rn.choice(self.nbArms, p=self.trusts)

    def choiceWithRank(self, rank=1):
        """Multiple (rank >= 1) random selection, with probabilities = trusts, thank to :func:`numpy.random.choice`, and select the last one (least probable one).

        - Note that if not enough entries in the trust vector are non-zero, then :func:`choice` is called instead (rank is ignored).
        """
        if (self.t < self.nbArms) or (rank == 1) or np.sum(~np.isclose(self.trusts, 0)) < rank:
            return self.choice()
        else:
            return rn.choice(self.nbArms, size=rank, replace=False, p=self.trusts)[-1]

    def choiceFromSubSet(self, availableArms='all'):
        """One random selection, from availableArms, with probabilities = trusts, thank to :func:`numpy.random.choice`."""
        if self.t < self.nbArms:
            return availableArms[self.t % len(availableArms)]
        elif (availableArms == 'all') or (len(availableArms) == self.nbArms):
            return self.choice()
        else:
            return rn.choice(availableArms, p=self.trusts[availableArms])

    def choiceMultiple(self, nb=1):
        """Multiple (nb >= 1) random selection, with probabilities = trusts, thank to :func:`numpy.random.choice`."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            return rn.choice(self.nbArms, size=nb, replace=False, p=self.trusts)

    # --- Other methods

    def estimatedOrder(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing trust probabilities."""
        return np.argsort(self.trusts)


# --- Three special cases

class SoftmaxWithHorizon(Softmax):
    r""" Softmax with fixed temperature :math:`\eta_t = \eta_0` chosen with a knowledge of the horizon."""

    def __init__(self, nbArms, horizon, lower=0., amplitude=1.):
        super(SoftmaxWithHorizon, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert horizon > 0, "Error: the 'horizon' parameter for SoftmaxWithHorizon class has to be > 0."
        self.horizon = int(horizon) if horizon is not None else None  #: Parameter :math:`T` = known horizon of the experiment.

    def __str__(self):
        return "Softmax($T={}$)".format(self.horizon)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def temperature(self):
        r""" Fixed temperature, small, knowing the horizon: :math:`\eta_t = \sqrt(\frac{2 \log(K)}{T K})` (*heuristic*).

        - Cf. Theorem 3.1 case #1 of [Bubeck & Cesa-Bianchi, 2012](http://sbubeck.com/SurveyBCB12.pdf).
        """
        return np.sqrt(2 * np.log(self.nbArms) / (self.horizon * self.nbArms))


class SoftmaxDecreasing(Softmax):
    r""" Softmax with decreasing temperature :math:`\eta_t`."""

    def __str__(self):
        return "Softmax(decreasing)"

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def temperature(self):
        r""" Decreasing temperature with the time: :math:`\eta_t = \sqrt(\frac{\log(K)}{t K})` (*heuristic*).

        - Cf. Theorem 3.1 case #2 of [Bubeck & Cesa-Bianchi, 2012](http://sbubeck.com/SurveyBCB12.pdf).
        """
        return np.sqrt(np.log(self.nbArms) / (self.t * self.nbArms))


class SoftMix(Softmax):
    r""" Another Softmax with decreasing temperature :math:`\eta_t`."""

    def __str__(self):
        return "SoftMix"

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def temperature(self):
        r""" Decreasing temperature with the time: :math:`\eta_t = c \frac{\log(t)}{t}` (*heuristic*).

        - Cf. [Cesa-Bianchi & Fisher, 1998](http://dl.acm.org/citation.cfm?id=657473).
        - Default value for is :math:`c = \sqrt(\frac{\log(K)}{K})`.
        """
        c = np.sqrt(np.log(self.nbArms) / self.nbArms)
        if self.t <= 1:
            return c
        else:
            return c * np.log(self.t) / self.t
