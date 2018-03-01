# -*- coding: utf-8 -*-
""" The Hedge randomized index policy.

Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi](http://research.microsoft.com/en-us/um/people/sebubeck/SurveyBCB12.pdf)
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.7"

import numpy as np
import numpy.random as rn
from .BasePolicy import BasePolicy

#: Default :math:`\varepsilon` parameter.
EPSILON = 0.01


class Hedge(BasePolicy):
    """ The Hedge randomized index policy.

    Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi, §3.1](http://research.microsoft.com/en-us/um/people/sebubeck/SurveyBCB12.pdf).
    """

    def __init__(self, nbArms, epsilon=EPSILON,
                 lower=0., amplitude=1.):
        super(Hedge, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        if epsilon is None:  # Use a default value for the epsilon parameter
            epsilon = np.sqrt(np.log(nbArms) / nbArms)
        assert 0 < epsilon <= 1, "Error: the 'epsilon' parameter for Hedge class has to be in (0, 1]."  # DEBUG
        self._epsilon = epsilon
        # Internal memory
        self.weights = np.full(nbArms, 1. / nbArms)  #: Weights on the arms
        # trying to randomize the order of the initial visit to each arm; as this determinism breaks its habitility to play efficiently in multi-players games
        # XXX do even more randomized, take a random permutation of the arm ?
        self._initial_exploration = rn.permutation(nbArms)
        # The proba that another player has the same is nbPlayers / factorial(nbArms) : should be SMALL !

    def startGame(self):
        """Start with uniform weights."""
        super(Hedge, self).startGame()
        self.weights.fill(1. / self.nbArms)

    def __str__(self):
        return r"Hedge($\varepsilon: {:.3g}$)".format(self.epsilon)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):
        r"""Constant :math:`\varepsilon_t = \varepsilon`."""
        return self._epsilon

    @property
    def trusts(self):
        r"""Update the trusts probabilities according to Hedge formula, and the parameter :math:`\varepsilon_t`.

        .. math::

           \mathrm{trusts}'_k(t+1) &= (1 - \varepsilon_t) w_k(t) + \varepsilon_t \frac{1}{K}, \\
           \mathrm{trusts}(t+1) &= \mathrm{trusts}'(t+1) / \sum_{k=1}^{K} \mathrm{trusts}'_k(t+1).

        If :math:`w_k(t)` is the current weight from arm k.
        """
        # Mixture between the weights and the uniform distribution
        trusts = self.weights
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

        .. math::

           w'_k(t+1) &= w_k(t) \times \exp\left( \frac{\tilde{r}_k(t)}{\varepsilon_t N_k(t)} \right) \\
           w(t+1) &= w'(t+1) / \sum_{k=1}^{K} w'_k(t+1).
        """
        super(Hedge, self).getReward(arm, reward)  # XXX Call to BasePolicy
        # Update weight of THIS arm, with this reward
        reward = (reward - self.lower) / self.amplitude
        loss = 1 - reward
        # Multiplicative weights
        self.weights[arm] *= np.exp(- loss * self.epsilon)
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
        if (self.t < self.nbArms) or (nb == 1):
            return np.array([self.choice() for _ in range(nb)])  # good size if nb > 1 but t < nbArms
        else:
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


# --- Two special cases

class HedgeWithHorizon(Hedge):
    r""" Hedge with fixed epsilon, :math:`\varepsilon_t = \varepsilon_0`, chosen with a knowledge of the horizon."""

    def __init__(self, nbArms, horizon, lower=0., amplitude=1.):
        super(HedgeWithHorizon, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert horizon > 0, "Error: the 'horizon' parameter for SoftmaxWithHorizon class has to be > 0."
        self.horizon = int(horizon) if horizon is not None else None  #: Parameter :math:`T` = known horizon of the experiment.

    def __str__(self):
        return r"Hedge($T={}$)".format(self.horizon)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):
        r""" Fixed temperature, small, knowing the horizon: :math:`\varepsilon_t = \sqrt(\frac{2 \log(K)}{T K})` (*heuristic*).

        - Cf. Theorem 3.1 case #1 of [Bubeck & Cesa-Bianchi, 2012](http://sbubeck.com/SurveyBCB12.pdf).
        """
        return np.sqrt(2 * np.log(self.nbArms) / (self.horizon * self.nbArms))


class HedgeDecreasing(Hedge):
    r""" Hedge with decreasing parameter :math:`\varepsilon_t`."""

    def __str__(self):
        return "Hedge(decreasing)"

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):
        r""" Decreasing epsilon with the time: :math:`\varepsilon_t = \min(\frac{1}{K}, \sqrt(\frac{\log(K)}{t K}))` (*heuristic*).

        - Cf. Theorem 3.1 case #2 of [Bubeck & Cesa-Bianchi, 2012](http://sbubeck.com/SurveyBCB12.pdf).
        """
        return min(1. / self.nbArms, np.sqrt(np.log(self.nbArms) / (self.t * self.nbArms)))
