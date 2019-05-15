# -*- coding: utf-8 -*-
r""" The historical Exp3.S algorithm for non-stationary bandits.

- Reference: [["The nonstochastic multiarmed bandit problem", P. Auer, N. Cesa-Bianchi, Y. Freund, R.E. Schapire, SIAM journal on computing, 2002]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.21.8735&rep=rep1&type=pdf)
- It is a simple extension of the :class:`Exp3` policy:

    >>> policy = Exp3S(nbArms, C=1)
    >>> # use policy as usual, with policy.startGame(), r = policy.choice(), policy.getReward(arm, r)

- It uses an additional :math:`\mathcal{O}(\tau_\max)` memory for a game of maximum stationary length :math:`\tau_\max`.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
import numpy.random as rn
from math import log, sqrt

try:
    from .Exp3 import Exp3
except ImportError:
    from Exp3 import Exp3


CONSTANT_e = np.exp(1)


# --- Exp3S

class Exp3S(Exp3):
    r""" The historical Exp3.S algorithm for non-stationary bandits.

    - Reference: [["The nonstochastic multiarmed bandit problem", P. Auer, N. Cesa-Bianchi, Y. Freund, R.E. Schapire, SIAM journal on computing, 2002]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.21.8735&rep=rep1&type=pdf)
    """

    def __init__(self, nbArms,
                gamma=None, alpha=None,
                gamma0=1.0, alpha0=1.0,
                horizon=None, max_nb_random_events=None,
                 *args, **kwargs):
        super(Exp3S, self).__init__(nbArms, *args, **kwargs)
        # set default values of gamma and alpha
        if gamma is None:  # Use a default value for the gamma parameter
            if horizon is None:
                gamma = np.sqrt(np.log(nbArms) / nbArms)
                print("For Exp3S: by using the formula with unknown T and unknown Upsilon_T, gamma = {}...".format(gamma))  # DEBUG
            elif max_nb_random_events is None:
                gamma = min(1.0, np.sqrt(nbArms * np.log(nbArms * horizon) / horizon))
                print("For Exp3S: by using the formula with known T but unknown Upsilon_T, gamma = {}...".format(gamma))  # DEBUG
            else:
                # Corollary 8.3
                gamma = min(1.0, np.sqrt(nbArms * (max_nb_random_events * np.log(nbArms * horizon) + CONSTANT_e) / ((CONSTANT_e - 1.0) * horizon)))
                print("For Exp3S: by using the formula with known T and Upsilon_T, gamma = {}...".format(gamma))  # DEBUG
        if gamma0 is not None and gamma0 >= 0:
            gamma *= gamma0
        if gamma > 1:  # XXX for to be in [0, 1]
            gamma = 1.0
        assert 0 <= gamma <= 1, "Error: the parameter 'gamma' for the Exp3S class has to be in [0,1] !"  # DEBUG
        self._gamma = gamma
        if alpha is None:  # Use a default value for the alpha parameter
            if horizon is None:
                alpha = 0.001  # XXX bad!
            else:
                alpha = 1.0 / horizon
                print("For Exp3S: by using the formula with known T, alpha = {}...".format(alpha))  # DEBUG
        if alpha0 is not None and alpha0 >= 0:
            alpha *= alpha0
        assert alpha >= 0, "Error: the parameter 'alpha' for the Exp3S class has to be > 0!"  # DEBUG
        self._alpha = alpha
        # Just store horizon and max_nb_random_events, for pretty printing?
        self.horizon = horizon if horizon is not None else "?"
        self.max_nb_random_events = max_nb_random_events if max_nb_random_events is not None else "?"
        # Internal memory
        self.weights = np.full(nbArms, 1. / nbArms)  #: Weights on the arms

    def __str__(self):
        # return r"Exp3.S($T={}$, $\Upsilon_T={}$, $\alpha={:.6g}$, $\gamma={:.6g}$)".format(self.horizon, self.max_nb_random_events, self._alpha, self._gamma)
        return "Exp3.S"

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def gamma(self):
        r"""Constant :math:`\gamma_t = \gamma`."""
        return self._gamma

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def alpha(self):
        r"""Constant :math:`\alpha_t = \alpha`."""
        return self._alpha

    def startGame(self):
        """Start with uniform weights."""
        super(Exp3S, self).startGame()
        self.weights.fill(1. / self.nbArms)

    @property
    def trusts(self):
        r"""Update the trusts probabilities according to Exp3 formula, and the parameter :math:`\gamma_t`.

        .. math::

           \mathrm{trusts}'_k(t+1) &= (1 - \gamma_t) w_k(t) + \gamma_t \frac{1}{K}, \\
           \mathrm{trusts}(t+1) &= \mathrm{trusts}'(t+1) / \sum_{k=1}^{K} \mathrm{trusts}'_k(t+1).

        If :math:`w_k(t)` is the current weight from arm k.
        """
        # Mixture between the weights and the uniform distribution
        trusts = ((1 - self.gamma) * self.weights / np.sum(self.weights)) + (self.gamma / self.nbArms)
        # XXX Handle weird cases, slow down everything but safer!
        if not np.all(np.isfinite(trusts)):
            trusts[~np.isfinite(trusts)] = 0  # set bad values to 0
        # Bad case, where the sum is so small that it's only rounding errors
        # or where all values where bad and forced to 0, start with trusts=[1/K...]
        if np.isclose(np.sum(trusts), 0):
            trusts[:] = 1.0 / self.nbArms
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
        # XXX DONT call getReward of Exp3 (mother class), it already updates the weights!
        # super(Exp3S, self).getReward(arm, reward)  # XXX Call to BasePolicy
        # XXX Manually copy/paste from BasePolicy.getReward
        self.t += 1
        self.pulls[arm] += 1
        reward = (reward - self.lower) / self.amplitude
        self.rewards[arm] += reward
        # Update weight of THIS arm, with this biased or unbiased reward
        if self.unbiased:
            reward = reward / self.trusts[arm]
        # Multiplicative weights + uniform share of previous weights (alpha is used for this)
        old_weights = self.weights[:]
        sum_of_weights = np.sum(old_weights)
        for otherArm in range(self.nbArms):
            if otherArm != arm:
                self.weights[otherArm] = old_weights[otherArm] + CONSTANT_e * (self.alpha / self.nbArms) * sum_of_weights
        self.weights[arm] = old_weights[arm] * np.exp(reward * (self.gamma / self.nbArms)) + CONSTANT_e * (self.alpha / self.nbArms) * sum_of_weights
        # WARNING DONT Renormalize weights at each step !!
        # self.weights /= np.sum(self.weights)
