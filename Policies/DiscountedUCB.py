# -*- coding: utf-8 -*-
r""" The Discounted-UCB index policy, with a discount factor of :math:`\gamma\in(0,1]`.

- Reference: ["On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems", by A.Garivier & E.Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)
- :math:`\gamma` should not be 1, otherwise you should rather use :class:`Policies.UCBalpha.UCBalpha` instead.
- The smaller the :math:`\gamma`, the shorter the *"memory"* of the algorithm is.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

from math import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .UCBalpha import UCBalpha

#: Default parameter for alpha.
ALPHA = 4


#: Default parameter for gamma.
GAMMA = 1.0


class DiscountedUCB(UCBalpha):
    r""" The Discounted-UCB index policy, with a discount factor of :math:`\gamma\in(0,1]`.

    - Reference: ["On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems", by A.Garivier & E.Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)
    """

    def __init__(self, nbArms,
                 alpha=ALPHA, gamma=GAMMA,
                 lower=0., amplitude=1., *args, **kwargs):
        super(DiscountedUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude, *args, **kwargs)
        assert alpha >= 0, "Error: the 'alpha' parameter for DiscountedUCB class has to be >= 0."  # DEBUG
        self.alpha = alpha  #: Parameter alpha
        assert 0 < gamma <= 1, "Error: the 'gamma' parameter for DiscountedUCB class has to be 0 < gamma <= 1."  # DEBUG
        if np.isclose(gamma, 1):
            print("Warning: using DiscountedUCB with 'gamma' too close to 1 will result in UCBalpha, you should rather use it...")  # DEBUG
        self.gamma = gamma  #: Parameter gamma

    def __str__(self):
        return r"D-UCB($\alpha={:.3g}$, $\gamma={:.5g}$)".format(self.alpha, self.gamma)

    def getReward(self, arm, reward):
        r""" Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1]).

        - Keep up-to date the following two quantities, using different definition and notation as from the article, but being consistent w.r.t. my project:

        .. math::

           N_{k,\gamma}(t+1) &:= \sum_{s=1}^{t} \gamma^{t - s} N_k(s), \\
           X_{k,\gamma}(t+1) &:= \sum_{s=1}^{t} \gamma^{t - s} X_k(s).

        - Instead of keeping the whole history of rewards, as expressed in the math formula, we keep the sum of discounted rewards from `s=0` to `s=t`, because updating it is easy (2 operations instead of just 1 for classical :class:`Policies.UCBalpha.UCBalpha`, and 2 operations instead of :math:`\mathcal{O}(t)` as expressed mathematically).

        .. math::

           N_{k,\gamma}(t+1) &= \gamma \times N_{k,\gamma}(t) + \mathbb{1}(A(t+1) = k), \\
           X_{k,\gamma}(t+1) &= \gamma \times X_{k,\gamma}(t) + X_k(t+1).
        """
        self.t += 1
        self.pulls[arm] = (self.gamma * self.pulls[arm]) + 1
        reward = (reward - self.lower) / self.amplitude
        self.rewards[arm] = (self.gamma * self.rewards[arm]) + reward

    def computeIndex(self, arm):
        r""" Compute the current index, at time :math:`t` and after :math:`N_{k,\gamma}(t)` *"discounted"* pulls of arm k, and :math:`n_{\gamma}(t)` *"discounted"* pulls of all arms:

        .. math::

           I_k(t) &:= \frac{X_{k,\gamma}(t)}{N_{k,\gamma}(t)} + \sqrt{\frac{\alpha \log(n_{\gamma}(t))}{2 N_{k,\gamma}(t)}}, \\
           \text{where}\;\; n_{\gamma}(t) &:= \sum_{k=1}^{K} N_{k,\gamma}(t).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            n_t_gamma = np.sum(self.pulls)
            assert n_t_gamma <= self.t, "Error: n_t_gamma was computed as {:.3g} but should be < t = {:.3g}...".format(n_t_gamma, self.t)  # DEBUG
            return (self.rewards[arm] / self.pulls[arm]) + sqrt((self.alpha * log(n_t_gamma)) / (2 * self.pulls[arm]))

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        n_t_gamma = np.sum(self.pulls)
        assert n_t_gamma <= self.t, "Error: n_t_gamma was computed as {:.3g} but should be < t = {:.3g}...".format(n_t_gamma, self.t)  # DEBUG
        indexes = (self.rewards / self.pulls) + np.sqrt((self.alpha * np.log(n_t_gamma)) / (2 * self.pulls))
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes


# --- Horizon dependent version

class DiscountedUCBPlus(DiscountedUCB):
    r""" The Discounted-UCB index policy, with a particular value of the discount factor of :math:`\gamma\in(0,1]`, knowing the horizon and the number of breakpoints (or an upper-bound).

    - Reference: ["On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems", by A.Garivier & E.Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)
    - Uses :math:`\gamma =  1 - \frac{1}{4}\sqrt{\frac{\Upsilon}{T}}`, if the horizon :math:`T` is given and an upper-bound on the number of random events ("breakpoints") :math:`\Upsilon` is known, otherwise use the default value.
    """

    def __init__(self, nbArms,
                 horizon=None, max_nb_random_events=None,
                 alpha=ALPHA,
                 lower=0., amplitude=1., *args, **kwargs):
        # New parameter
        if horizon is not None and max_nb_random_events is not None:
            gamma = 1 - np.sqrt(max_nb_random_events / horizon) / 4.
            if gamma > 1 or gamma <= 0:
                gamma = 1.
        else:
            gamma = GAMMA
        super(DiscountedUCBPlus, self).__init__(nbArms, alpha=alpha, gamma=gamma, lower=lower, amplitude=amplitude, *args, **kwargs)

    def __str__(self):
        return r"D-UCB+($\alpha={:.3g}$, $\gamma={:.5g}$)".format(self.alpha, self.gamma)
