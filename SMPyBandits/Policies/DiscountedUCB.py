# -*- coding: utf-8 -*-
r""" The Discounted-UCB index policy, with a discount factor of :math:`\gamma\in(0,1]`.

- Reference: ["On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems", by A.Garivier & E.Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)
- :math:`\gamma` should not be 1, otherwise you should rather use :class:`Policies.UCBalpha.UCBalpha` instead.
- The smaller the :math:`\gamma`, the shorter the *"memory"* of the algorithm is.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from math import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .UCBalpha import UCBalpha
except ImportError:
    from UCBalpha import UCBalpha

#: Default parameter for alpha.
ALPHA = 1

#: Default parameter for gamma.
GAMMA = 0.99


class DiscountedUCB(UCBalpha):
    r""" The Discounted-UCB index policy, with a discount factor of :math:`\gamma\in(0,1]`.

    - Reference: ["On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems", by A.Garivier & E.Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)
    """

    def __init__(self, nbArms,
                 alpha=ALPHA, gamma=GAMMA,
                 useRealDiscount=True,
                 *args, **kwargs):
        super(DiscountedUCB, self).__init__(nbArms, *args, **kwargs)
        self.discounted_pulls = np.zeros(nbArms)  #: Number of pulls of each arms
        self.discounted_rewards = np.zeros(nbArms)  #: Cumulated rewards of each arms
        assert alpha >= 0, "Error: the 'alpha' parameter for DiscountedUCB class has to be >= 0."  # DEBUG
        self.alpha = alpha  #: Parameter alpha
        assert 0 < gamma <= 1, "Error: the 'gamma' parameter for DiscountedUCB class has to be 0 < gamma <= 1."  # DEBUG
        if np.isclose(gamma, 1):
            print("Warning: using DiscountedUCB with 'gamma' too close to 1 will result in UCBalpha, you should rather use it...")  # DEBUG
        self.gamma = gamma  #: Parameter gamma
        self.delta_time_steps = np.zeros(self.nbArms, dtype=int)  #: Keep memory of the :math:`\Delta_k(t)` for each time step.
        self.useRealDiscount = useRealDiscount  #: Flag to know if the real update should be used, the one with a multiplication by :math:`\gamma^{1+\Delta_k(t)}` and not simply a multiplication by :math:`\gamma`.

    def __str__(self):
        return r"D-UCB({}$\gamma={:.5g}${})".format(
            "no delay, " if not self.useRealDiscount else "",
            self.gamma,
            ", $\alpha={:.3g}$".format(self.alpha) if self.alpha != ALPHA else "",
        )

    def getReward(self, arm, reward):
        r""" Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1]).

        - Keep up-to date the following two quantities, using different definition and notation as from the article, but being consistent w.r.t. my project:

        .. math::

            N_{k,\gamma}(t+1) &:= \sum_{s=1}^{t} \gamma^{t - s} N_k(s), \\
            X_{k,\gamma}(t+1) &:= \sum_{s=1}^{t} \gamma^{t - s} X_k(s).

        - Instead of keeping the whole history of rewards, as expressed in the math formula, we keep the sum of discounted rewards from ``s=0`` to ``s=t``, because updating it is easy (2 operations instead of just 1 for classical :class:`Policies.UCBalpha.UCBalpha`, and 2 operations instead of :math:`\mathcal{O}(t)` as expressed mathematically). Denote :math:`\Delta_k(t)` the number of time steps during which the arm ``k`` was *not* selected (maybe 0 if it is selected twice in a row). Then the update can be done easily by multiplying by :math:`\gamma^{1+\Delta_k(t)}`:

        .. math::

            N_{k,\gamma}(t+1) &= \gamma^{1+\Delta_k(t)} \times N_{k,\gamma}(\text{last pull}) + \mathbb{1}(A(t+1) = k), \\
            X_{k,\gamma}(t+1) &= \gamma^{1+\Delta_k(t)} \times X_{k,\gamma}(\text{last pull}) + X_k(t+1).
        """
        super(DiscountedUCB, self).getReward(arm, reward)
        # FIXED we should multiply by gamma^delta where delta is the number of time steps where we didn't play this arm, +1
        self.discounted_pulls *= self.gamma
        self.discounted_rewards *= self.gamma
        self.discounted_pulls[arm] += 1
        reward = (reward - self.lower) / self.amplitude
        self.discounted_rewards[arm] += reward
        # XXX self.discounted_pulls[arm] += 1  # if we were using N_k(t) and not N_{k,gamma}(t).
        # Ok and we saw this arm so no delta now
        if self.useRealDiscount:
            self.delta_time_steps += 1  # increase delay for each algorithms
            self.delta_time_steps[arm] = 0

    def computeIndex(self, arm):
        r""" Compute the current index, at time :math:`t` and after :math:`N_{k,\gamma}(t)` *"discounted"* pulls of arm k, and :math:`n_{\gamma}(t)` *"discounted"* pulls of all arms:

        .. math::

            I_k(t) &:= \frac{X_{k,\gamma}(t)}{N_{k,\gamma}(t)} + \sqrt{\frac{\alpha \log(n_{\gamma}(t))}{2 N_{k,\gamma}(t)}}, \\
            \text{where}\;\; n_{\gamma}(t) &:= \sum_{k=1}^{K} N_{k,\gamma}(t).
        """
        if self.discounted_pulls[arm] < 1:
            return float('+inf')
        else:
            n_t_gamma = np.sum(self.discounted_pulls)
            assert n_t_gamma <= self.t, "Error: n_t_gamma was computed as {:.3g} but should be < t = {:.3g}...".format(n_t_gamma, self.t)  # DEBUG
            return (self.discounted_rewards[arm] / self.discounted_pulls[arm]) + sqrt((self.alpha * log(n_t_gamma)) / (2 * self.discounted_pulls[arm]))

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        n_t_gamma = np.sum(self.discounted_pulls)
        assert n_t_gamma <= self.t, "Error: n_t_gamma was computed as {:.3g} but should be < t = {:.3g}...".format(n_t_gamma, self.t)  # DEBUG
        indexes = (self.discounted_rewards / self.discounted_pulls) + np.sqrt((self.alpha * np.log(n_t_gamma)) / (2 * self.discounted_pulls))
        indexes[self.discounted_pulls < 1] = float('+inf')
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
                 *args, **kwargs):
        # New parameter
        if horizon is not None and max_nb_random_events is not None:
            gamma = 1 - np.sqrt(max_nb_random_events / horizon) / 4.
            if gamma > 1 or gamma <= 0:
                gamma = 1.
        else:
            gamma = GAMMA
        super(DiscountedUCBPlus, self).__init__(nbArms, alpha=alpha, gamma=gamma, *args, **kwargs)

    # def __str__(self):
    #     return r"D-UCB+($\alpha={:.3g}$, $\gamma={:.5g}$)".format(self.alpha, self.gamma)
    #     return r"D-UCB({}$\gamma={:.5g}${})".format(
    #         "no delay, " if not self.useRealDiscount else "",
    #         self.gamma,
    #         ", $\alpha={:.3g}$".format(self.alpha) if self.alpha != ALPHA else "",
    #     )


# --- SW-klUCB

try:
    from .kullback import klucbBern
except (ImportError, SystemError):
    from kullback import klucbBern

#: Default value for the constant c used in the computation of KL-UCB index.
constant_c = 1.  #: default value, as it was in pymaBandits v1.0
# c = 1.  #: as suggested in the Theorem 1 in https://arxiv.org/pdf/1102.2490.pdf


#: Default value for the tolerance for computing numerical approximations of the kl-UCB indexes.
tolerance = 1e-4


class DiscountedklUCB(DiscountedUCB):
    r""" The Discounted-klUCB index policy, with a particular value of the discount factor of :math:`\gamma\in(0,1]`, knowing the horizon and the number of breakpoints (or an upper-bound).

    - Reference: ["On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems", by A.Garivier & E.Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)
    """

    def __init__(self, nbArms, klucb=klucbBern, *args, **kwargs):
        super(DiscountedklUCB, self).__init__(nbArms, *args, **kwargs)
        self.klucb = klucb  #: kl function to use

    def __str__(self):
        name = self.klucb.__name__[5:]
        if name == "Bern": name = ""
        if name != "": name = "({})".format(name)
        return r"D-klUCB{}({}$\gamma={:.5g}$)".format(name, "no delay, " if not self.useRealDiscount else "", self.gamma)

    def computeIndex(self, arm):
        r""" Compute the current index, at time :math:`t` and after :math:`N_{k,\gamma}(t)` *"discounted"* pulls of arm k, and :math:`n_{\gamma}(t)` *"discounted"* pulls of all arms:

        .. math::

            \hat{\mu'}_k(t) &= \frac{X_{k,\gamma}(t)}{N_{k,\gamma}(t)} , \\
            U_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu'}_k(t), q) \leq \frac{c \log(t)}{N_{k,\gamma}(t)} \right\},\\
            I_k(t) &= U_k(t),\\
            \text{where}\;\; n_{\gamma}(t) &:= \sum_{k=1}^{K} N_{k,\gamma}(t).

        If rewards are in :math:`[a, b]` (default to :math:`[0, 1]`) and :math:`\mathrm{kl}(x, y)` is the Kullback-Leibler divergence between two distributions of means x and y (see :mod:`Arms.kullback`),
        and c is the parameter (default to 1).
        """
        if self.discounted_pulls[arm] < 1:
            return float('+inf')
        else:
            n_t_gamma = np.sum(self.discounted_pulls)
            assert n_t_gamma <= self.t, "Error: n_t_gamma was computed as {:.3g} but should be < t = {:.3g}...".format(n_t_gamma, self.t)  # DEBUG
            mean = self.discounted_rewards[arm] / self.discounted_pulls[arm]
            level = constant_c * log(n_t_gamma) / self.discounted_pulls[arm]
            return self.klucb(mean, level, tolerance)

    def computeAllIndex(self):
        """ Compute the current indexes for all arms. Possibly vectorized, by default it can *not* be vectorized automatically."""
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)



class DiscountedklUCBPlus(DiscountedklUCB, DiscountedUCBPlus):
    r""" The Discounted-klUCB index policy, with a particular value of the discount factor of :math:`\gamma\in(0,1]`, knowing the horizon and the number of breakpoints (or an upper-bound).

    - Reference: ["On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems", by A.Garivier & E.Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)
    - Uses :math:`\gamma =  1 - \frac{1}{4}\sqrt{\frac{\Upsilon}{T}}`, if the horizon :math:`T` is given and an upper-bound on the number of random events ("breakpoints") :math:`\Upsilon` is known, otherwise use the default value.
    """
    def __str__(self):
        name = self.klucb.__name__[5:]
        if name == "Bern": name = ""
        if name != "": name = "({})".format(name)
        return r"D-klUCB{}+({}$\gamma={:.5g}$)".format(name, "no delay, " if not self.useRealDiscount else "", self.gamma)
