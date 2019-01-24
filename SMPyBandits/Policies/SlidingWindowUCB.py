# -*- coding: utf-8 -*-
r""" An experimental policy, using only a sliding window (of for instance :math:`\tau=1000` *steps*, not counting draws of each arms) instead of using the full-size history.

- Reference: [On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems, by A.Garivier & E.Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)

- It uses an additional :math:`\mathcal{O}(\tau)` memory but do not cost anything else in terms of time complexity (the average is done with a sliding window, and costs :math:`\mathcal{O}(1)` at every time step).

.. warning:: This is very experimental!
.. note:: This is similar to :class:`SlidingWindowRestart.SWR_UCB` but slightly different: :class:`SlidingWindowRestart.SWR_UCB` uses a window of size :math:`T_0=100` to keep in memory the last 100 *draws* of *each* arm, and restart the index if the small history mean is too far away from the whole mean, while this :class:`SWUCB` uses a fixed-size window of size :math:`\tau=1000` to keep in memory the last 1000 *steps*.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from math import log, sqrt
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .IndexPolicy import IndexPolicy
except ImportError:
    from IndexPolicy import IndexPolicy


#: Size of the sliding window.
TAU = 1000

#: Default value for the constant :math:`\alpha`.
ALPHA = 1.0


# --- Manually written

class SWUCB(IndexPolicy):
    r""" An experimental policy, using only a sliding window (of for instance :math:`\tau=1000` *steps*, not counting draws of each arms) instead of using the full-size history.
    """

    def __init__(self, nbArms,
                 tau=TAU, alpha=ALPHA,
                 *args, **kwargs):
        super(SWUCB, self).__init__(nbArms, *args, **kwargs)
        # New parameters
        assert 1 <= tau, "Error: parameter 'tau' for class SWUCB has to be >= 1, but was {}.".format(tau)  # DEBUG
        self.tau = int(tau)  #: Size :math:`\tau` of the sliding window.
        assert alpha > 0, "Error: parameter 'alpha' for class SWUCB has to be > 0, but was {}.".format(alpha)  # DEBUG
        self.alpha = alpha  #: Constant :math:`\alpha` in the square-root in the computation for the index.
        # Internal memory
        self.last_rewards = np.zeros(tau)  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        self.last_choices = np.full(tau, -1)  #: Keep in memory the times where each arm was last seen.

    def __str__(self):
        return r"SW-UCB($\tau={}${})".format(
            self.tau,
            ", $\alpha={:.3g}$".format(self.alpha) if self.alpha != ALPHA else "",
        )

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).
        """
        now = self.t % self.tau
        # Get reward, normalized to [0, 1]
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_choices[now] = arm
        # Store it in place for the empirical average of that arm
        self.last_rewards[now] = reward
        self.t += 1

    def computeIndex(self, arm):
        r""" Compute the current index, at time :math:`t` and after :math:`N_{k,\tau}(t)` pulls of arm :math:`k`:

        .. math::

            I_k(t) &= \frac{X_{k,\tau}(t)}{N_{k,\tau}(t)} + c_{k,\tau}(t),\\
            \text{where}\;\; c_{k,\tau}(t) &:= \sqrt{\alpha \frac{\log(\min(t,\tau))}{N_{k,\tau}(t)}},\\
            \text{and}\;\; X_{k,\tau}(t) &:= \sum_{s=t-\tau+1}^{t} X_k(s) \mathbb{1}(A(t) = k),\\
            \text{and}\;\; N_{k,\tau}(t) &:= \sum_{s=t-\tau+1}^{t} \mathbb{1}(A(t) = k).
        """
        last_pulls_of_this_arm = np.count_nonzero(self.last_choices == arm)
        if last_pulls_of_this_arm < 1:
            return float('+inf')
        else:
            return (np.sum(self.last_rewards[self.last_choices == arm]) / last_pulls_of_this_arm) + sqrt((self.alpha * log(min(self.t, self.tau))) / last_pulls_of_this_arm)


# --- Horizon dependent version

class SWUCBPlus(SWUCB):
    r""" An experimental policy, using only a sliding window (of :math:`\tau` *steps*, not counting draws of each arms) instead of using the full-size history.

    - Uses :math:`\tau = 4 \sqrt{T \log(T)}` if the horizon :math:`T` is given, otherwise use the default value.
    """

    def __init__(self, nbArms, horizon=None,
                 *args, **kwargs):
        if horizon is not None:
            T = int(horizon)
            tau = int(4 * sqrt(T * log(T)))
        else:
            tau = TAU
        super(SWUCBPlus, self).__init__(nbArms, tau=tau, *args, **kwargs)
        # New parameter

    def __str__(self):
        return r"SW-UCB+($\tau={}$, $\alpha={:.3g}$)".format(self.tau, self.alpha)


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


class SWklUCB(SWUCB):
    r""" An experimental policy, using only a sliding window (of :math:`\tau` *steps*, not counting draws of each arms) instead of using the full-size history, and using klUCB (see :class:`Policy.klUCB`) indexes instead of UCB.
    """

    def __init__(self, nbArms, tau=TAU, klucb=klucbBern, *args, **kwargs):
        super(SWklUCB, self).__init__(nbArms, tau=tau, *args, **kwargs)
        self.klucb = klucb  #: kl function to use

    def __str__(self):
        name = self.klucb.__name__[5:]
        if name == "Bern": name = ""
        if name != "": name = "({})".format(name)
        return r"SW-klUCB{}($\tau={}$)".format(name, self.tau)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu'}_k(t) &= \frac{X_{k,\tau}(t)}{N_{k,\tau}(t)} , \\
            U_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu'}_k(t), q) \leq \frac{c \log(\min(t,\tau))}{N_{k,\tau}(t)} \right\},\\
            I_k(t) &= U_k(t),\\
            \text{where}\;\; X_{k,\tau}(t) &:= \sum_{s=t-\tau+1}^{t} X_k(s) \mathbb{1}(A(t) = k),\\
            \text{and}\;\; N_{k,\tau}(t) &:= \sum_{s=t-\tau+1}^{t} \mathbb{1}(A(t) = k).

        If rewards are in :math:`[a, b]` (default to :math:`[0, 1]`) and :math:`\mathrm{kl}(x, y)` is the Kullback-Leibler divergence between two distributions of means x and y (see :mod:`Arms.kullback`),
        and c is the parameter (default to 1).
        """
        last_pulls_of_this_arm = np.count_nonzero(self.last_choices == arm)
        if last_pulls_of_this_arm < 1:
            return float('+inf')
        else:
            mean = np.sum(self.last_rewards[self.last_choices == arm]) / last_pulls_of_this_arm
            level = constant_c * log(min(self.t, self.tau)) / last_pulls_of_this_arm
            return self.klucb(mean, level, tolerance)


class SWklUCBPlus(SWklUCB, SWUCBPlus):
    r""" An experimental policy, using only a sliding window (of :math:`\tau` *steps*, not counting draws of each arms) instead of using the full-size history, and using klUCB (see :class:`Policy.klUCB`) indexes instead of UCB.

    - Uses :math:`\tau = 4 \sqrt{T \log(T)}` if the horizon :math:`T` is given, otherwise use the default value.
    """

    def __str__(self):
        name = self.klucb.__name__[5:]
        if name == "Bern": name = ""
        if name != "": name = "({})".format(name)
        return r"SW-klUCB{}+($\tau={}$)".format(name, self.tau)
