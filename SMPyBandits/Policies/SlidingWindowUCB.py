# -*- coding: utf-8 -*-
r""" An experimental policy, using only a sliding window (of for instance :math:`\tau=1000` *steps*, not counting draws of each arms) instead of using the full-size history.

- Reference: [On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems, by A.Garivier & E.Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)

- It uses an additional :math:`\mathcal{O}(\tau)` memory but do not cost anything else in terms of time complexity (the average is done with a sliding window, and costs :math:`\mathcal{O}(1)` at every time step).

.. warning:: This is very experimental!
.. note:: This is similar to :class:`SlidingWindowRestart.SWR_UCB` but slightly different: :class:`SlidingWindowRestart.SWR_UCB` uses a window of size :math:`T_0=100` to keep in memory the last 100 *draws* of *each* arm, and restart the index if the small history mean is too far away from the whole mean, while this :class:`SWUCB` uses a fixed-size window of size :math:`\tau=1000` to keep in memory the last 1000 *steps*.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .IndexPolicy import IndexPolicy


#: Size of the sliding window.
TAU = 1000

#: Default value for the constant :math:`\alpha`.
DEFAULT_ALPHA = 0.6


# --- Manually written

class SWUCB(IndexPolicy):
    r""" An experimental policy, using only a sliding window (of for instance :math:`\tau=1000` *steps*, not counting draws of each arms) instead of using the full-size history.
    """

    def __init__(self, nbArms,
                 tau=TAU, alpha=DEFAULT_ALPHA,
                 lower=0., amplitude=1., *args, **kwargs):
        super(SWUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude, *args, **kwargs)
        # New parameters
        assert 1 <= tau, "Error: parameter 'tau' for class SWUCB has to be >= 1, but was {}.".format(tau)  # DEBUG
        self.tau = int(tau)  #: Size :math:`\tau` of the sliding window.
        assert alpha > 0, "Error: parameter 'alpha' for class SWUCB has to be > 0, but was {}.".format(alpha)  # DEBUG
        self.alpha = alpha  #: Constant :math:`\alpha` in the square-root in the computation for the index.
        # Internal memory
        self.last_rewards = np.zeros(tau)  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        self.last_choices = np.full(tau, -1)  #: Keep in memory the times where each arm was last seen.

    def __str__(self):
        return r"SW-UCB($\tau={}$, $\alpha={:.3g}$)".format(self.tau, self.alpha)

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
            return (np.sum(self.last_rewards[self.last_choices == arm]) / last_pulls_of_this_arm) + np.sqrt((self.alpha * np.log(min(self.t, self.tau))) / last_pulls_of_this_arm)


# --- Horizon dependent version

class SWUCBPlus(SWUCB):
    r""" An experimental policy, using only a sliding window (of :math:`\tau` *steps*, not counting draws of each arms) instead of using the full-size history.

    - Uses :math:`\tau = 4 \sqrt{T \log(T)}` if the horizon :math:`T` is given, otherwise use the default value.
    """

    def __init__(self, nbArms, horizon=None,
                 lower=0., amplitude=1., *args, **kwargs):
        if horizon is not None:
            T = int(horizon) if horizon is not None else None
            tau = int(4 * np.sqrt(T * np.log(T)))
        else:
            tau = TAU
        super(SWUCBPlus, self).__init__(nbArms, tau=tau, lower=lower, amplitude=amplitude, *args, **kwargs)
        # New parameter

    def __str__(self):
        return r"SW-UCB+($\tau={}$, $\alpha={:.3g}$)".format(self.tau, self.alpha)
