# -*- coding: utf-8 -*-
r""" The UCB-dagger (:math:`\mathrm{UCB}{\dagger}`, UCB†) policy, a significant improvement over UCB by auto-tuning the confidence level.

- Reference: [[Auto-tuning the Confidence Level for Optimistic Bandit Strategies, Lattimore, unpublished, 2017]](http://tor-lattimore.com/)
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!
_e = np.e

from .IndexPolicy import IndexPolicy

#: Default value for the parameter :math:`\alpha > 0` for UCBdagger.
ALPHA = 1


def log_bar(x):
    r"""The function defined as :math:`\mathrm{l\overline{og}}` by Lattimore:

    .. math:: \mathrm{l\overline{og}}(x) := \log\left((x+e)\sqrt{\log(x+e)}\right)

    Some values:

    >>> for x in np.logspace(0, 7, 8):
    ...     print("x = {:<5.3g} gives log_bar(x) = {:<5.3g}".format(x, log_bar(x)))
    x = 1     gives log_bar(x) = 1.45
    x = 10    gives log_bar(x) = 3.01
    x = 100   gives log_bar(x) = 5.4
    x = 1e+03 gives log_bar(x) = 7.88
    x = 1e+04 gives log_bar(x) = 10.3
    x = 1e+05 gives log_bar(x) = 12.7
    x = 1e+06 gives log_bar(x) = 15.1
    x = 1e+07 gives log_bar(x) = 17.5


    Illustration:

    >>> import matplotlib.pyplot as plt
    >>> X = np.linspace(0, 1000, 2000)
    >>> Y = log_bar(X)
    >>> plt.plot(X, Y)
    >>> plt.title(r"The $\mathrm{l\overline{og}}$ function")
    >>> plt.show()
    """
    return np.log((x + _e) * np.sqrt(np.log(x + _e)))


def Ki_function(pulls, i):
    r"""Compute the :math:`K_i(t)` index as defined in the article, for one arm i."""
    if pulls[i] <= 0:
        return len(pulls)
    else:
        return np.sum(np.minimum(1., np.sqrt(pulls / pulls[i])))


def Ki_vectorized(pulls):
    r"""Compute the :math:`K_i(t)` index as defined in the article, for all arms (in a vectorized manner).

    .. warning:: I didn't find a fast vectorized formula, so don't use this one.
    """
    return np.array([Ki_function(pulls, i) for i in range(len(pulls))])
    # ratios = np.zeros_like(pulls)
    # for i, p in enumerate(pulls):
    #     ratios[i] = np.sum(np.minimum(1., np.sqrt(pulls / p)))
    # return ratios


# --- The UCBdagger policy

class UCBdagger(IndexPolicy):
    r""" The UCB-dagger (:math:`\mathrm{UCB}{\dagger}`, UCB†) policy, a significant improvement over UCB by auto-tuning the confidence level.

    - Reference: [[Auto-tuning the Confidence Level for Optimistic Bandit Strategies, Lattimore, unpublished, 2017]](http://downloads.tor-lattimore.com/papers/XXX)
    """

    def __init__(self, nbArms, horizon=None,
                 alpha=ALPHA,
                 lower=0., amplitude=1.):
        super(UCBdagger, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert alpha > 0, "Error: parameter 'alpha' for UCBdagger should be > 0."  # DEBUG
        self.alpha = alpha  #: Parameter :math:`\alpha > 0`.
        assert horizon > 0, "Error: parameter 'horizon' for UCBdagger should be > 0."  # DEBUG
        self.horizon = int(horizon)  #: Parameter :math:`T > 0`.
        # # Internal memory
        # self.indexes_K = nbArms * np.ones(nbArms)  #: Keep in memory the indexes :math:`K_i(t)`, to speed up their computation from :math:`\mathcal{O}(K^2)` to :math:`\mathcal{O}(K)` (where :math:`K` is the number of arm).

    def __str__(self):
        if self.alpha == ALPHA:
            return r"UCB$\dagger$($T={}$)".format(self.horizon)
        else:
            return r"UCB$\dagger$($T={}$, $\alpha={:.3g}$)".format(self.horizon, self.alpha)

    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
        # Do this in less than O(K) times?
        # Don't see the point, maintaining a list of arm sorted by their N_k(t) actually takes more time than just looping

        # # XXX old fashion
        # # for other_arm in range(self.nbArms):
        # #     if self.pulls[other_arm] > self.pulls[arm]:

        # # Cf. Appendix D. Computation of UCB†
        # for other_arm in np.where(self.pulls > self.pulls[arm])[0]:
        #     self.indexes_K[other_arm] += np.sqrt((self.pulls[arm] + 1) / self.pulls[other_arm]) - np.sqrt(self.pulls[arm] / self.pulls[other_arm])

        # # Just debugging
        # assert np.all(self.indexes_K >= 0), "Error: an index K was found < 0..."  # DEBUG
        # _K = Ki_vectorized(self.pulls)
        # for other_arm in range(self.nbArms):
        #     # assert self.indexes_K[other_arm] == _K[other_arm], "Error: an index K = {:.3g} was found wrong for arm {} (the reference value was {:.3g})...".format(self.indexes_K[other_arm], other_arm, _K[other_arm])  # DEBUG
        #     if not (self.indexes_K[other_arm] == _K[other_arm]):
        #         print("Error: an index K = {:.3g} was found wrong for arm {} (the reference value was {:.3g})...".format(self.indexes_K[other_arm], other_arm, _K[other_arm]))  # DEBUG

        # Now actually store the reward and pull
        super(UCBdagger, self).getReward(arm, reward)

    # --- Computation

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            I_k(t) &= \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{2 \alpha}{N_k(t)} \mathrm{l}\overline{\mathrm{og}}\left( \frac{T}{H_k(t)} \right)}, \\
            \text{where}\;\; & H_k(t) := N_k(t) K_k(t) \\
            \text{and}\;\; & K_k(t) := \sum_{j=1}^{K} \min(1, \sqrt{\frac{T_j(t)}{T_i(t)}}).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX Use the internal indexes (faster?) if they work
            # H_arm = self.pulls[arm] * self.indexes_K[arm]
            H_arm = self.pulls[arm] * Ki_function(self.pulls, arm)
            return (self.rewards[arm] / self.pulls[arm]) + np.sqrt(((2. * self.alpha) / self.pulls[arm]) * log_bar(self.horizon / H_arm))

    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     # H_arms = self.pulls * Ki_vectorized(self.pulls)
    #     H_arms = self.pulls * self.indexes_K
    #     indexes (self.rewards / self.pulls) + np.sqrt(((2. * self.alpha) / self.pulls) * log_bar(self.horizon / H_arms))
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index[:] = indexes
