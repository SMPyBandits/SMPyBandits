# -*- coding: utf-8 -*-
""" The UCBoost policy for one-parameter exponential distributions.
By default, it assumes Bernoulli arms.

- Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!


from .IndexPolicy import IndexPolicy


# --- New distances

def biquadratic_distance(p, q):
    r"""The *biquadratic distance*, :math:`d_{bq}(p, q) := 2 (p - q)^2 + (4/9) * (p - q)^4`."""
    # return 2 * (p - q)**2 + (4./9) * (p - q)**4
    # XXX about 20% slower than the second less naive solution
    d = 2 * (p - q)**2
    return d + d**2 / 9.


def solution_pb_bq(p, upperbound):
    r"""Closed-form solution of the following optimisation problem, for :math:`d = d_{bq}` the :func:`biquadratic_distance` function:

    .. math::

        P_1(d)(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d(p, q) \leq \delta.

    .. math::

        q^* = \min(1, p + \sqrt{-\frac{9}{4} + \sqrt{\sqrt{81}{16} + \sqrt{9}{4} \delta}).

    - :math:`\delta` is the ``upperbound`` parameter on the semi-distance between input :math:`p` and solution :math:`q^*`.
    """
    if np.any(upperbound) < 0:
        return np.ones_like(p) * np.nan
    # XXX is it faster to precompute the constants ? yes, about 12% faster
    # q_star = np.minimum(1, p + np.sqrt(-9./4 + np.sqrt(81./16 + 9./4 * upperbound)))
    q_star = np.minimum(1, p + np.sqrt(-2.25 + np.sqrt(5.0625 + 2.25 * upperbound)))
    if not np.all(biquadratic_distance(p, q_star) <= 1.0001 * upperbound):
        print("Error: the solution to the optimisation problem P_1(d_bq), with p = {:.3g} and delta = {:.3g} was computed to be q^* = {:.3g} which seem incorrect (bq(p,q^*) = {:.3g} > {:.3g}...".format(p, upperbound, q_star, biquadratic_distance(p, q_star), upperbound))  # DEBUG
    return q_star



def hellinger_distance(p, q):
    r"""The *Hellinger distance*, :math:`d_{h}(p, q) := (\sqrt{p} - \sqrt{q})^2 + (\sqrt{1 - p} - \sqrt{1 - q})^2`."""
    return (np.sqrt(p) - np.sqrt(q))**2 + (np.sqrt(1. - p) - np.sqrt(1. - q))**2


def solution_pb_hellinger(p, upperbound):
    r"""Closed-form solution of the following optimisation problem, for :math:`d = d_{h}` the :func:`hellinger_distance` function:

    .. math::

        P_1(d_h)(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d_h(p, q) \leq \delta.

    .. math::

        q^* = \left( (1 - \frac{\delta}{2}) \sqrt{p} + \sqrt{(1 - p) (\delta - \frac{\delta^2}{4})} \right)^{2 \times \boldsymbol{1}(\delta < 2 - 2 \sqrt{p})}

    - :math:`\delta` is the ``upperbound`` parameter on the semi-distance between input :math:`p` and solution :math:`q^*`.
    """
    if np.any(upperbound < 0):
        return np.ones_like(p) * np.nan
    # XXX is it faster to precompute the constants ? yes, about 12% faster
    sqrt_p = np.sqrt(p)
    if np.any(upperbound < (2 - 2 * sqrt_p)):
        q_star = (1 - upperbound/2.) * sqrt_p + np.sqrt((1 - p) * (upperbound - upperbound**2 / 4.)) ** 2
    else:
        q_star = np.ones_like(p)
    if not np.all(hellinger_distance(p, q_star) <= 1.0001 * upperbound):
        print("Error: the solution to the optimisation problem P_1(d_h), with p = {:.3g} and delta = {:.3g} was computed to be q^* = {:.3g} which seem incorrect (h(p,q^*) = {:.3g} > {:.3g}...".format(p, upperbound, q_star, hellinger_distance(p, q_star), upperbound))  # DEBUG
    return q_star



#: Default value for the constant c used in the computation of the index
# c = 0.  #: Default value.
c = 3.  #: Default value.


class UCB_bq(IndexPolicy):
    """ The UCB(d_bq) policy for one-parameter exponential distributions.

    - It uses :func:`solution_pb_bq` as a closed-form solution to compute the UCB indexes (using the biquadratic distance).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, c=c, lower=0., amplitude=1.):
        super(UCB_bq, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.c = c  #: Parameter c

    def __str__(self):
        return r"${}$($c={:.3g}$)".format(r"\mathrm{UCB}_{bq}", self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= P_1(d_{bq})(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)})
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return solution_pb_bq(self.rewards[arm] / self.pulls[arm], (np.log(self.t) + self.c * np.log(max(1, np.log(self.t)))) / self.pulls[arm])

    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     indexes = solution_pb_bq(self.rewards / self.pulls, (np.log(self.t) + self.c * np.log(max(1, np.log(self.t)))) / self.pulls)
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index[:] = indexes


class UCB_h(IndexPolicy):
    """ The UCB(d_h) policy for one-parameter exponential distributions.

    - It uses :func:`solution_pb_hellinger` as a closed-form solution to compute the UCB indexes (using the Hellinger distance).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, c=c, lower=0., amplitude=1.):
        super(UCB_h, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.c = c  #: Parameter c

    def __str__(self):
        return r"${}$($c={:.3g}$)".format(r"\mathrm{UCB}_{h}", self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= P_1(d_{bq})(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)})
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return solution_pb_hellinger(self.rewards[arm] / self.pulls[arm], (np.log(self.t) + self.c * np.log(max(1, np.log(self.t)))) / self.pulls[arm])

    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     indexes = solution_pb_hellinger(self.rewards / self.pulls, (np.log(self.t) + self.c * np.log(max(1, np.log(self.t)))) / self.pulls)
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index[:] = indexes


class UCBoost(IndexPolicy):
    """ The UCBoost policy for one-parameter exponential distributions.
    By default, it assumes Bernoulli arms.

    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, variant='D', tolerance=1e-4, c=c, lower=0., amplitude=1.):
        super(UCBoost, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.variant_is_D = variant == 'D'  #: Whether the UCBoost(D) variant is used or not.
        self.c = c  #: Parameter c
        self.tolerance = tolerance  #: Numerical tolerance

    def __str__(self):
        if self.variant_is_D:
            return r"UCBoost(D, $c={:.3g}$)".format(self.c)
        else:
            return r"UCBoost($c={:.3g}$, $\varepsilon={:.3g}$)".format(self.c, self.tolerance)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

           \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
           I_k(t) &= XXX
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return self.klucb(self.rewards[arm] / self.pulls[arm], self.c * log(self.t) / self.pulls[arm], self.tolerance)

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = self.klucb(self.rewards / self.pulls, self.c * np.log(self.t) / self.pulls, self.tolerance)
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
