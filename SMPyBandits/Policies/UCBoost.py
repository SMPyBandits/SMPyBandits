# -*- coding: utf-8 -*-
""" The UCBoost policy for one-parameter exponential distributions.

- Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).

.. warning:: The whole goal of their paper is to provide a numerically efficient alternative to kl-UCB, so for my comparison to be fair, I should either use the Python versions of klUCB utility functions (using :mod:`kullback`) or write C or Cython versions of this UCBoost module.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!


from .IndexPolicy import IndexPolicy

try:
    from .usenumba import jit  # Import numba.jit or a dummy jit(f)=f
except (ValueError, ImportError, SystemError):
    from usenumba import jit  # Import numba.jit or a dummy jit(f)=f


#: Default value for the constant c used in the computation of the index
# c = 0.  #: Default value.
c = 3.  #: Default value.


# --- New distance and algorithm: biquadratic

@jit
def biquadratic_distance(p, q):
    r"""The *biquadratic distance*, :math:`d_{bq}(p, q) := 2 (p - q)^2 + (4/9) * (p - q)^4`."""
    # return 2 * (p - q)**2 + (4./9) * (p - q)**4
    # XXX about 20% slower than the second less naive solution
    d = 2 * (p - q)**2
    return d + d**2 / 9.


@jit
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
            I_k(t) &= P_1(d_{bq})(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}).
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


# --- New distance and algorithm: Hellinger


@jit
def hellinger_distance(p, q):
    r"""The *Hellinger distance*, :math:`d_{h}(p, q) := (\sqrt{p} - \sqrt{q})^2 + (\sqrt{1 - p} - \sqrt{1 - q})^2`."""
    return (np.sqrt(p) - np.sqrt(q))**2 + (np.sqrt(1. - p) - np.sqrt(1. - q))**2


@jit
def solution_pb_hellinger(p, upperbound):
    r"""Closed-form solution of the following optimisation problem, for :math:`d = d_{h}` the :func:`hellinger_distance` function:

    .. math::

        P_1(d_h)(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d_h(p, q) \leq \delta.

    .. math::

        q^* = \left( (1 - \frac{\delta}{2}) \sqrt{p} + \sqrt{(1 - p) (\delta - \frac{\delta^2}{4})} \right)^{2 \times \boldsymbol{1}(\delta < 2 - 2 \sqrt{p})}.

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
            I_k(t) &= P_1(d_h)(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}).
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


# --- New distance and algorithm: lower-bound on the Kullback-Leibler distance


eps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]


@jit
def kullback_leibler_distance(x, y):
    r""" Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution#Kullback.E2.80.93Leibler_divergence

    .. math:: kl(x, y) = \mathrm{KL}(\mathcal{B}(x), \mathcal{B}(y)) = x \log(\frac{x}{y}) + (1-x) \log(\frac{1-x}{1-y}).
    """
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


@jit
def kullback_leibler_distance_lowerbound(x, y):
    r""" Lower-bound on the Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution#Kullback.E2.80.93Leibler_divergence

    .. math:: d_{lb}(x, y) = x \log(\frac{x}{y}) + (1-x) \log(\frac{1-x}{1-y}).
    """
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log(x) + (1 - x) * np.log((1 - x) / (1 - y))


@jit
def solution_pb_kllb(p, upperbound):
    r"""Closed-form solution of the following optimisation problem, for :math:`d = d_{lb}` the proposed lower-bound on the Kullback-Leibler binary distance (:func:`kullback_leibler_distance_lowerbound`) function:

    .. math::

        P_1(d_{lb})(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d_{lb}(p, q) \leq \delta.

    .. math::

        q^* = XXX.

    - :math:`\delta` is the ``upperbound`` parameter on the semi-distance between input :math:`p` and solution :math:`q^*`.
    """
    if np.any(upperbound < 0):
        return np.ones_like(p) * np.nan
    one_m_p = 1 - p
    q_star = 1 - one_m_p * np.exp( (p * np.log(p) - upperbound) / one_m_p )
    if not np.all(kullback_leibler_distance_lowerbound(p, q_star) <= 1.0001 * upperbound):
        print("Error: the solution to the optimisation problem P_1(d_lb), with p = {:.3g} and delta = {:.3g} was computed to be q^* = {:.3g} which seem incorrect (h(p,q^*) = {:.3g} > {:.3g}...".format(p, upperbound, q_star, kullback_leibler_distance_lowerbound(p, q_star), upperbound))  # DEBUG
    return q_star


class UCB_lb(IndexPolicy):
    """ The UCB(d_lb) policy for one-parameter exponential distributions.

    - It uses :func:`solution_pb_kllb` as a closed-form solution to compute the UCB indexes (using the lower-bound on the Kullback-Leibler distance).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, c=c, lower=0., amplitude=1.):
        super(UCB_lb, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.c = c  #: Parameter c

    def __str__(self):
        return r"${}$($c={:.3g}$)".format(r"\mathrm{UCB}_{lb}", self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= P_1(d_lb)(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return solution_pb_kllb(self.rewards[arm] / self.pulls[arm], (np.log(self.t) + self.c * np.log(max(1, np.log(self.t)))) / self.pulls[arm])

    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     indexes = solution_pb_kllb(self.rewards / self.pulls, (np.log(self.t) + self.c * np.log(max(1, np.log(self.t)))) / self.pulls)
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index[:] = indexes


# --- New distance and algorithm: a shifted tangent line function of d_kl


@jit
def distance_t(x, y):
    r""" A shifted tangent line function of :func:`kullback_leibler_distance`.

    .. math:: d_t(x, y) = x \log(\frac{x}{y}) + (1-x) \log(\frac{1-x}{1-y}).
    """
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log(x) + (1 - x) * np.log((1 - x) / (1 - y))


@jit
def solution_pb_t(p, upperbound):
    r"""Closed-form solution of the following optimisation problem, for :math:`d = d_t` a shifted tangent line function of :func:`kullback_leibler_distance` (:func:`distance_t`) function:

    .. math::

        P_1(d_t)(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d_t(p, q) \leq \delta.

    .. math::

        q^* = XXX.

    - :math:`\delta` is the ``upperbound`` parameter on the semi-distance between input :math:`p` and solution :math:`q^*`.
    """
    if np.any(upperbound < 0):
        return np.ones_like(p) * np.nan
    q_star = XXX
    np.minimum(1, (p + 1) / 2. * (upperbound - p * np.log(p / (p + 1)) - np.log(2 / (np.e * (1 + p)))))
    if not np.all(distance_t(p, q_star) <= 1.0001 * upperbound):
        print("Error: the solution to the optimisation problem P_1(d_t), with p = {:.3g} and delta = {:.3g} was computed to be q^* = {:.3g} which seem incorrect (h(p,q^*) = {:.3g} > {:.3g}...".format(p, upperbound, q_star, distance_t(p, q_star), upperbound))  # DEBUG
    return q_star


class UCB_t(IndexPolicy):
    """ The UCB(d_t) policy for one-parameter exponential distributions.

    - It uses :func:`solution_pb_t` as a closed-form solution to compute the UCB indexes (using the a shifted tangent line function of :func:`kullback_leibler_distance`).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, c=c, lower=0., amplitude=1.):
        super(UCB_t, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.c = c  #: Parameter c

    def __str__(self):
        return r"${}$($c={:.3g}$)".format(r"\mathrm{UCB}_{t}", self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= P_1(d_t)(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return solution_pb_t(self.rewards[arm] / self.pulls[arm], (np.log(self.t) + self.c * np.log(max(1, np.log(self.t)))) / self.pulls[arm])

    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     indexes = solution_pb_t(self.rewards / self.pulls, (np.log(self.t) + self.c * np.log(max(1, np.log(self.t)))) / self.pulls)
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index[:] = indexes


# --- Now the generic UCBoost algorithm

try:
    from numbers import Number
    def is_a_true_number(n)
        return isinstance(n, Number)
except ImportError:
    def is_a_true_number(n):
        try:
            float(n)
            return True
        except:
            return False


# This is a hack, so that we can store a list of functions in the UCBoost algorithm,
# without actually storing functions (which are unhashable).
_private_mapping_of_d_distances = {
    'solution_pb_bq': solution_pb_bq,
    'solution_pb_hellinger': solution_pb_hellinger,
    'solution_pb_kllb': solution_pb_kllb,
    'solution_pb_t': solution_pb_t,
}


class UCBoost(IndexPolicy):
    """ The UCBoost policy for one-parameter exponential distributions.

    - It is quite simple: using a set of kl-dominated and candidate semi-distances D, the UCB index for each arm (at each step) is computed as the *smallest* upper confidence bound given (for this arm at this time t) for each distance d.
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, set_D=None, c=c, lower=0., amplitude=1.):
        super(UCBoost, self).__init__(nbArms, lower=lower, amplitude=amplitude)

        # FIXED having a set of functions as attribute will make this object unhashable! that's bad for pickling and parallelization!
        # DONE One solution is to store keys, and look up the functions in a fixed (hidden) dictionary
        if set_D is None:
            set_D = ["solution_pb_bq", "solution_pb_hellinger", "solution_pb_kllb", "solution_pb_t"]
        elif is_a_true_number(set_D):
            assert set_D in {3, 4}, "Error: if set_D is an integer, it should be 3 or 4."
            if set_D == 3:
                set_D = ["solution_pb_bq", "solution_pb_hellinger", "solution_pb_kllb"]
            elif set_D == 4:
                set_D = ["solution_pb_bq", "solution_pb_hellinger", "solution_pb_kllb", "solution_pb_t"]

        self.set_D = set_D
        self.c = c  #: Parameter c

    def __str__(self):
        return r"UCBoost(|D|={}, $c={:.3g}$)".format(len(self.set_D), self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= \min_{d\in D} P_1(d)(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # FIXME do this more efficiently!
            solutions_from_each_d = []
            for key in self.set_D:
                solution_pb = _private_mapping_of_d_distances[key]
                solutions_from_each_d.append(
                    solution_pb(self.rewards[arm] / self.pulls[arm], (np.log(self.t) + self.c * np.log(max(1, np.log(self.t)))) / self.pulls[arm])
                )
            return np.min(solutions_from_each_d)


    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     indexes = self.klucb(self.rewards / self.pulls, self.c * np.log(self.t) / self.pulls, self.tolerance)
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index[:] = indexes
