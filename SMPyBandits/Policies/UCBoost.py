# -*- coding: utf-8 -*-
""" The UCBoost policy for bounded bandits (on [0, 1]).

- Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).

.. warning:: The whole goal of their paper is to provide a numerically efficient alternative to kl-UCB, so for my comparison to be fair, I should either use the Python versions of klUCB utility functions (using :mod:`kullback`) or write C or Cython versions of this UCBoost module. My conclusion is that kl-UCB is *always* faster than UCBoost.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from math import log, sqrt, exp, ceil, floor

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!


try:
    from .IndexPolicy import IndexPolicy
except ImportError:
    from IndexPolicy import IndexPolicy

try:
    from .usenumba import jit  # Import numba.jit or a dummy jit(f)=f
except (ValueError, ImportError, SystemError):
    from usenumba import jit  # Import numba.jit or a dummy jit(f)=f


#: Default value for the constant c used in the computation of the index
c = 3.  #: Default value for the theorems to hold.
c = 0.  #: Default value for better practical performance.


#: Tolerance when checking (with ``assert``) that the solution(s) of any convex problem are correct.
tolerance_with_upperbound = 1.0001


#: Whether to check that the solution(s) of any convex problem are correct.
#:
#: .. warning:: This is currently disabled, to try to optimize this module! WARNING bring it back when debugging!
CHECK_SOLUTION = True
CHECK_SOLUTION = False  # XXX Faster!

# --- New distance and algorithm: quadratic

# @jit
def squadratic_distance(p, q):
    r""" The *quadratic distance*, :math:`d_{sq}(p, q) := 2 (p - q)^2`."""
    # p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    # q = min(max(q, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    return 2 * (p - q)**2


# @jit
def solution_pb_sq(p, upperbound, check_solution=CHECK_SOLUTION):
    r""" Closed-form solution of the following optimisation problem, for :math:`d = d_{sq}` the :func:`biquadratic_distance` function:

    .. math::

        P_1(d_{sq})(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d_{sq}(p, q) \leq \delta.

    - The solution is:

    .. math::

        q^* = p + \sqrt{\frac{\delta}{2}}.

    - :math:`\delta` is the ``upperbound`` parameter on the semi-distance between input :math:`p` and solution :math:`q^*`.
    """
    return p + sqrt(upperbound / 2.)

    # XXX useless checking of the solution, takes time
    # if check_solution and not np.all(squadratic_distance(p, q_star) <= tolerance_with_upperbound * upperbound):
    #     print("Error: the solution to the optimisation problem P_1(d_sq), with p = {:.3g} and delta = {:.3g} was computed to be q^* = {:.3g} which seem incorrect (sq(p,q^*) = {:.3g} > {:.3g})...".format(p, upperbound, q_star, squadratic_distance(p, q_star), upperbound))  # DEBUG
    # return q_star


class UCB_sq(IndexPolicy):
    """ The UCB(d_sq) policy for bounded bandits (on [0, 1]).

    - It uses :func:`solution_pb_sq` as a closed-form solution to compute the UCB indexes (using the quadratic distance).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, c=c, lower=0., amplitude=1.):
        super(UCB_sq, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert c >= 0, "Error: parameter c should be > 0 strictly, but = {:.3g} is not!".format(c)  # DEBUG
        self.c = c  #: Parameter c

    def __str__(self):
        return r"${}$($c={:.3g}$)".format(r"UCB_{d=d_{sq}}", self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= P_1(d_{sq})\left(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}\right).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        if self.c == 0:
            return solution_pb_sq(self.rewards[arm] / self.pulls[arm], log(self.t) / self.pulls[arm])  # XXX Faster if c=0
        return solution_pb_sq(self.rewards[arm] / self.pulls[arm], (log(self.t) + self.c * log(max(1, log(self.t)))) / self.pulls[arm])

    # TODO make this vectorized function working!
    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     indexes = solution_pb_bq(self.rewards / self.pulls, (log(self.t) + self.c * log(max(1, log(self.t)))) / self.pulls)
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index[:] = indexes


# --- New distance and algorithm: biquadratic

# @jit
def biquadratic_distance(p, q):
    r""" The *biquadratic distance*, :math:`d_{bq}(p, q) := 2 (p - q)^2 + (4/9) * (p - q)^4`."""
    # return 2 * (p - q)**2 + (4./9) * (p - q)**4
    # XXX about 20% slower than the second less naive solution
    # p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    # q = min(max(q, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    d = 2 * (p - q)**2
    return d + d**2 / 9.


# @jit
def solution_pb_bq(p, upperbound, check_solution=CHECK_SOLUTION):
    r""" Closed-form solution of the following optimisation problem, for :math:`d = d_{bq}` the :func:`biquadratic_distance` function:

    .. math::

        P_1(d_{bq})(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d_{bq}(p, q) \leq \delta.

    - The solution is:

    .. math::

        q^* = \min(1, p + \sqrt{-\frac{9}{4} + \sqrt{\frac{81}{16} + \frac{9}{4} \delta}}).

    - :math:`\delta` is the ``upperbound`` parameter on the semi-distance between input :math:`p` and solution :math:`q^*`.
    """
    # p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    # DONE is it faster to precompute the constants ? yes, about 12% faster
    return min(1, p + sqrt(-2.25 + sqrt(5.0625 + 2.25 * upperbound)))

    # XXX useless checking of the solution, takes time
    # if check_solution and not np.all(biquadratic_distance(p, q_star) <= tolerance_with_upperbound * upperbound):
    #     print("Error: the solution to the optimisation problem P_1(d_bq), with p = {:.3g} and delta = {:.3g} was computed to be q^* = {:.3g} which seem incorrect (bq(p,q^*) = {:.3g} > {:.3g})...".format(p, upperbound, q_star, biquadratic_distance(p, q_star), upperbound))  # DEBUG
    # return q_star


class UCB_bq(IndexPolicy):
    """ The UCB(d_bq) policy for bounded bandits (on [0, 1]).

    - It uses :func:`solution_pb_bq` as a closed-form solution to compute the UCB indexes (using the biquadratic distance).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, c=c, lower=0., amplitude=1.):
        super(UCB_bq, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert c >= 0, "Error: parameter c should be > 0 strictly, but = {:.3g} is not!".format(c)  # DEBUG
        self.c = c  #: Parameter c

    def __str__(self):
        return r"${}$($c={:.3g}$)".format(r"\mathrm{UCB}_{d=d_{bq}}", self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= P_1(d_{bq})\left(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}\right).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        if self.c == 0:
            return solution_pb_bq(self.rewards[arm] / self.pulls[arm], log(self.t) / self.pulls[arm])  # XXX Faster if c=0
        return solution_pb_bq(self.rewards[arm] / self.pulls[arm], (log(self.t) + self.c * log(max(1, log(self.t)))) / self.pulls[arm])


# --- New distance and algorithm: Hellinger


# @jit
def hellinger_distance(p, q):
    r""" The *Hellinger distance*, :math:`d_{h}(p, q) := (\sqrt{p} - \sqrt{q})^2 + (\sqrt{1 - p} - \sqrt{1 - q})^2`."""
    # p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    # q = min(max(q, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    return (sqrt(p) - sqrt(q))**2 + (sqrt(1. - p) - sqrt(1. - q))**2


# @jit
def solution_pb_hellinger(p, upperbound, check_solution=CHECK_SOLUTION):
    r""" Closed-form solution of the following optimisation problem, for :math:`d = d_{h}` the :func:`hellinger_distance` function:

    .. math::

        P_1(d_h)(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d_h(p, q) \leq \delta.

    - The solution is:

    .. math::

        q^* = \left( (1 - \frac{\delta}{2}) \sqrt{p} + \sqrt{(1 - p) (\delta - \frac{\delta^2}{4})} \right)^{2 \times \boldsymbol{1}(\delta < 2 - 2 \sqrt{p})}.

    - :math:`\delta` is the ``upperbound`` parameter on the semi-distance between input :math:`p` and solution :math:`q^*`.
    """
    # DONE is it faster to precompute the constants ? yes, about 12% faster
    # p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    sqrt_p = sqrt(p)
    if upperbound < (2 - 2 * sqrt_p):
        return (1 - upperbound/2.) * sqrt_p + sqrt((1 - p) * (upperbound - upperbound**2 / 4.)) ** 2
    else:
        return p

    # XXX useless checking of the solution, takes time
    # if check_solution and not np.all(hellinger_distance(p, q_star) <= tolerance_with_upperbound * upperbound):
    #     print("Error: the solution to the optimisation problem P_1(d_h), with p = {:.3g} and delta = {:.3g} was computed to be q^* = {:.3g} which seem incorrect (h(p,q^*) = {:.3g} > {:.3g})...".format(p, upperbound, q_star, hellinger_distance(p, q_star), upperbound))  # DEBUG
    # return q_star


class UCB_h(IndexPolicy):
    """ The UCB(d_h) policy for bounded bandits (on [0, 1]).

    - It uses :func:`solution_pb_hellinger` as a closed-form solution to compute the UCB indexes (using the Hellinger distance).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, c=c, lower=0., amplitude=1.):
        super(UCB_h, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert c >= 0, "Error: parameter c should be > 0 strictly, but = {:.3g} is not!".format(c)  # DEBUG
        self.c = c  #: Parameter c

    def __str__(self):
        return r"${}$($c={:.3g}$)".format(r"\mathrm{UCB}_{d=d_h}", self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= P_1(d_h)\left(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}\right).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        if self.c == 0:
            return solution_pb_hellinger(self.rewards[arm] / self.pulls[arm], log(self.t) / self.pulls[arm])  # XXX Faster if c=0
        return solution_pb_hellinger(self.rewards[arm] / self.pulls[arm], (log(self.t) + self.c * log(max(1, log(self.t)))) / self.pulls[arm])


# --- New distance and algorithm: lower-bound on the Kullback-Leibler distance


eps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]


# @jit
def kullback_leibler_distance_on_mean(p, q):
    r""" Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution#Kullback.E2.80.93Leibler_divergence

    .. math:: \mathrm{kl}(p, q) = \mathrm{KL}(\mathcal{B}(p), \mathcal{B}(q)) = p \log\left(\frac{p}{q}\right) + (1-p) \log\left(\frac{1-p}{1-q}\right).
    """
    p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    q = min(max(q, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))


# @jit
def kullback_leibler_distance_lowerbound(p, q):
    r""" Lower-bound on the Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution#Kullback.E2.80.93Leibler_divergence

    .. math:: d_{lb}(p, q) = p \log\left( p \right) + (1-p) \log\left(\frac{1-p}{1-q}\right).
    """
    p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    q = min(max(q, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    return p * log(p) + (1 - p) * log((1 - p) / (1 - q))


# @jit
def solution_pb_kllb(p, upperbound, check_solution=CHECK_SOLUTION):
    r""" Closed-form solution of the following optimisation problem, for :math:`d = d_{lb}` the proposed lower-bound on the Kullback-Leibler binary distance (:func:`kullback_leibler_distance_lowerbound`) function:

    .. math::

        P_1(d_{lb})(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d_{lb}(p, q) \leq \delta.

    - The solution is:

    .. math::

        q^* = 1 - (1 - p) \exp\left(\frac{p \log(p) - \delta}{1 - p}\right).

    - :math:`\delta` is the ``upperbound`` parameter on the semi-distance between input :math:`p` and solution :math:`q^*`.
    """
    p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    return 1 - (1 - p) * exp((p * log(p) - upperbound) / (1 - p))

    # XXX useless checking of the solution, takes time
    # if check_solution and not np.all(kullback_leibler_distance_lowerbound(p, q_star) <= tolerance_with_upperbound * upperbound):
    #     print("Error: the solution to the optimisation problem P_1(d_lb), with p = {:.3g} and delta = {:.3g} was computed to be q^* = {:.3g} which seem incorrect (h(p,q^*) = {:.3g} > {:.3g})...".format(p, upperbound, q_star, kullback_leibler_distance_lowerbound(p, q_star), upperbound))  # DEBUG
    # return q_star


class UCB_lb(IndexPolicy):
    """ The UCB(d_lb) policy for bounded bandits (on [0, 1]).

    - It uses :func:`solution_pb_kllb` as a closed-form solution to compute the UCB indexes (using the lower-bound on the Kullback-Leibler distance).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, c=c, lower=0., amplitude=1.):
        super(UCB_lb, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert c >= 0, "Error: parameter c should be > 0 strictly, but = {:.3g} is not!".format(c)  # DEBUG
        self.c = c  #: Parameter c

    def __str__(self):
        return r"${}$($c={:.3g}$)".format(r"\mathrm{UCB}_{d=d_{lb}}", self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= P_1(d_{lb})\left(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}\right).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        if self.c == 0:
            return solution_pb_kllb(self.rewards[arm] / self.pulls[arm], log(self.t) / self.pulls[arm])  # XXX Faster if c=0
        return solution_pb_kllb(self.rewards[arm] / self.pulls[arm], (log(self.t) + self.c * log(max(1, log(self.t)))) / self.pulls[arm])


# --- New distance and algorithm: a shifted tangent line function of d_kl


# @jit
def distance_t(p, q):
    r""" A shifted tangent line function of :func:`kullback_leibler_distance_on_mean`.

    .. math:: d_t(p, q) = \frac{2 q}{p + 1} + p \log\left(\frac{p}{p + 1}\right) + \log\left(\frac{2}{\mathrm{e}(p + 1)}\right).

    .. warning:: I think there might be a typo in the formula in the article, as this :math:`d_t` does not seem to "depend enough on q" *(just intuition)*.
    """
    p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    # q = min(max(q, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    return (2 * q) / (p + 1) + p * log(p / (p + 1)) + log(2 / (p + 1)) - 1.
    # # XXX second computation about 12% faster...
    # p_plus_one = p + 1
    # return (2 * q) / p_plus_one + p * log(p) - (p_plus_one) * log(p_plus_one) - 0.306853



# @jit
def solution_pb_t(p, upperbound, check_solution=CHECK_SOLUTION):
    r""" Closed-form solution of the following optimisation problem, for :math:`d = d_t` a shifted tangent line function of :func:`kullback_leibler_distance_on_mean` (:func:`distance_t`) function:

    .. math::

        P_1(d_t)(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d_t(p, q) \leq \delta.

    - The solution is:

    .. math::

        q^* = \min\left(1, \frac{p + 1}{2} \left( \delta - p \log\left(\frac{p}{p + 1}\right) - \log\left(\frac{2}{\mathrm{e} (p + 1)}\right) \right)\right).

    - :math:`\delta` is the ``upperbound`` parameter on the semi-distance between input :math:`p` and solution :math:`q^*`.
    """
    # p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    return min(1, ((p + 1) / 2.) * (upperbound - p * log(p / (p + 1)) - log(2 / (p + 1)) + 1))

    # XXX useless checking of the solution, takes time
    # if check_solution and not np.all(distance_t(p, q_star) <= tolerance_with_upperbound * upperbound):
    #     print("Error: the solution to the optimisation problem P_1(d_t), with p = {:.3g} and delta = {:.3g} was computed to be q^* = {:.3g} which seem incorrect (h(p,q^*) = {:.3g} > {:.3g})...".format(p, upperbound, q_star, distance_t(p, q_star), upperbound))  # DEBUG
    # return q_star


class UCB_t(IndexPolicy):
    """ The UCB(d_t) policy for bounded bandits (on [0, 1]).

    - It uses :func:`solution_pb_t` as a closed-form solution to compute the UCB indexes (using a shifted tangent line function of :func:`kullback_leibler_distance_on_mean`).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).

    .. warning:: It has bad performance, as expected (see the paper for their remark).
    """

    def __init__(self, nbArms, c=c, lower=0., amplitude=1.):
        super(UCB_t, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert c >= 0, "Error: parameter c should be > 0 strictly, but = {:.3g} is not!".format(c)  # DEBUG
        self.c = c  #: Parameter c

    def __str__(self):
        return r"${}$($c={:.3g}$)".format(r"\mathrm{UCB}_{d=d_t}", self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= P_1(d_t)\left(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}\right).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        if self.c == 0:
            return solution_pb_t(self.rewards[arm] / self.pulls[arm], log(self.t) / self.pulls[arm])  # XXX Faster if c=0
        return solution_pb_t(self.rewards[arm] / self.pulls[arm], (log(self.t) + self.c * log(max(1, log(self.t)))) / self.pulls[arm])


# --- Now the generic UCBoost algorithm

try:
    from numbers import Number
    def is_a_true_number(n):
        """ Check if n is a number or not (``int``, ``float``, ``complex`` etc, any instance of :py:class:`numbers.Number` class."""
        return isinstance(n, Number)
except ImportError:
    def is_a_true_number(n):
        """ Check if n is a number or not (``int``, ``float``, ``complex`` etc, any instance of :py:class:`numbers.Number` class."""
        try:
            float(n)
            return True
        except:
            return False


# This is a hack, so that we can store a list of functions in the UCBoost algorithm,
# without actually storing functions (which are unhashable).
_distance_of_key = {
    'solution_pb_sq': solution_pb_sq,
    'solution_pb_bq': solution_pb_bq,
    'solution_pb_hellinger': solution_pb_hellinger,
    'solution_pb_kllb': solution_pb_kllb,
    'solution_pb_t': solution_pb_t,
}


class UCBoost(IndexPolicy):
    """ The UCBoost policy for bounded bandits (on [0, 1]).

    - It is quite simple: using a set of kl-dominated and candidate semi-distances D, the UCB index for each arm (at each step) is computed as the *smallest* upper confidence bound given (for this arm at this time t) for each distance d.
    - ``set_D`` should be either a set of *strings* (and NOT functions), or a number (3, 4 or 5). 3 indicate using ``d_bq``, ``d_h``, ``d_lb``, 4 adds ``d_t``, and 5 adds ``d_sq`` (see the article, Corollary 3, p5, for more details).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, set_D=None, c=c, lower=0., amplitude=1.):
        super(UCBoost, self).__init__(nbArms, lower=lower, amplitude=amplitude)

        # FIXED having a set of functions as attribute will make this object unhashable! that's bad for pickling and parallelization!
        # DONE One solution is to store keys, and look up the functions in a fixed (hidden) dictionary
        if set_D is None:
            set_D = 4
        if is_a_true_number(set_D):
            assert set_D in {3, 4, 5}, "Error: if set_D is an integer, it should be 3 or 4 or 5."
            if set_D == 3:
                set_D = ["solution_pb_bq", "solution_pb_hellinger", "solution_pb_kllb"]
            elif set_D == 4:
                set_D = ["solution_pb_bq", "solution_pb_hellinger", "solution_pb_kllb", "solution_pb_t"]
            elif set_D == 5:
                set_D = ["solution_pb_sq", "solution_pb_bq", "solution_pb_hellinger", "solution_pb_kllb", "solution_pb_t"]
        assert all(key in _distance_of_key for key in set_D), "Error: one key in set_D = {} was found to not correspond to a distance (list of possible keys are {}).".format(set_D, list(_distance_of_key.keys()))  # DEBUG

        self.set_D = set_D  #: Set of *strings* that indicate which d functions are in the set of functions D. Warning: do not use real functions here, or the object won't be hashable!
        assert c >= 0, "Error: parameter c should be > 0 strictly, but = {:.3g} is not!".format(c)  # DEBUG
        self.c = c  #: Parameter c

    def __str__(self):
        return r"UCBoost($|D|={}$, $c={:.3g}$)".format(len(self.set_D), self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= \min_{d\in D} P_1(d)\left(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}\right).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        p = self.rewards[arm] / self.pulls[arm]
        # upperbound = log(self.t) / self.pulls[arm]
        upperbound = (log(self.t) + self.c * log(max(1, log(self.t)))) / self.pulls[arm]
        return min(
            _distance_of_key[key](p, upperbound)
            for key in self.set_D
        )


_bq_h_lb = [solution_pb_bq, solution_pb_hellinger, solution_pb_kllb]

class UCBoost_bq_h_lb(UCBoost):
    """ The UCBoost policy for bounded bandits (on [0, 1]).

    - It is quite simple: using a set of kl-dominated and candidate semi-distances D, the UCB index for each arm (at each step) is computed as the *smallest* upper confidence bound given (for this arm at this time t) for each distance d.
    - ``set_D`` is ``d_bq``, ``d_h``, ``d_lb`` (see the article, Corollary 3, p5, for more details).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, c=c, lower=0., amplitude=1.):
        super(UCBoost_bq_h_lb, self).__init__(nbArms, set_D=3, c=c, lower=lower, amplitude=amplitude)

    def __str__(self):
        return r"UCBoost($D={}$, $c={:.3g}$)".format("\{d_{bq},d_h,d_{lb}\}", self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= \min_{d\in D} P_1(d)\left(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}\right).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        p = self.rewards[arm] / self.pulls[arm]
        # upperbound = log(self.t) / self.pulls[arm]
        upperbound = (log(self.t) + self.c * log(max(1, log(self.t)))) / self.pulls[arm]
        return min(
            solution_pb(p, upperbound)
            for solution_pb in _bq_h_lb
        )


_bq_h_lb_t = [solution_pb_bq, solution_pb_hellinger, solution_pb_kllb, solution_pb_t]

class UCBoost_bq_h_lb_t(UCBoost):
    """ The UCBoost policy for bounded bandits (on [0, 1]).

    - It is quite simple: using a set of kl-dominated and candidate semi-distances D, the UCB index for each arm (at each step) is computed as the *smallest* upper confidence bound given (for this arm at this time t) for each distance d.
    - ``set_D`` is ``d_bq``, ``d_h``, ``d_lb``, ``d_t`` (see the article, Corollary 3, p5, for more details).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, c=c, lower=0., amplitude=1.):
        super(UCBoost_bq_h_lb_t, self).__init__(nbArms, set_D=4, c=c, lower=lower, amplitude=amplitude)

    def __str__(self):
        return r"UCBoost($D={}$, $c={:.3g}$)".format("\{d_{bq},d_h,d_{lb},d_t\}", self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= \min_{d\in D} P_1(d)\left(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}\right).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        p = self.rewards[arm] / self.pulls[arm]
        # upperbound = log(self.t) / self.pulls[arm]
        upperbound = (log(self.t) + self.c * log(max(1, log(self.t)))) / self.pulls[arm]
        return min(
            solution_pb(p, upperbound)
            for solution_pb in _bq_h_lb_t
        )


_bq_h_lb_t_sq = [solution_pb_bq, solution_pb_hellinger, solution_pb_kllb, solution_pb_t, solution_pb_sq]

class UCBoost_bq_h_lb_t_sq(UCBoost):
    """ The UCBoost policy for bounded bandits (on [0, 1]).

    - It is quite simple: using a set of kl-dominated and candidate semi-distances D, the UCB index for each arm (at each step) is computed as the *smallest* upper confidence bound given (for this arm at this time t) for each distance d.
    - ``set_D`` is ``d_bq``, ``d_h``, ``d_lb``, ``d_t``, ``d_sq`` (see the article, Corollary 3, p5, for more details).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, c=c, lower=0., amplitude=1.):
        super(UCBoost_bq_h_lb_t_sq, self).__init__(nbArms, set_D=5, c=c, lower=lower, amplitude=amplitude)

    def __str__(self):
        return r"UCBoost($D={}$, $c={:.3g}$)".format("\{d_{bq},d_h,d_{lb},d_t,d_{sq}\}", self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= \min_{d\in D} P_1(d)\left(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}\right).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        p = self.rewards[arm] / self.pulls[arm]
        # upperbound = log(self.t) / self.pulls[arm]
        upperbound = (log(self.t) + self.c * log(max(1, log(self.t)))) / self.pulls[arm]
        return min(
            solution_pb(p, upperbound)
            for solution_pb in _bq_h_lb_t_sq
        )


# --- New distance and algorithm: epsilon approximation on the Kullback-Leibler distance

# @jit
def min_solutions_pb_from_epsilon(p, upperbound, epsilon=0.001, check_solution=CHECK_SOLUTION):
    r""" List of closed-form solutions of the following optimisation problems, for :math:`d = d_s^k` approximation of :math:`d_{kl}` and any :math:`\tau_1(p) \leq k \leq \tau_2(p)`:

    .. math::

        P_1(d_s^k)(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d_s^k(p, q) \leq \delta.

    - The solution is:

    .. math::

        q^* &= q_k^{\boldsymbol{1}(\delta < d_{kl}(p, q_k))},\\
        d_s^k &: (p, q) \mapsto d_{kl}(p, q_k) \boldsymbol{1}(q > q_k),\\
        q_k &:= 1 - \left( 1 - \frac{\varepsilon}{1 + \varepsilon} \right)^k.

    - :math:`\delta` is the ``upperbound`` parameter on the semi-distance between input :math:`p` and solution :math:`q^*`.
    """
    # assert 0 < epsilon < 1, "Error: epsilon should be in (0, 1) strictly, but = {:.3g} is not!".format(epsilon)  # DEBUG
    # eta doesn't depend on p
    eta = epsilon / (1.0 + epsilon)
    # tau_1 and tau_2 depend on p, XXX cannot be precomputed!
    p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    tau_1_p = int(ceil((log(1 - p)) / (log(1 - eta))))
    tau_2_p = int(ceil((log(1 - exp(- epsilon / p))) / (log(1 - eta))))
    # if tau_1_p > tau_2_p:
    #     print("Error: tau_1_p = {:.3g} should be <= tau_2_p = {:.3g}...".format(tau_1_p, tau_2_p))  # DEBUG

    min_of_solutions = float('+inf')
    for k in range(tau_1_p, tau_2_p + 1):
        temp = 1 - (1.0 - eta) ** float(k)
        if upperbound < kullback_leibler_distance_on_mean(p, temp):
            min_of_solutions = min(min_of_solutions, temp)
        else:
            min_of_solutions = min(min_of_solutions, 1)
    return min_of_solutions


class UCBoostEpsilon(IndexPolicy):
    r""" The UCBoostEpsilon policy for bounded bandits (on [0, 1]).

    - It is quite simple: using a set of kl-dominated and candidate semi-distances D, the UCB index for each arm (at each step) is computed as the *smallest* upper confidence bound given (for this arm at this time t) for each distance d.
    - This variant uses :func:`solutions_pb_from_epsilon` to also compute the :math:`\varepsilon` approximation of the :func:`kullback_leibler_distance_on_mean` function (see the article for details, Th.3 p6).
    - Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).
    """

    def __init__(self, nbArms, epsilon=0.01, c=c, lower=0., amplitude=1.):
        super(UCBoostEpsilon, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert c >= 0, "Error: parameter c should be > 0 strictly, but = {:.3g} is not!".format(c)  # DEBUG
        self.c = c  #: Parameter c
        assert 0 < epsilon < 1, "Error: parameter epsilon should be in (0, 1) strictly, but = {:.3g} is not!".format(epsilon)  # DEBUG
        self.epsilon = epsilon  #: Parameter epsilon

    def __str__(self):
        return r"UCBoost($\varepsilon={:.3g}$, $c={:.3g}$)".format(self.epsilon, self.c)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            I_k(t) &= \min_{d\in D_{\varepsilon}} P_1(d)\left(\hat{\mu}_k(t), \frac{\log(t) + c\log(\log(t))}{N_k(t)}\right).
        """
        if self.pulls[arm] < 1:
            return float('+inf')

        p = self.rewards[arm] / self.pulls[arm]
        # upperbound = log(self.t) / self.pulls[arm]
        upperbound = (log(self.t) + self.c * log(max(1, log(self.t)))) / self.pulls[arm]

        min_solutions = min_solutions_pb_from_epsilon(p, upperbound, epsilon=self.epsilon)
        return min(
            min(
                solution_pb_kllb(p, upperbound),
                solution_pb_sq(p, upperbound)
            ),
            min_solutions
        )
