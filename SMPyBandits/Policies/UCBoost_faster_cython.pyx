# -*- coding: utf-8 -*-
""" Optimized version of some utility functions for the UCBoost module, that should be compiled using Cython (http://docs.cython.org/).

.. warning::

    This extension should be used with the ``setup.py`` script, by running::

        $ python setup.py build_ext --inplace

    You can also use [pyximport](http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html#pyximport-cython-compilation-for-developers) to import the ``kullback_cython`` module transparently:

    >>> import pyximport; pyximport.install()  # instantaneous  # doctest: +ELLIPSIS
    (None, <pyximport.pyximport.PyxImporter at 0x...>)
    >>> from UCBoost_faster_cython import *     # takes about two seconds
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from libc.math cimport log, sqrt, exp, ceil, floor


cdef float eps = 1e-5  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]


#: Default value for the constant c used in the computation of the index
cdef float c
c = 3.  #: Default value for the theorems to hold.
c = 0.  #: Default value for better practical performance.


#: Tolerance when checking (with ``assert``) that the solution(s) of any convex problem are correct.
cdef float tolerance_with_upperbound = 1.0001


# --- New distance and algorithm: quadratic

def squadratic_distance(float p, float q) -> float:
    r""" The *quadratic distance*, :math:`d_{sq}(p, q) := 2 (p - q)^2`."""
    # p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    # q = min(max(q, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    return 2 * (p - q)**2


def solution_pb_sq(float p, float upperbound) -> float:
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


# --- New distance and algorithm: biquadratic

def biquadratic_distance(float p, float q) -> float:
    r""" The *biquadratic distance*, :math:`d_{bq}(p, q) := 2 (p - q)^2 + (4/9) * (p - q)^4`."""
    # return 2 * (p - q)**2 + (4./9) * (p - q)**4
    # XXX about 20% slower than the second less naive solution
    # p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    # q = min(max(q, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    cdef float d = 2 * (p - q)**2
    return d + d**2 / 9.


def solution_pb_bq(float p, float upperbound) -> float:
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


# --- New distance and algorithm: Hellinger

def hellinger_distance(float p, float q) -> float:
    r""" The *Hellinger distance*, :math:`d_{h}(p, q) := (\sqrt{p} - \sqrt{q})^2 + (\sqrt{1 - p} - \sqrt{1 - q})^2`."""
    # p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    # q = min(max(q, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    return (sqrt(p) - sqrt(q))**2 + (sqrt(1. - p) - sqrt(1. - q))**2


def solution_pb_hellinger(float p, float upperbound) -> float:
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
    cdef float sqrt_p = sqrt(p)
    if upperbound < (2 - 2 * sqrt_p):
        return (1 - upperbound/2.) * sqrt_p + sqrt((1 - p) * (upperbound - upperbound**2 / 4.)) ** 2
    else:
        return p


# --- New distance and algorithm: lower-bound on the Kullback-Leibler distance


def kullback_leibler_distance_on_mean(float p, float q) -> float:
    r""" Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution#Kullback.E2.80.93Leibler_divergence

    .. math:: \mathrm{kl}(p, q) = \mathrm{KL}(\mathcal{B}(p), \mathcal{B}(q)) = p \log\left(\frac{p}{q}\right) + (1-p) \log\left(\frac{1-p}{1-q}\right).
    """
    p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    q = min(max(q, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))


def kullback_leibler_distance_lowerbound(float p, float q) -> float:
    r""" Lower-bound on the Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution#Kullback.E2.80.93Leibler_divergence

    .. math:: d_{lb}(p, q) = p \log\left( p \right) + (1-p) \log\left(\frac{1-p}{1-q}\right).
    """
    p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    q = min(max(q, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    return p * log(p) + (1 - p) * log((1 - p) / (1 - q))


def solution_pb_kllb(float p, float upperbound) -> float:
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


# --- New distance and algorithm: a shifted tangent line function of d_kl


def distance_t(float p, float q) -> float:
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



def solution_pb_t(float p, float upperbound) -> float:
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



# --- New distance and algorithm: epsilon approximation on the Kullback-Leibler distance

def min_solutions_pb_from_epsilon(float p, float upperbound, float epsilon=0.001) -> float:
    r""" Minimum of the closed-form solutions of the following optimisation problems, for :math:`d = d_s^k` approximation of :math:`d_{kl}` and any :math:`\tau_1(p) \leq k \leq \tau_2(p)`:

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
    cdef float eta = epsilon / (1.0 + epsilon)
    cdef float temp
    # tau_1 and tau_2 depend on p, XXX cannot be precomputed!
    p = min(max(p, eps), 1 - eps)  # XXX project [0,1] to [eps,1-eps]
    cdef int tau_1_p = int(ceil((log(1 - p)) / (log(1 - eta))))
    cdef int tau_2_p = int(ceil((log(1 - exp(- epsilon / p))) / (log(1 - eta))))
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
