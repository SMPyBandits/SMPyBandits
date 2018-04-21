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


from .kullback import klBern, klucb
from .IndexPolicy import IndexPolicy


# --- New distances

def biquadratic_distance(p, q):
    r"""The *biquadratic distance*, :math:`d_{bq}(p, q) := 2 (p - q)^2 + (4/9) * (p - q)^4`."""
    # return 2 * (p - q)**2 + (4./9) * (p - q)**4
    # XXX about 20% slower than the second less naive solution
    d = 2 * (p - q)**2
    return d + d**2 / 4.


def solution_pb_bq(p, upperbound):
    r"""Closed-form solution of the following optimisation problem, for :math:`d = d_{bq}` the :func:`biquadratic_distance` function:

    .. math::

        P_1(d)(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d(p, q) \leq \delta.

    .. math::

        q^* = \min(1, p + \sqrt{-\frac{4}{9} + \sqrt{\sqrt{81}{16} + \sqrt{9}{4} \delta}).

    - :math:`\delta` is the ``upperbound`` parameter on the semi-distance between input :math:`p` and solution :math:`q^*`.
    """
    # XXX is it faster to precompute the constants ? yes, about 12% faster
    # q_star = np.minimum(1, p + np.sqrt(-4./9 + np.sqrt(81./16 + 9./4 * upperbound)))
    q_star = np.minimum(1, p + np.sqrt(-0.4444444444444444 + np.sqrt(5.0625 + 2.25 * upperbound)))
    assert biquadratic_distance(p, q_star) <= upperbound, "Error: the solution to the optimisation problem P_1(d_bq), with p = {:.3g} and delta = {:.3g} was computed to be q^* = {:.3g} which seem incorrect (bq(p,q^*) = {:.3g} > {:.3g}...".format(p, upperbound, q_star, biquadratic_distance(p, q_star), upperbound)  # DEBUG
    return q_star



def hellinger_distance(p, q):
    r"""The *Hellinger distance*, :math:`d_{h}(p, q) := (\sqrt{p} - \sqrt{q})^2 + (\sqrt{1 - p} - \sqrt{1 - q})^2`."""
    return (np.sqrt(p) - np.sqrt(q))**2 + (np.sqrt(1. - p) - np.sqrt(1. - q))**2


def solution_pb_h(p, upperbound):
    r"""Closed-form solution of the following optimisation problem, for :math:`d = d_{h}` the :func:`hellinger_distance` function:

    .. math::

        P_1(d_h)(p, \delta): & \max_{q \in \Theta} q,\\
        \text{such that }  & d_h(p, q) \leq \delta.

    .. math::

        q^* = \min(1, p + \sqrt{-\frac{4}{9} + \sqrt{\sqrt{81}{16} + \sqrt{9}{4} \delta}).

    - :math:`\delta` is the ``upperbound`` parameter on the semi-distance between input :math:`p` and solution :math:`q^*`.
    """
    # XXX is it faster to precompute the constants ? yes, about 12% faster
    sqrt_p = np.sqrt(p)
    indicator = 1 if upperbound < 2 - 2 * sqrt_p else 0
    q_star = (1 - delta/2.) * np.sqrt(p) + np.sqrt((1 - p) * (upperbound - upperbound**2 / 4.)) ** (2 * indicator)
    assert hellinger_distance(p, q_star) <= upperbound, "Error: the solution to the optimisation problem P_1(d_h), with p = {:.3g} and delta = {:.3g} was computed to be q^* = {:.3g} which seem incorrect (bq(p,q^*) = {:.3g} > {:.3g}...".format(p, upperbound, q_star, hellinger_distance(p, q_star), upperbound)  # DEBUG
    return q_star



#: Default value for the constant c used in the computation of the index
c = 1.  #: Default value.


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
