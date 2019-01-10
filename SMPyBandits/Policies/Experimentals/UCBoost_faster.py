# -*- coding: utf-8 -*-
""" The UCBoost policy for bounded bandits (on [0, 1]), using a small Cython extension for the intermediate functions that require heavy computations.

- Reference: [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929).

.. warning:: The whole goal of their paper is to provide a numerically efficient alternative to kl-UCB, so for my comparison to be fair, I should either use the Python versions of klUCB utility functions (using :mod:`kullback`) or write C or Cython versions of this UCBoost module. My conclusion is that kl-UCB is *always* faster than UCBoost.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from math import log, sqrt, exp, ceil, floor

# WARNING: this is a HUGE hack to fix a mystery bug on importing this policy
from sys import path
from os.path import dirname
path.insert(0, '/'.join(dirname(__file__).split('/')[:-1]))

try:
    import pyximport; pyximport.install()
    try:
        from .UCBoost_faster_cython import *
    except ImportError:
        from UCBoost_faster_cython import *
except ImportError:
    print("Warning: the 'UCBoost_faster' module failed to import the Cython version of utility functions, defined in 'UCBoost_faster_cython.pyx'. Maybe there is something wrong with your installation of Cython?")  # DEBUG
    try:
        from .UCBoost import *
    except ImportError:
        from UCBoost import *


#: Default value for the constant c used in the computation of the index
c = 3.  #: Default value for the theorems to hold.
c = 0.  #: Default value for better practical performance.

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!


try:
    from .IndexPolicy import IndexPolicy
except ImportError:
    from IndexPolicy import IndexPolicy


# --- New distance and algorithm: quadratic


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
        return r"${}$($c={:.3g}$)".format(r"UCBfaster_{d=d_{sq}}", self.c)

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
        return r"${}$($c={:.3g}$)".format(r"UCBfaster_{d=d_{bq}}", self.c)

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
        return r"${}$($c={:.3g}$)".format(r"UCBfaster_{d=d_h}", self.c)

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
        return r"${}$($c={:.3g}$)".format(r"UCBfaster_{d=d_{lb}}", self.c)

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
        return r"${}$($c={:.3g}$)".format(r"UCBfaster_{d=d_t}", self.c)

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
        return r"UCBoostFaster($|D|={}$, $c={:.3g}$)".format(len(self.set_D), self.c)

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
        return r"UCBoostFaster($D={}$, $c={:.3g}$)".format("\{d_{bq},d_h,d_{lb}\}", self.c)

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
        return r"UCBoostFaster($D={}$, $c={:.3g}$)".format("\{d_{bq},d_h,d_{lb},d_t\}", self.c)

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
        return r"UCBoostFaster($D={}$, $c={:.3g}$)".format("\{d_{bq},d_h,d_{lb},d_t,d_{sq}\}", self.c)

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
        return r"UCBoostFaster($\varepsilon={:.3g}$, $c={:.3g}$)".format(self.epsilon, self.c)

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

del pyximport
