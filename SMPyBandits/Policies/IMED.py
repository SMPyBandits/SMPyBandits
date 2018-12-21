# -*- coding: utf-8 -*-
""" The IMED policy of [Honda & Takemura, JMLR 2015].

- Reference: [["Non-asymptotic analysis of a new bandit algorithm for semi-bounded rewards", J. Honda and A. Takemura, JMLR, 2015](http://jmlr.csail.mit.edu/papers/volume16/honda15a/honda15a.pdf)].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .DMED import DMED
    from .usenumba import jit
    from .kullback import klBern
except ImportError:
    from DMED import DMED
    from usenumba import jit
    from kullback import klBern


# --- Utilitary functions Dinf

from scipy.optimize import minimize_scalar
from warnings import filterwarnings
filterwarnings("ignore", message="Method 'bounded' does not support relative tolerance in x; defaulting to absolute tolerance", category=RuntimeWarning)


def Dinf(x=None, mu=None, kl=klBern,
        lowerbound=0, upperbound=1,
        precision=1e-6, max_iterations=50
    ):
    r""" The generic Dinf index computation.

    - ``x``: value of the cum reward,
    - ``mu``: upperbound on the mean ``y``,
    - ``kl``: the KL divergence to be used (:func:`klBern`, :func:`klGauss`, etc),
    - ``lowerbound``, ``upperbound=1``: the known bound of the values ``y`` and ``x``,
    - ``precision=1e-6``: the threshold from where to stop the research,
    - ``max_iterations``: max number of iterations of the loop (safer to bound it to reduce time complexity).

    .. math::

        D_{\inf}(x, d) \simeq \inf_{\max(\mu, \mathrm{lowerbound}) \leq y \leq \mathrm{upperbound}} \mathrm{kl}(x, y).

    .. note:: It uses a call the :func:`scipy.optimize.minimize_scalar`. If this fails, it uses a **bisection search**, and one call to ``kl`` for each step of the bisection search.
    """
    # lower and upper bounds
    l = max(lowerbound, mu)
    u = upperbound
    def f(i): return kl(x, i)  # objective function!
    res = minimize_scalar(f, bounds=[l, u],
        method="bounded",
        tol=precision,
        options={"maxiter": max_iterations, "disp": False}
    )
    if hasattr(res, "x"):
        return res.x
    else:
        print("Warning: the call to scipy.optimize.minimize_scalar failed, using hand-written bisection search instead...")  # DEBUG
    # start in the middle
    value = (l + u) / 2.
    current_kl = kl(x, value)
    _count_iteration = 0
    while _count_iteration < max_iterations and u - value > precision:
        _count_iteration += 1
        # try to see right
        value = (l + u) / 2.
        new_kl = kl(x, value)
        if new_kl > current_kl:  # need to go left
            u = value
        else:  # need to go right
            current_kl = new_kl
            l = value
    return current_kl


# --- IMED

class IMED(DMED):
    """ The IMED policy of [Honda & Takemura, JMLR 2015].

    - Reference: [["Non-asymptotic analysis of a new bandit algorithm for semi-bounded rewards", J. Honda and A. Takemura, JMLR, 2015](http://jmlr.csail.mit.edu/papers/volume16/honda15a/honda15a.pdf)].
    """

    def __init__(self, nbArms, tolerance=1e-4, kl=klBern, lower=0., amplitude=1.):
        super(IMED, self).__init__(nbArms, tolerance=tolerance, kl=kl, lower=lower, amplitude=amplitude)

    def __str__(self):
        return r"IMED({})".format(self.kl.__name__[2:])

    def one_Dinf(self, x, mu):
        r""" Compute the :math:`D_{\inf}` solution, for one value of ``x``, and one value for ``mu``."""
        return Dinf(x=x, mu=mu, kl=self.kl, lowerbound=self.lower, upperbound=self.lower + self.amplitude, precision=self.tolerance)

    # XXX Use this hack to vectorize one_Dinf ?
    # Dinf = np.vectorize(one_Dinf) ?
    def Dinf(self, xs, mu):
        r""" Compute the :math:`D_{\inf}` solution, for a vector of value of ``xs``, and one value for ``mu``."""
        return np.array([ self.one_Dinf(x, mu) for x in xs ])

    def choice(self):
        r""" Choose an arm with **minimal** index (uniformly at random):

        .. math:: A(t) \sim U(\arg\min_{1 \leq k \leq K} I_k(t)).

        Where the indexes are:

        .. math:: I_k(t) = N_k(t) D_{\inf}(\hat{\mu_{k}}(t), \max_{k'} \hat{\mu_{k'}}(t)) + \log(N_k(t)).
        """
        empiricalMeans = self.rewards / self.pulls
        bestEmpiricalMean = np.max(empiricalMeans)
        values_Dinf = self.Dinf(empiricalMeans, bestEmpiricalMean)

        # now compute the indexes
        indexes_to_minimize = self.pulls * values_Dinf + np.log(self.pulls)
        indexes_to_minimize[self.pulls < 1] = float('-inf')

        # then do as IndexPolicy but with a min instead
        # try:
        return np.random.choice(np.nonzero(indexes_to_minimize == np.min(indexes_to_minimize))[0])
        # except ValueError:
        #     if not np.all(np.isnan(indexes_to_minimize)):
        #         raise ValueError("Error: unknown error in IMED.choice(): the indexes were {} but couldn't be used to select an arm.".format(indexes_to_minimize))
        #     return np.random.randint(self.nbArms)
