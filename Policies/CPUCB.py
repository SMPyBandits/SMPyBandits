# -*- coding: utf-8 -*-
""" The Clopper-Pearson UCB policy for bounded bandits.
Reference: [Garivier & Cappé, COLT 2011](https://arxiv.org/pdf/1102.2490.pdf).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.6"

import numpy as np
import scipy.stats
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .UCB import UCB


def binofit_scalar(x, n, alpha=0.05):
    r""" Parameter estimates and confidence intervals for binomial data.

    For example:

    >>> np.random.seed(1234)  # reproducible results
    >>> true_p = 0.6
    >>> N = 100
    >>> x = np.random.binomial(N, true_p)
    >>> (phat, pci) = binofit_scalar(x, N)
    >>> phat
    0.61
    >>> pci  # 0.6 of course lies in the 95% confidence interval  # doctest: +ELLIPSIS
    (0.507..., 0.705...)
    >>> (phat, pci) = binofit_scalar(x, N, 0.01)
    >>> pci  # 0.6 is also in the 99% confidence interval, but it is larger  # doctest: +ELLIPSIS
    (0.476..., 0.732...)

    Like binofit_scalar in MATLAB, see https://fr.mathworks.com/help/stats/binofit_scalar.html.

    - ``(phat, pci) = binofit_scalar(x, n)`` returns a maximum likelihood estimate of the probability of success in a given binomial trial based on the number of successes, ``x``, observed in ``n`` independent trials.

    - ``(phat, pci) = binofit_scalar(x, n)`` returns the probability estimate, phat, and the 95% confidence intervals, pci, by using the `Clopper-Pearson method <https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper-Pearson_interval>`_ to calculate confidence intervals.

    - ``(phat, pci) = binofit_scalar(x, n, alpha)`` returns the ``100(1 - alpha)%`` confidence intervals. For example, ``alpha = 0.01`` yields ``99%`` confidence intervals.


    For the Clopper-Pearson UCB algorithms:

    - x is the cum rewards of some arm k, :math:`x = X_k(t)`,
    - n is the number of samples of that arm k, :math:`n = N_k(t)`,
    - and alpha is a small positive number, :math:`\alpha = \frac{1}{t^c}` in this algorithm (for :math:`c > 1, \simeq 1`, for instance `c = 1.01`).

    Returns: (phat, pci)

    - phat: is the estimate of p
    - pci: is the confidence interval

    .. note::

       My reference implementation was https://github.com/sjara/extracellpy/blob/master/extrastats.py#L35,
       but http://statsmodels.sourceforge.net/devel/generated/statsmodels.stats.proportion.proportion_confint.html can also be used (it implies an extra requirement for the project).
    """
    # - If ``x = [x[0], x[1], ... x[k-1]]` is a vector, binofit_scalar returns a vector of the same size as ``x`` whose ``i``th entry is the parameter estimate for ``x[i]``.
    #   All ``k`` estimates are independent of each other.

    # - If ``n = [n[0], n[1], ... n[k-1]]`` is a vector of the same size as ``x``, the binomial fit, ``binofit_scalar``, returns a vector whose ``i``th entry is the parameter estimate based on the number of successes ``x[i]`` in ``n[i]`` independent trials.
    #   A scalar value for ``x`` or ``n`` is expanded to the same size as the other input.
    #
    if n < 1:  # Extreme case
        phat = np.NaN
        pci = (np.NaN, np.NaN)
    else:
        assert 0 <= x <= n, "Error: binofit_scalar(x, n) invalid value for x, not in [0, n], invalid outcome of binomial trials."  # DEBUG
        # Empirical mean, for one parameter
        phat = float(x) / float(n)
        # See https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper-Pearson_interval for the computation
        # Lowerbound
        nu1 = 2 * x
        nu2 = 2 * (n - x + 1)
        F = scipy.stats.f.ppf(alpha / 2., nu1, nu2)
        lowerbound = (nu1 * F) / (nu2 + nu1 * F)
        if x == 0:  # extreme left case, truncate lowerbound to 0
            lowerbound = 0
            # upperbound = 1 - (alpha / 2.)**(1. / float(n))  # closed-form is available!
        # Upperbound
        nu1 = 2 * (x + 1)
        nu2 = 2 * (n - x)
        F = scipy.stats.f.ppf(1 - (alpha / 2.), nu1, nu2)
        upperbound = (nu1 * F) / (nu2 + nu1 * F)
        if x == n:  # extreme right case, truncate upperbound to 1
            # lowerbound = (alpha / 2.)**(1. / float(n))  # closed-form is available!
            upperbound = 1
        pci = (lowerbound, upperbound)
    return phat, pci


def binofit(xArray, nArray, alpha=0.05):
    """ Parameter estimates and confidence intervals for binomial data, for vectorial inputs.

    For example:

    >>> np.random.seed(1234)  # reproducible results
    >>> true_p = 0.6
    >>> N = 100
    >>> xArray = np.random.binomial(N, true_p, 4)
    >>> xArray
    array([61, 54, 61, 52])

    >>> (phat, pci) = binofit(xArray, N)
    >>> phat
    array([ 0.61,  0.54,  0.61,  0.52])
    >>> pci  # 0.6 of course lies in the 95% confidence intervals  # doctest: +ELLIPSIS
    array([[ 0.507...,  0.705...],
           [ 0.437...,  0.640...],
           [ 0.507...,  0.705...],
           [ 0.417...,  0.620...]])

    >>> (phat, pci) = binofit(xArray, N, 0.01)
    >>> pci  # 0.6 is also in the 99% confidence intervals, but it is larger  # doctest: +ELLIPSIS
    array([[ 0.476...,  0.732...],
           [ 0.407...,  0.668...],
           [ 0.476...,  0.732...],
           [ 0.387...,  0.650...]])
    """
    # If inputs are list or tuples
    if isinstance(xArray, (list, tuple)):
        xArray = np.asarray(xArray)
    if isinstance(nArray, (list, tuple)):
        nArray = np.asarray(nArray)
    # If x is vectorial
    if isinstance(xArray, np.ndarray):
        origShape = xArray.shape
        Psuccess = np.empty(xArray.size)
        ConfIntervals = np.empty((xArray.size, 2))
        if not isinstance(nArray, np.ndarray):
            nArray = nArray * np.ones_like(xArray)
        if not isinstance(alpha, np.ndarray):
            alpha = alpha * np.ones_like(xArray)
        for inde in range(xArray.size):
            Psuccess[inde], ConfIntervals[inde, :] = binofit_scalar(xArray.flat[inde], nArray.flat[inde], alpha.flat[inde])
        return Psuccess.reshape(origShape), ConfIntervals.reshape(origShape + (2, ))
    else:
        return binofit_scalar(xArray, nArray, alpha)


def ClopperPearsonUCB(x, N, alpha=0.05):
    """ Returns just the upper-confidence bound of the confidence interval. """
    phat, (lowerbound, upperbound) = binofit(x, N, alpha=alpha)
    return upperbound


# # Define a vectorized clopperPearsonUCB function, in ONE line!
# clopperPearsonUCB = np.vectorize(ClopperPearsonUCB)
# clopperPearsonUCB.__doc__ = ClopperPearsonUCB.__doc__


#: Default value for the parameter c for CP-UCB
C = 1.01


class CPUCB(UCB):
    """ The Clopper-Pearson UCB policy for bounded bandits.
    Reference: [Garivier & Cappé, COLT 2011].
    """

    def __init__(self, nbArms, c=C, lower=0., amplitude=1.):
        super(CPUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.c = c  #: Parameter c for the CP-UCB formula (see below)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math:: I_k(t) = \mathrm{ClopperPearsonUCB}\left( X_k(t), N_k(t), \frac{1}{t^c} \right).

        Where :math:`\mathrm{ClopperPearsonUCB}` is defined above.
        The index is the upper-confidence bound of the binomial trial of :math:`N_k(t)` samples from arm k,
        having mean :math:`\mu_k`, and empirical outcome :math:`X_k(t)`.
        The confidence interval is with :math:`\alpha = 1 / t^c`, for a :math:`100(1 - \alpha)\%` confidence bound.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return ClopperPearsonUCB(self.rewards[arm], self.pulls[arm], 1. / (self.t ** self.c))

    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     indexes = ClopperPearsonUCB(self.rewards, self.pulls, 1. / (self.t ** self.c))
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index[:] = indexes


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
