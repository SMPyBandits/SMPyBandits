# -*- coding: utf-8 -*-
""" Kullback-Leibler utilities.
Cf. https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
"""
from __future__ import division, print_function

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.3"

from math import log, sqrt, exp
import numpy as np

# Warning: np.dot is miserably slow!

eps = 1e-15  # Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]


# --- Simple Kullback-Leibler divergence for known distributions


def klBern(x, y):
    """ Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution

    >>> klBern(0.5, 0.5)
    0.0
    >>> klBern(0.1, 0.9)
    1.7577796618689758
    >>> klBern(0.9, 0.1)  # And this KL is symetric
    1.7577796618689758
    >>> klBern(0.4, 0.5)
    0.020135513550688863
    >>> klBern(0.01, 0.99)
    4.503217453131898

    - Special values:

    >>> klBern(0, 1)  # Should be +inf, but 0 --> eps, 1 --> 1 - eps
    34.53957599234081
    """
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))


def klPoisson(x, y):
    """ Kullback-Leibler divergence for Poison distributions. https://en.wikipedia.org/wiki/Poisson_distribution

    >>> klPoisson(3, 3)
    0.0
    >>> klPoisson(2, 1)
    0.3862943611198906
    >>> klPoisson(1, 2)  # And this KL is non-symetric
    0.3068528194400547
    >>> klPoisson(3, 6)
    0.9205584583201643
    >>> klPoisson(6, 8)
    0.2739075652893146

    - Special values:

    >>> klPoisson(1, 0)  # Should be +inf, but 0 --> eps, 1 --> 1 - eps
    33.538776394910684
    >>> klPoisson(0, 0)
    0.0
    """
    x = max(x, eps)
    y = max(y, eps)
    return y - x + x * log(x / y)


def klExp(x, y):
    """ Kullback-Leibler divergence for exponential distributions. https://en.wikipedia.org/wiki/Exponential_distribution

    >>> klExp(3, 3)
    0.0
    >>> klExp(3, 6)
    0.1931471805599453
    >>> klExp(1, 2)  # Only the proportion between x and y is used
    0.1931471805599453
    >>> klExp(2, 1)  # And this KL is non-symetric
    0.3068528194400547
    >>> klExp(4, 2)  # Only the proportion between x and y is used
    0.3068528194400547
    >>> klExp(6, 8)
    0.0376820724517809

    - x, y have to be positive:

    >>> klExp(-3, 2)
    inf
    >>> klExp(3, -2)
    inf
    >>> klExp(-3, -2)
    inf
    """
    if x <= 0 or y <= 0:
        return float('+inf')
    else:
        x = max(x, eps)
        y = max(y, eps)
        return x / y - 1 - log(x / y)


def klGamma(x, y, a=1):
    """ Kullback-Leibler divergence for gamma distributions. https://en.wikipedia.org/wiki/Gamma_distribution

    >>> klGamma(3, 3)
    0.0
    >>> klGamma(3, 6)
    0.1931471805599453
    >>> klGamma(1, 2)  # Only the proportion between x and y is used
    0.1931471805599453
    >>> klGamma(2, 1)  # And this KL is non-symetric
    0.3068528194400547
    >>> klGamma(4, 2)  # Only the proportion between x and y is used
    0.3068528194400547
    >>> klGamma(6, 8)
    0.0376820724517809

    - x, y have to be positive:

    >>> klGamma(-3, 2)
    inf
    >>> klGamma(3, -2)
    inf
    >>> klGamma(-3, -2)
    inf
    """
    if x <= 0 or y <= 0:
        return float('+inf')
    else:
        x = max(x, eps)
        y = max(y, eps)
        return a * (x / y - 1 - log(x / y))


def klNegBin(x, y, r=1):
    """ Kullback-Leibler divergence for negative binomial distributions. https://en.wikipedia.org/wiki/Gamma_distribution

    >>> klNegBin(0.5, 0.5)
    0.0
    >>> klNegBin(0.1, 0.9)
    -0.7116117934648849
    >>> klNegBin(0.9, 0.1)  # And this KL is non-symetric
    2.0321564902394043
    >>> klNegBin(0.4, 0.5)
    -0.13065314341785483
    >>> klNegBin(0.01, 0.99)
    -0.7173536633057466

    - Special values:

    >>> klBern(0, 1)  # Should be +inf, but 0 --> eps, 1 --> 1 - eps
    34.53957599234081

    - With other values for `r`:

    >>> klNegBin(0.5, 0.5, r=2)
    0.0
    >>> klNegBin(0.1, 0.9, r=2)
    -0.8329919030334189
    >>> klNegBin(0.1, 0.9, r=4)
    -0.9148905602182661
    >>> klNegBin(0.9, 0.1, r=2)  # And this KL is non-symetric
    2.332552851091954
    >>> klNegBin(0.4, 0.5, r=2)
    -0.15457261175809217
    >>> klNegBin(0.01, 0.99, r=2)
    -0.8362571425112515
    """
    x = max(x, eps)
    y = max(y, eps)
    return r * log((r + x) / (r + y)) - x * log(y * (r + x) / (x * (r + y)))


def klGauss(x, y, sig2=0.25):
    """ Kullback-Leibler divergence for Gaussian distributions. https://en.wikipedia.org/wiki/Normal_distribution

    >>> klGauss(3, 3)
    0.0
    >>> klGauss(3, 6)
    18.0
    >>> klGauss(1, 2)
    2.0
    >>> klGauss(2, 1)  # And this KL is symetric
    2.0
    >>> klGauss(4, 2)
    8.0
    >>> klGauss(6, 8)
    8.0

    - x, y can be negative:

    >>> klGauss(-3, 2)
    50.0
    >>> klGauss(3, -2)
    50.0
    >>> klGauss(-3, -2)
    2.0
    >>> klGauss(3, 2)
    2.0

    - With other values for `r`:

    >>> klGauss(3, 3, sig2=10)
    0.0
    >>> klGauss(3, 6, sig2=10)
    0.45
    >>> klGauss(1, 2, sig2=10)
    0.05
    >>> klGauss(2, 1, sig2=10)  # And this KL is symetric
    0.05
    >>> klGauss(4, 2, sig2=10)
    0.2
    >>> klGauss(6, 8, sig2=10)
    0.2
    """
    return (x - y) ** 2 / (2 * sig2)


# --- KL functions, for the KL-UCB policy

def klucb(x, d, kl, upperbound, lowerbound=float('-inf'), precision=1e-6):
    """ The generic KL-UCB index computation.

    - x: value of the cum reward,
    - d: upper bound on the divergence,
    - kl: the KL divergence to be used (klBern, klGauss, etc),
    - upperbound, lowerbound=float('-inf'): the known bound of the values x,
    - precision=1e-6: the threshold from where to stop the research,

    - Note: it uses a bisection search.
    """
    value = max(x, lowerbound)
    u = upperbound
    while u - value > precision:
        m = (value + u) / 2.
        if kl(x, m) > d:
            u = m
        else:
            value = m
    return (value + u) / 2.


def klucbBern(x, d, precision=1e-6):
    """ KL-UCB index computation for Bernoulli distributions, using :func:`klucb`.

    - Influence of x:

    >>> klucbBern(0.1, 0.2)
    0.37839145109809247
    >>> klucbBern(0.5, 0.2)
    0.7870888710021973
    >>> klucbBern(0.9, 0.2)
    0.9944896697998048

    - Influence of d:

    >>> klucbBern(0.1, 0.4)
    0.5194755673450786
    >>> klucbBern(0.1, 0.9)
    0.734714937210083

    >>> klucbBern(0.5, 0.4)
    0.8710360527038574
    >>> klucbBern(0.5, 0.9)
    0.9568095207214355

    >>> klucbBern(0.9, 0.4)
    0.9992855072021485
    >>> klucbBern(0.9, 0.9)
    0.9999950408935546
    """
    upperbound = min(1., klucbGauss(x, d))
    # upperbound = min(1., klucbPoisson(x,d))  # also safe, and better ?
    return klucb(x, d, klBern, upperbound, precision)


def klucbGauss(x, d, sig2=1., precision=0.):
    """ KL-UCB index computation for Gaussian distributions.

    - Note that it does not require any search.
    - Warning: it works only if the good variance constant is given.

    - Influence of x:

    >>> klucbGauss(0.1, 0.2)
    0.7324555320336759
    >>> klucbGauss(0.5, 0.2)
    1.132455532033676
    >>> klucbGauss(0.9, 0.2)
    1.532455532033676

    - Influence of d:

    >>> klucbGauss(0.1, 0.4)
    0.9944271909999158
    >>> klucbGauss(0.1, 0.9)
    1.441640786499874

    >>> klucbGauss(0.5, 0.4)
    1.3944271909999157
    >>> klucbGauss(0.5, 0.9)
    1.8416407864998738

    >>> klucbGauss(0.9, 0.4)
    1.7944271909999159
    >>> klucbGauss(0.9, 0.9)
    2.241640786499874
    """
    return x + sqrt(2 * sig2 * d)


def klucbPoisson(x, d, precision=1e-6):
    """ KL-UCB index computation for Poisson distributions, using :func:`klucb`.

    - Influence of x:

    >>> klucbPoisson(0.1, 0.2)
    0.45052392780119604
    >>> klucbPoisson(0.5, 0.2)
    1.0893765430263218
    >>> klucbPoisson(0.9, 0.2)
    1.6401128559741487

    - Influence of d:

    >>> klucbPoisson(0.1, 0.4)
    0.6936844019642616
    >>> klucbPoisson(0.1, 0.9)
    1.2527967047658155

    >>> klucbPoisson(0.5, 0.4)
    1.4229339603816749
    >>> klucbPoisson(0.5, 0.9)
    2.122985165630671

    >>> klucbPoisson(0.9, 0.4)
    2.033691887156203
    >>> klucbPoisson(0.9, 0.9)
    2.8315738094979777
    """
    upperbound = x + d + sqrt(d * d + 2 * x * d)  # looks safe, to check: left (Gaussian) tail of Poisson dev
    return klucb(x, d, klPoisson, upperbound, precision)


def klucbExp(x, d, precision=1e-6):
    """ KL-UCB index computation for exponential distributions, using :func:`klucb`.

    - Influence of x:

    >>> klucbExp(0.1, 0.2)
    0.20274118449172676
    >>> klucbExp(0.5, 0.2)
    1.013706285168157
    >>> klucbExp(0.9, 0.2)
    1.8246716397412546

    - Influence of d:

    >>> klucbExp(0.1, 0.4)
    0.2857928251730546
    >>> klucbExp(0.1, 0.9)
    0.5590884945251575

    >>> klucbExp(0.5, 0.4)
    1.428962647183463
    >>> klucbExp(0.5, 0.9)
    2.7954420946912126

    >>> klucbExp(0.9, 0.4)
    2.572132498767508
    >>> klucbExp(0.9, 0.9)
    5.031795430303065
    """
    if d < 0.77:  # XXX where does this value come from?
        upperbound = x / (1 + 2. / 3 * d - sqrt(4. / 9 * d * d + 2 * d))
        # safe, klexp(x,y) >= e^2/(2*(1-2e/3)) if x=y(1-e)
    else:
        upperbound = x * exp(d + 1)
    if d > 1.61:  # XXX where does this value come from?
        lowerbound = x * exp(d)
    else:
        lowerbound = x / (1 + d - sqrt(d * d + 2 * d))
    return klucb(x, d, klGamma, upperbound, lowerbound, precision)


# --- max EV functions

def maxEV(p, V, klMax):
    """ Maximize expectation of V wrt. q st. KL(p, q) < klMax.

    - Input args.: p, V, klMax.
    - Reference: Section 3.2 of [Filippi, Cappé & Garivier - Allerton, 2011].
    """
    Uq = np.zeros(len(p))
    Kb = p > 0.
    K = ~Kb
    if any(K):
        # Do we need to put some mass on a point where p is zero?
        # If yes, this has to be on one which maximizes V.
        eta = max(V[K])
        J = K & (V == eta)
        if eta > max(V[Kb]):
            y = np.dot(p[Kb], np.log(eta - V[Kb])) + log(np.dot(p[Kb], (1. / (eta - V[Kb]))))
            # print("eta = ", eta, ", y = ", y)
            if y < klMax:
                rb = exp(y - klMax)
                Uqtemp = p[Kb] / (eta - V[Kb])
                Uq[Kb] = rb * Uqtemp / sum(Uqtemp)
                Uq[J] = (1. - rb) / sum(J)
                # or j = min([j for j in range(k) if J[j]])
                # Uq[j] = r
                return Uq
    # Here, only points where p is strictly positive (in Kb) will receive non-zero mass.
    if any(abs(V[Kb] - V[Kb][0]) > 1e-8):
        eta = reseqp(p[Kb], V[Kb], klMax)  # (eta = nu in the article)
        Uq = p / (eta - V)
        Uq = Uq / sum(Uq)
    else:
        # Case where all values in V(Kb) are almost identical.
        Uq[Kb] = 1.0 / len(Kb)
    return Uq


def reseqp(p, V, klMax):
    """ Solve f(reseqp(p, V, klMax)) = klMax, using Newton method.

    - Note: This is a subroutine of maxEV.
    - Reference: Eq. (4) in Section 3.2 of [Filippi, Cappé & Garivier - Allerton, 2011].
    - Warning: `np.dot` is very slow!
    """
    mV = max(V)
    value = mV + 0.1
    tol = 1e-4
    if mV < min(V) + tol:
        return float('inf')
    u = np.dot(p, (1 / (value - V)))
    y = np.dot(p, np.log(value - V)) + log(u) - klMax
    # print("value =", value, ", y = ", y)  # DEBUG
    while abs(y) > tol:
        yp = u - np.dot(p, (1 / (value - V)**2)) / u  # derivative
        value = value - y / yp
        # print("value = ", value)  # DEBUG  # newton iteration
        if value < mV:
            value = (value + y / yp + mV) / 2  # unlikely, but not impossible
        u = np.dot(p, (1 / (value - V)))
        y = np.dot(p, np.log(value - V)) + np.log(u) - klMax
        # print("value = ", value, ", y = ", y)  # DEBUG  # function
    return value


# --- Debugging

if __name__ == "__main__":
    """ Code for debugging purposes."""
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)

    # import matplotlib.pyplot as plt
    # t = np.linspace(0, 1)
    # plt.subplot(2, 1, 1)
    # plt.plot(t, kl(t, 0.6))
    # plt.subplot(2, 1, 2)
    # d = np.linspace(0, 1, 100)
    # plt.plot(d, [klucb(0.3, dd) for dd in d])
    # plt.show()

    print("\nklucbGauss(0.9, 0.2) =", klucbGauss(0.9, 0.2))
    print("klucbBern(0.9, 0.2) =", klucbBern(0.9, 0.2))
    print("klucbPoisson(0.9, 0.2) =", klucbPoisson(0.9, 0.2))

    p = np.array([0., 1.])
    print("\np =", p)
    V = np.array([10, 3])
    print("V =", V)
    klMax = 0.1
    print("klMax =", klMax)
    print("eta = ", reseqp(p, V, klMax))
    print("Uq = ", maxEV(p, V, klMax))

    print("\np =", p)
    p = np.array([0.11794872, 0.27948718, 0.31538462, 0.14102564, 0.0974359, 0.03076923, 0.00769231, 0.01025641, 0.])
    print("V =", V)
    V = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])
    klMax = 0.0168913409484
    print("klMax =", klMax)
    print("eta = ", reseqp(p, V, klMax))
    print("Uq = ", maxEV(p, V, klMax))

    x = 2
    print("\nx =", x)
    d = 2.51
    print("d =", d)
    print("klucbExp(x, d) = ", klucbExp(x, d))

    ub = x / (1 + 2. / 3 * d - sqrt(4. / 9 * d * d + 2 * d))
    print("Upper bound = ", ub)
    print("Stupid upperbound = ", x * exp(d + 1))

    print("\nDone for tests of 'kullback.py' ...")
