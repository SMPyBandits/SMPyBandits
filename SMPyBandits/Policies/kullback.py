# -*- coding: utf-8 -*-
""" Kullback-Leibler divergence functions and klUCB utilities.

- Faster implementation can be found in a C file, in ``Policies/C``, and should be compiled to speedup computations.
- However, the version here have examples, doctests, and are jit compiled on the fly (with numba, cf. http://numba.pydata.org/).
- Cf. https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
- Reference: [Filippi, Cappé & Garivier - Allerton, 2011](https://arxiv.org/pdf/1004.5229.pdf) and [Garivier & Cappé, 2011](https://arxiv.org/pdf/1102.2490.pdf)


.. warning::

    All functions are *not* vectorized, and assume only one value for each argument.
    If you want vectorized function, use the wrapper :py:class:`numpy.vectorize`:

    >>> import numpy as np
    >>> klBern_vect = np.vectorize(klBern)
    >>> klBern_vect([0.1, 0.5, 0.9], 0.2)  # doctest: +ELLIPSIS
    array([0.036..., 0.223..., 1.145...])
    >>> klBern_vect(0.4, [0.2, 0.3, 0.4])  # doctest: +ELLIPSIS
    array([0.104..., 0.022..., 0...])
    >>> klBern_vect([0.1, 0.5, 0.9], [0.2, 0.3, 0.4])  # doctest: +ELLIPSIS
    array([0.036..., 0.087..., 0.550...])

    For some functions, you would be better off writing a vectorized version manually, for instance if you want to fix a value of some optional parameters:

    >>> # WARNING using np.vectorize gave weird result on klGauss
    >>> # klGauss_vect = np.vectorize(klGauss, excluded="y")
    >>> def klGauss_vect(xs, y, sig2x=0.25):  # vectorized for first input only
    ...    return np.array([klGauss(x, y, sig2x) for x in xs])
    >>> klGauss_vect([-1, 0, 1], 0.1)  # doctest: +ELLIPSIS
    array([2.42, 0.02, 1.62])
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.9"

from math import log, sqrt, exp

import numpy as np

# https://www.docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.fixed_point
from scipy import optimize

try:
    from .usenumba import jit  # Import numba.jit or a dummy jit(f)=f
except (ValueError, ImportError, SystemError):
    from usenumba import jit  # Import numba.jit or a dummy jit(f)=f


eps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]


# --- Simple Kullback-Leibler divergence for known distributions


@jit
def klBern(x, y):
    r""" Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution#Kullback.E2.80.93Leibler_divergence

    .. math:: \mathrm{KL}(\mathcal{B}(x), \mathcal{B}(y)) = x \log(\frac{x}{y}) + (1-x) \log(\frac{1-x}{1-y}).

    >>> klBern(0.5, 0.5)
    0.0
    >>> klBern(0.1, 0.9)  # doctest: +ELLIPSIS
    1.757779...
    >>> klBern(0.9, 0.1)  # And this KL is symmetric  # doctest: +ELLIPSIS
    1.757779...
    >>> klBern(0.4, 0.5)  # doctest: +ELLIPSIS
    0.020135...
    >>> klBern(0.01, 0.99)  # doctest: +ELLIPSIS
    4.503217...

    - Special values:

    >>> klBern(0, 1)  # Should be +inf, but 0 --> eps, 1 --> 1 - eps  # doctest: +ELLIPSIS
    34.539575...
    """
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))


@jit
def klBin(x, y, n):
    r""" Kullback-Leibler divergence for Binomial distributions. https://math.stackexchange.com/questions/320399/kullback-leibner-divergence-of-binomial-distributions

    - It is simply the n times :func:`klBern` on x and y.

    .. math:: \mathrm{KL}(\mathrm{Bin}(x, n), \mathrm{Bin}(y, n)) = n \times \left(x \log(\frac{x}{y}) + (1-x) \log(\frac{1-x}{1-y}) \right).

    .. warning:: The two distributions must have the same parameter n, and x, y are p, q in (0, 1).

    >>> klBin(0.5, 0.5, 10)
    0.0
    >>> klBin(0.1, 0.9, 10)  # doctest: +ELLIPSIS
    17.57779...
    >>> klBin(0.9, 0.1, 10)  # And this KL is symmetric  # doctest: +ELLIPSIS
    17.57779...
    >>> klBin(0.4, 0.5, 10)  # doctest: +ELLIPSIS
    0.20135...
    >>> klBin(0.01, 0.99, 10)  # doctest: +ELLIPSIS
    45.03217...

    - Special values:

    >>> klBin(0, 1, 10)  # Should be +inf, but 0 --> eps, 1 --> 1 - eps  # doctest: +ELLIPSIS
    345.39575...
    """
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return n * (x * log(x / y) + (1 - x) * log((1 - x) / (1 - y)))


@jit
def klPoisson(x, y):
    r""" Kullback-Leibler divergence for Poison distributions. https://en.wikipedia.org/wiki/Poisson_distribution#Kullback.E2.80.93Leibler_divergence

    .. math:: \mathrm{KL}(\mathrm{Poisson}(x), \mathrm{Poisson}(y)) = y - x + x \times \log(\frac{x}{y}).

    >>> klPoisson(3, 3)
    0.0
    >>> klPoisson(2, 1)  # doctest: +ELLIPSIS
    0.386294...
    >>> klPoisson(1, 2)  # And this KL is non-symmetric  # doctest: +ELLIPSIS
    0.306852...
    >>> klPoisson(3, 6)  # doctest: +ELLIPSIS
    0.920558...
    >>> klPoisson(6, 8)  # doctest: +ELLIPSIS
    0.273907...

    - Special values:

    >>> klPoisson(1, 0)  # Should be +inf, but 0 --> eps, 1 --> 1 - eps  # doctest: +ELLIPSIS
    33.538776...
    >>> klPoisson(0, 0)
    0.0
    """
    x = max(x, eps)
    y = max(y, eps)
    return y - x + x * log(x / y)


inf = float('+inf')

@jit
def klExp(x, y):
    r""" Kullback-Leibler divergence for exponential distributions. https://en.wikipedia.org/wiki/Exponential_distribution#Kullback.E2.80.93Leibler_divergence

    .. math::

        \mathrm{KL}(\mathrm{Exp}(x), \mathrm{Exp}(y)) = \begin{cases}
        \frac{x}{y} - 1 - \log(\frac{x}{y}) & \text{if} x > 0, y > 0\\
        +\infty & \text{otherwise}
        \end{cases}

    >>> klExp(3, 3)
    0.0
    >>> klExp(3, 6)  # doctest: +ELLIPSIS
    0.193147...
    >>> klExp(1, 2)  # Only the proportion between x and y is used  # doctest: +ELLIPSIS
    0.193147...
    >>> klExp(2, 1)  # And this KL is non-symmetric  # doctest: +ELLIPSIS
    0.306852...
    >>> klExp(4, 2)  # Only the proportion between x and y is used  # doctest: +ELLIPSIS
    0.306852...
    >>> klExp(6, 8)  # doctest: +ELLIPSIS
    0.037682...

    - x, y have to be positive:

    >>> klExp(-3, 2)
    inf
    >>> klExp(3, -2)
    inf
    >>> klExp(-3, -2)
    inf
    """
    if x <= 0 or y <= 0:
        return inf
    else:
        x = max(x, eps)
        y = max(y, eps)
        return x / y - 1 - log(x / y)


@jit
def klGamma(x, y, a=1):
    r""" Kullback-Leibler divergence for gamma distributions. https://en.wikipedia.org/wiki/Gamma_distribution#Kullback.E2.80.93Leibler_divergence

    - It is simply the a times :func:`klExp` on x and y.

    .. math::

        \mathrm{KL}(\Gamma(x, a), \Gamma(y, a)) = \begin{cases}
        a \times \left( \frac{x}{y} - 1 - \log(\frac{x}{y}) \right) & \text{if} x > 0, y > 0\\
        +\infty & \text{otherwise}
        \end{cases}

    .. warning:: The two distributions must have the same parameter a.

    >>> klGamma(3, 3)
    0.0
    >>> klGamma(3, 6)  # doctest: +ELLIPSIS
    0.193147...
    >>> klGamma(1, 2)  # Only the proportion between x and y is used  # doctest: +ELLIPSIS
    0.193147...
    >>> klGamma(2, 1)  # And this KL is non-symmetric  # doctest: +ELLIPSIS
    0.306852...
    >>> klGamma(4, 2)  # Only the proportion between x and y is used  # doctest: +ELLIPSIS
    0.306852...
    >>> klGamma(6, 8)  # doctest: +ELLIPSIS
    0.037682...

    - x, y have to be positive:

    >>> klGamma(-3, 2)
    inf
    >>> klGamma(3, -2)
    inf
    >>> klGamma(-3, -2)
    inf
    """
    if x <= 0 or y <= 0:
        return inf
    else:
        x = max(x, eps)
        y = max(y, eps)
        return a * (x / y - 1 - log(x / y))


@jit
def klNegBin(x, y, r=1):
    r""" Kullback-Leibler divergence for negative binomial distributions. https://en.wikipedia.org/wiki/Negative_binomial_distribution

    .. math:: \mathrm{KL}(\mathrm{NegBin}(x, r), \mathrm{NegBin}(y, r)) = r \times \log((r + x) / (r + y)) - x \times \log(y \times (r + x) / (x \times (r + y))).

    .. warning:: The two distributions must have the same parameter r.

    >>> klNegBin(0.5, 0.5)
    0.0
    >>> klNegBin(0.1, 0.9)  # doctest: +ELLIPSIS
    -0.711611...
    >>> klNegBin(0.9, 0.1)  # And this KL is non-symmetric  # doctest: +ELLIPSIS
    2.0321564...
    >>> klNegBin(0.4, 0.5)  # doctest: +ELLIPSIS
    -0.130653...
    >>> klNegBin(0.01, 0.99)  # doctest: +ELLIPSIS
    -0.717353...

    - Special values:

    >>> klBern(0, 1)  # Should be +inf, but 0 --> eps, 1 --> 1 - eps  # doctest: +ELLIPSIS
    34.539575...

    - With other values for `r`:

    >>> klNegBin(0.5, 0.5, r=2)
    0.0
    >>> klNegBin(0.1, 0.9, r=2)  # doctest: +ELLIPSIS
    -0.832991...
    >>> klNegBin(0.1, 0.9, r=4)  # doctest: +ELLIPSIS
    -0.914890...
    >>> klNegBin(0.9, 0.1, r=2)  # And this KL is non-symmetric  # doctest: +ELLIPSIS
    2.3325528...
    >>> klNegBin(0.4, 0.5, r=2)  # doctest: +ELLIPSIS
    -0.154572...
    >>> klNegBin(0.01, 0.99, r=2)  # doctest: +ELLIPSIS
    -0.836257...
    """
    x = max(x, eps)
    y = max(y, eps)
    return r * log((r + x) / (r + y)) - x * log(y * (r + x) / (x * (r + y)))


@jit
def klGauss(x, y, sig2x=0.25, sig2y=None):
    r""" Kullback-Leibler divergence for Gaussian distributions of means ``x`` and ``y`` and variances ``sig2x`` and ``sig2y``, :math:`\nu_1 = \mathcal{N}(x, \sigma_x^2)` and :math:`\nu_2 = \mathcal{N}(y, \sigma_x^2)`:

    .. math:: \mathrm{KL}(\nu_1, \nu_2) = \frac{(x - y)^2}{2 \sigma_y^2} + \frac{1}{2}\left( \frac{\sigma_x^2}{\sigma_y^2} - 1 \log\left(\frac{\sigma_x^2}{\sigma_y^2}\right) \right).

    See https://en.wikipedia.org/wiki/Normal_distribution#Other_properties

    - By default, sig2y is assumed to be sig2x (same variance).

    .. warning:: The C version does not support different variances.

    >>> klGauss(3, 3)
    0.0
    >>> klGauss(3, 6)
    18.0
    >>> klGauss(1, 2)
    2.0
    >>> klGauss(2, 1)  # And this KL is symmetric
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

    - With other values for `sig2x`:

    >>> klGauss(3, 3, sig2x=10)
    0.0
    >>> klGauss(3, 6, sig2x=10)
    0.45
    >>> klGauss(1, 2, sig2x=10)
    0.05
    >>> klGauss(2, 1, sig2x=10)  # And this KL is symmetric
    0.05
    >>> klGauss(4, 2, sig2x=10)
    0.2
    >>> klGauss(6, 8, sig2x=10)
    0.2

    - With different values for `sig2x` and `sig2y`:

    >>> klGauss(0, 0, sig2x=0.25, sig2y=0.5)  # doctest: +ELLIPSIS
    -0.0284...
    >>> klGauss(0, 0, sig2x=0.25, sig2y=1.0)  # doctest: +ELLIPSIS
    0.2243...
    >>> klGauss(0, 0, sig2x=0.5, sig2y=0.25)  # not symmetric here!  # doctest: +ELLIPSIS
    1.1534...

    >>> klGauss(0, 1, sig2x=0.25, sig2y=0.5)  # doctest: +ELLIPSIS
    0.9715...
    >>> klGauss(0, 1, sig2x=0.25, sig2y=1.0)  # doctest: +ELLIPSIS
    0.7243...
    >>> klGauss(0, 1, sig2x=0.5, sig2y=0.25)  # not symmetric here!  # doctest: +ELLIPSIS
    3.1534...

    >>> klGauss(1, 0, sig2x=0.25, sig2y=0.5)  # doctest: +ELLIPSIS
    0.9715...
    >>> klGauss(1, 0, sig2x=0.25, sig2y=1.0)  # doctest: +ELLIPSIS
    0.7243...
    >>> klGauss(1, 0, sig2x=0.5, sig2y=0.25)  # not symmetric here!  # doctest: +ELLIPSIS
    3.1534...

    .. warning:: Using :class:`Policies.klUCB` (and variants) with :func:`klGauss` is equivalent to use :class:`Policies.UCB`, so prefer the simpler version.
    """
    if sig2y is None or - eps < (sig2y - sig2x) < eps:
        return (x - y) ** 2 / (2. * sig2x)
    else:
        return (x - y) ** 2 / (2. * sig2y) + 0.5 * ((sig2x/sig2y)**2 - 1 - log(sig2x/sig2y))


# --- KL functions, for the KL-UCB policy

@jit
def klucb(x, d, kl, upperbound,
        precision=1e-6, lowerbound=float('-inf'), max_iterations=50,
    ):
    r""" The generic KL-UCB index computation.

    - ``x``: value of the cum reward,
    - ``d``: upper bound on the divergence,
    - ``kl``: the KL divergence to be used (:func:`klBern`, :func:`klGauss`, etc),
    - ``upperbound``, ``lowerbound=float('-inf')``: the known bound of the values ``x``,
    - ``precision=1e-6``: the threshold from where to stop the research,
    - ``max_iterations=50``: max number of iterations of the loop (safer to bound it to reduce time complexity).

    .. math::

        \mathrm{klucb}(x, d) \simeq \sup_{\mathrm{lowerbound} \leq y \leq \mathrm{upperbound}} \{ y : \mathrm{kl}(x, y) < d \}.

    .. note:: It uses a **bisection search**, and one call to ``kl`` for each step of the bisection search.

    For example, for :func:`klucbBern`, the two steps are to first compute an upperbound (as precise as possible) and the compute the kl-UCB index:

    >>> x, d = 0.9, 0.2   # mean x, exploration term d
    >>> upperbound = min(1., klucbGauss(x, d, sig2x=0.25))  # variance 1/4 for [0,1] bounded distributions
    >>> upperbound  # doctest: +ELLIPSIS
    1.0
    >>> klucb(x, d, klBern, upperbound, lowerbound=0, precision=1e-3, max_iterations=10)  # doctest: +ELLIPSIS
    0.9941...
    >>> klucb(x, d, klBern, upperbound, lowerbound=0, precision=1e-6, max_iterations=10)  # doctest: +ELLIPSIS
    0.9944...
    >>> klucb(x, d, klBern, upperbound, lowerbound=0, precision=1e-3, max_iterations=50)  # doctest: +ELLIPSIS
    0.9941...
    >>> klucb(x, d, klBern, upperbound, lowerbound=0, precision=1e-6, max_iterations=100)  # more and more precise!  # doctest: +ELLIPSIS
    0.994489...

    .. note:: See below for more examples for different KL divergence functions.
    """
    value = max(x, lowerbound)
    u = upperbound
    _count_iteration = 0
    while _count_iteration < max_iterations and u - value > precision:
        _count_iteration += 1
        m = (value + u) * 0.5
        if kl(x, m) > d:
            u = m
        else:
            value = m
    return (value + u) * 0.5


@jit
def klucbBern(x, d, precision=1e-6):
    """ KL-UCB index computation for Bernoulli distributions, using :func:`klucb`.

    - Influence of x:

    >>> klucbBern(0.1, 0.2)  # doctest: +ELLIPSIS
    0.378391...
    >>> klucbBern(0.5, 0.2)  # doctest: +ELLIPSIS
    0.787088...
    >>> klucbBern(0.9, 0.2)  # doctest: +ELLIPSIS
    0.994489...

    - Influence of d:

    >>> klucbBern(0.1, 0.4)  # doctest: +ELLIPSIS
    0.519475...
    >>> klucbBern(0.1, 0.9)  # doctest: +ELLIPSIS
    0.734714...

    >>> klucbBern(0.5, 0.4)  # doctest: +ELLIPSIS
    0.871035...
    >>> klucbBern(0.5, 0.9)  # doctest: +ELLIPSIS
    0.956809...

    >>> klucbBern(0.9, 0.4)  # doctest: +ELLIPSIS
    0.999285...
    >>> klucbBern(0.9, 0.9)  # doctest: +ELLIPSIS
    0.999995...
    """
    upperbound = min(1., klucbGauss(x, d, sig2x=0.25, precision=precision))  # variance 1/4 for [0,1] bounded distributions
    # upperbound = min(1., klucbPoisson(x, d))  # also safe, and better ?
    return klucb(x, d, klBern, upperbound, precision)


@jit
def klucbGauss(x, d, sig2x=0.25, precision=0.):
    """ KL-UCB index computation for Gaussian distributions.

    - Note that it does not require any search.

    .. warning:: it works only if the good variance constant is given.

    - Influence of x:

    >>> klucbGauss(0.1, 0.2)  # doctest: +ELLIPSIS
    0.416227...
    >>> klucbGauss(0.5, 0.2)  # doctest: +ELLIPSIS
    0.816227...
    >>> klucbGauss(0.9, 0.2)  # doctest: +ELLIPSIS
    1.216227...

    - Influence of d:

    >>> klucbGauss(0.1, 0.4)  # doctest: +ELLIPSIS
    0.547213...
    >>> klucbGauss(0.1, 0.9)  # doctest: +ELLIPSIS
    0.770820...

    >>> klucbGauss(0.5, 0.4)  # doctest: +ELLIPSIS
    0.947213...
    >>> klucbGauss(0.5, 0.9)  # doctest: +ELLIPSIS
    1.170820...

    >>> klucbGauss(0.9, 0.4)  # doctest: +ELLIPSIS
    1.347213...
    >>> klucbGauss(0.9, 0.9)  # doctest: +ELLIPSIS
    1.570820...

    .. warning:: Using :class:`Policies.klUCB` (and variants) with :func:`klucbGauss` is equivalent to use :class:`Policies.UCB`, so prefer the simpler version.
    """
    return x + sqrt(abs(2 * sig2x * d))


@jit
def klucbPoisson(x, d, precision=1e-6):
    """ KL-UCB index computation for Poisson distributions, using :func:`klucb`.

    - Influence of x:

    >>> klucbPoisson(0.1, 0.2)  # doctest: +ELLIPSIS
    0.450523...
    >>> klucbPoisson(0.5, 0.2)  # doctest: +ELLIPSIS
    1.089376...
    >>> klucbPoisson(0.9, 0.2)  # doctest: +ELLIPSIS
    1.640112...

    - Influence of d:

    >>> klucbPoisson(0.1, 0.4)  # doctest: +ELLIPSIS
    0.693684...
    >>> klucbPoisson(0.1, 0.9)  # doctest: +ELLIPSIS
    1.252796...

    >>> klucbPoisson(0.5, 0.4)  # doctest: +ELLIPSIS
    1.422933...
    >>> klucbPoisson(0.5, 0.9)  # doctest: +ELLIPSIS
    2.122985...

    >>> klucbPoisson(0.9, 0.4)  # doctest: +ELLIPSIS
    2.033691...
    >>> klucbPoisson(0.9, 0.9)  # doctest: +ELLIPSIS
    2.831573...
    """
    upperbound = x + d + sqrt(d * d + 2 * x * d)  # looks safe, to check: left (Gaussian) tail of Poisson dev
    return klucb(x, d, klPoisson, upperbound, precision)


@jit
def klucbExp(x, d, precision=1e-6):
    """ KL-UCB index computation for exponential distributions, using :func:`klucb`.

    - Influence of x:

    >>> klucbExp(0.1, 0.2)  # doctest: +ELLIPSIS
    0.202741...
    >>> klucbExp(0.5, 0.2)  # doctest: +ELLIPSIS
    1.013706...
    >>> klucbExp(0.9, 0.2)  # doctest: +ELLIPSIS
    1.824671...

    - Influence of d:

    >>> klucbExp(0.1, 0.4)  # doctest: +ELLIPSIS
    0.285792...
    >>> klucbExp(0.1, 0.9)  # doctest: +ELLIPSIS
    0.559088...

    >>> klucbExp(0.5, 0.4)  # doctest: +ELLIPSIS
    1.428962...
    >>> klucbExp(0.5, 0.9)  # doctest: +ELLIPSIS
    2.795442...

    >>> klucbExp(0.9, 0.4)  # doctest: +ELLIPSIS
    2.572132...
    >>> klucbExp(0.9, 0.9)  # doctest: +ELLIPSIS
    5.031795...
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
    return klucb(x, d, klGamma, upperbound, precision, lowerbound)


# FIXME this one is wrong!
@jit
def klucbGamma(x, d, precision=1e-6):
    """ KL-UCB index computation for Gamma distributions, using :func:`klucb`.

    - Influence of x:

    >>> klucbGamma(0.1, 0.2)  # doctest: +ELLIPSIS
    0.202...
    >>> klucbGamma(0.5, 0.2)  # doctest: +ELLIPSIS
    1.013...
    >>> klucbGamma(0.9, 0.2)  # doctest: +ELLIPSIS
    1.824...

    - Influence of d:

    >>> klucbGamma(0.1, 0.4)  # doctest: +ELLIPSIS
    0.285...
    >>> klucbGamma(0.1, 0.9)  # doctest: +ELLIPSIS
    0.559...

    >>> klucbGamma(0.5, 0.4)  # doctest: +ELLIPSIS
    1.428...
    >>> klucbGamma(0.5, 0.9)  # doctest: +ELLIPSIS
    2.795...

    >>> klucbGamma(0.9, 0.4)  # doctest: +ELLIPSIS
    2.572...
    >>> klucbGamma(0.9, 0.9)  # doctest: +ELLIPSIS
    5.031...
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
    # FIXME specify the value for a !
    return klucb(x, d, klGamma, max(upperbound, 1e2), min(-1e2, lowerbound), precision)


# --- KL functions, for the KL Lower Confidence Bound

@jit
def kllcb(x, d, kl, lowerbound,
        precision=1e-6, upperbound=float('+inf'), max_iterations=50,
    ):
    r""" The generic KL-LCB index computation.

    - ``x``: value of the cum reward,
    - ``d``: lower bound on the divergence,
    - ``kl``: the KL divergence to be used (:func:`klBern`, :func:`klGauss`, etc),
    - ``lowerbound``, ``upperbound=float('-inf')``: the known bound of the values ``x``,
    - ``precision=1e-6``: the threshold from where to stop the research,
    - ``max_iterations=50``: max number of iterations of the loop (safer to bound it to reduce time complexity).

    .. math::

        \mathrm{kllcb}(x, d) \simeq \inf_{\mathrm{lowerbound} \leq y \leq \mathrm{upperbound}} \{ y : \mathrm{kl}(x, y) > d \}.

    .. note:: It uses a **bisection search**, and one call to ``kl`` for each step of the bisection search.

    For example, for :func:`kllcbBern`, the two steps are to first compute an upperbound (as precise as possible) and the compute the kl-UCB index:

    >>> x, d = 0.9, 0.2   # mean x, exploration term d
    >>> lowerbound = max(0., kllcbGauss(x, d, sig2x=0.25))  # variance 1/4 for [0,1] bounded distributions
    >>> lowerbound  # doctest: +ELLIPSIS
    0.5837...
    >>> kllcb(x, d, klBern, lowerbound, upperbound=0, precision=1e-3, max_iterations=10)  # doctest: +ELLIPSIS
    0.29...
    >>> kllcb(x, d, klBern, lowerbound, upperbound=0, precision=1e-6, max_iterations=10)  # doctest: +ELLIPSIS
    0.29188...
    >>> kllcb(x, d, klBern, lowerbound, upperbound=0, precision=1e-3, max_iterations=50)  # doctest: +ELLIPSIS
    0.291886...
    >>> kllcb(x, d, klBern, lowerbound, upperbound=0, precision=1e-6, max_iterations=100)  # more and more precise!  # doctest: +ELLIPSIS
    0.29188611...

    .. note:: See below for more examples for different KL divergence functions.
    """
    value = min(x, upperbound)
    l = lowerbound
    _count_iteration = 0
    while _count_iteration < max_iterations and value - l > precision:
        _count_iteration += 1
        m = (value + l) * 0.5
        if kl(x, m) > d:
            l = m
        else:
            value = m
    return (value + l) * 0.5


@jit
def kllcbBern(x, d, precision=1e-6):
    """ KL-LCB index computation for Bernoulli distributions, using :func:`kllcb`.

    - Influence of x:

    >>> kllcbBern(0.1, 0.2)  # doctest: +ELLIPSIS
    0.09999...
    >>> kllcbBern(0.5, 0.2)  # doctest: +ELLIPSIS
    0.49999...
    >>> kllcbBern(0.9, 0.2)  # doctest: +ELLIPSIS
    0.89999...

    - Influence of d:

    >>> kllcbBern(0.1, 0.4)  # doctest: +ELLIPSIS
    0.09999...
    >>> kllcbBern(0.1, 0.9)  # doctest: +ELLIPSIS
    0.09999...

    >>> kllcbBern(0.5, 0.4)  # doctest: +ELLIPSIS
    0.4999...
    >>> kllcbBern(0.5, 0.9)  # doctest: +ELLIPSIS
    0.4999...

    >>> kllcbBern(0.9, 0.4)  # doctest: +ELLIPSIS
    0.8999...
    >>> kllcbBern(0.9, 0.9)  # doctest: +ELLIPSIS
    0.8999...
    """
    lowerbound = max(0., kllcbGauss(x, d, sig2x=0.25, precision=precision))  # variance 1/4 for [0,1] bounded distributions
    # lowerbound = max(0., kllcbPoisson(x, d))  # also safe, and better ?
    return kllcb(x, d, klBern, lowerbound, precision)


@jit
def kllcbGauss(x, d, sig2x=0.25, precision=0.):
    """ KL-LCB index computation for Gaussian distributions.

    - Note that it does not require any search.

    .. warning:: it works only if the good variance constant is given.

    - Influence of x:

    >>> kllcbGauss(0.1, 0.2)  # doctest: +ELLIPSIS
    -0.21622...
    >>> kllcbGauss(0.5, 0.2)  # doctest: +ELLIPSIS
    0.18377...
    >>> kllcbGauss(0.9, 0.2)  # doctest: +ELLIPSIS
    0.58377...

    - Influence of d:

    >>> kllcbGauss(0.1, 0.4)  # doctest: +ELLIPSIS
    -0.3472...
    >>> kllcbGauss(0.1, 0.9)  # doctest: +ELLIPSIS
    -0.5708...

    >>> kllcbGauss(0.5, 0.4)  # doctest: +ELLIPSIS
    0.0527...
    >>> kllcbGauss(0.5, 0.9)  # doctest: +ELLIPSIS
    -0.1708...

    >>> kllcbGauss(0.9, 0.4)  # doctest: +ELLIPSIS
    0.4527...
    >>> kllcbGauss(0.9, 0.9)  # doctest: +ELLIPSIS
    0.2291...

    .. warning:: Using :class:`Policies.kllCB` (and variants) with :func:`kllcbGauss` is equivalent to use :class:`Policies.UCB`, so prefer the simpler version.
    """
    return x - sqrt(abs(2 * sig2x * d))


@jit
def kllcbPoisson(x, d, precision=1e-6):
    """ KL-LCB index computation for Poisson distributions, using :func:`kllcb`.

    - Influence of x:

    >>> kllcbPoisson(0.1, 0.2)  # doctest: +ELLIPSIS
    0.09999...
    >>> kllcbPoisson(0.5, 0.2)  # doctest: +ELLIPSIS
    0.49999...
    >>> kllcbPoisson(0.9, 0.2)  # doctest: +ELLIPSIS
    0.89999...

    - Influence of d:

    >>> kllcbPoisson(0.1, 0.4)  # doctest: +ELLIPSIS
    0.09999...
    >>> kllcbPoisson(0.1, 0.9)  # doctest: +ELLIPSIS
    0.09999...

    >>> kllcbPoisson(0.5, 0.4)  # doctest: +ELLIPSIS
    0.49999...
    >>> kllcbPoisson(0.5, 0.9)  # doctest: +ELLIPSIS
    0.49999...

    >>> kllcbPoisson(0.9, 0.4)  # doctest: +ELLIPSIS
    0.89999...
    >>> kllcbPoisson(0.9, 0.9)  # doctest: +ELLIPSIS
    0.89999...
    """
    lowerbound = x + d - sqrt(d * d + 2 * x * d)  # looks safe, to check: left (Gaussian) tail of Poisson dev
    return kllcb(x, d, klPoisson, lowerbound, precision)


@jit
def kllcbExp(x, d, precision=1e-6):
    """ KL-LCB index computation for exponential distributions, using :func:`kllcb`.

    - Influence of x:

    >>> kllcbExp(0.1, 0.2)  # doctest: +ELLIPSIS
    0.15267...
    >>> kllcbExp(0.5, 0.2)  # doctest: +ELLIPSIS
    0.7633...
    >>> kllcbExp(0.9, 0.2)  # doctest: +ELLIPSIS
    1.3740...

    - Influence of d:

    >>> kllcbExp(0.1, 0.4)  # doctest: +ELLIPSIS
    0.2000...
    >>> kllcbExp(0.1, 0.9)  # doctest: +ELLIPSIS
    0.3842...

    >>> kllcbExp(0.5, 0.4)  # doctest: +ELLIPSIS
    1.0000...
    >>> kllcbExp(0.5, 0.9)  # doctest: +ELLIPSIS
    1.9214...

    >>> kllcbExp(0.9, 0.4)  # doctest: +ELLIPSIS
    1.8000...
    >>> kllcbExp(0.9, 0.9)  # doctest: +ELLIPSIS
    3.4586...
    """
    if d < 0.77:  # XXX where does this value come from?
        lowerbound = x / (1 + 2. / 3 * d - sqrt(4. / 9 * d * d + 2 * d))
        # safe, klexp(x,y) >= e^2/(2*(1-2e/3)) if x=y(1-e)
    else:
        lowerbound = x * exp(d + 1)
    if d > 1.61:  # XXX where does this value come from?
        upperbound = x * exp(d)
    else:
        upperbound = x / (1 + d - sqrt(d * d + 2 * d))
    return kllcb(x, d, klGamma, lowerbound, precision, upperbound)


# # FIXME this one is wrong!
# @jit
# def kllcbGamma(x, d, precision=1e-6):
#     """ KL-LCB index computation for Gamma distributions, using :func:`kllcb`.

#     - Influence of x:

#     >>> kllcbGamma(0.1, 0.2)  # doctest: +ELLIPSIS
#     0.202...
#     >>> kllcbGamma(0.5, 0.2)  # doctest: +ELLIPSIS
#     1.013...
#     >>> kllcbGamma(0.9, 0.2)  # doctest: +ELLIPSIS
#     1.824...

#     - Influence of d:

#     >>> kllcbGamma(0.1, 0.4)  # doctest: +ELLIPSIS
#     0.285...
#     >>> kllcbGamma(0.1, 0.9)  # doctest: +ELLIPSIS
#     0.559...

#     >>> kllcbGamma(0.5, 0.4)  # doctest: +ELLIPSIS
#     1.428...
#     >>> kllcbGamma(0.5, 0.9)  # doctest: +ELLIPSIS
#     2.795...

#     >>> kllcbGamma(0.9, 0.4)  # doctest: +ELLIPSIS
#     2.572...
#     >>> kllcbGamma(0.9, 0.9)  # doctest: +ELLIPSIS
#     5.031...
#     """
#     if d < 0.77:  # XXX where does this value come from?
#         lowerbound = x / (1 + 2. / 3 * d - sqrt(4. / 9 * d * d + 2 * d))
#         # safe, klexp(x,y) >= e^2/(2*(1-2e/3)) if x=y(1-e)
#     else:
#         lowerbound = x * exp(d + 1)
#     if d > 1.61:  # XXX where does this value come from?
#         upperbound = x * exp(d)
#     else:
#         upperbound = x / (1 + d - sqrt(d * d + 2 * d))
#     # FIXME specify the value for a !
#     return kllcb(x, d, klGamma, min(lowerbound, -1e2), precision, max(1e2, upperbound))


# --- max EV functions

#@jit
def maxEV(p, V, klMax):
    r""" Maximize expectation of :math:`V` with respect to :math:`q` st. :math:`\mathrm{KL}(p, q) < \text{klMax}`.

    - Input args.: p, V, klMax.
    - Reference: Section 3.2 of [Filippi, Cappé & Garivier - Allerton, 2011](https://arxiv.org/pdf/1004.5229.pdf).
    """
    Uq = np.zeros(len(p))
    Kb = p > 0.
    K = ~Kb
    if any(K):
        # Do we need to put some mass on a point where p is zero?
        # If yes, this has to be on one which maximizes V.
        eta = np.max(V[K])
        J = K & (V == eta)
        if eta > np.max(V[Kb]):
            y = np.dot(p[Kb], np.log(eta - V[Kb])) + log(np.dot(p[Kb], (1. / (eta - V[Kb]))))
            # print("eta = ", eta, ", y = ", y)
            if y < klMax:
                rb = exp(y - klMax)
                Uqtemp = p[Kb] / (eta - V[Kb])
                Uq[Kb] = rb * Uqtemp / np.sum(Uqtemp)
                Uq[J] = (1. - rb) / np.sum(J)
                # or j = min([j for j in range(k) if J[j]])
                # Uq[j] = r
                return Uq
    # Here, only points where p is strictly positive (in Kb) will receive non-zero mass.
    if any(np.abs(V[Kb] - V[Kb][0]) > 1e-8):
        eta = reseqp(p[Kb], V[Kb], klMax)  # (eta = nu in the article)
        Uq = p / (eta - V)
        Uq = Uq / np.sum(Uq)
    else:
        # Case where all values in V(Kb) are almost identical.
        Uq[Kb] = 1.0 / len(Kb)
    return Uq


#@jit
def reseqp(p, V, klMax, max_iterations=50):
    """ Solve ``f(reseqp(p, V, klMax)) = klMax``, using Newton method.

    .. note:: This is a subroutine of :func:`maxEV`.

    - Reference: Eq. (4) in Section 3.2 of [Filippi, Cappé & Garivier - Allerton, 2011](https://arxiv.org/pdf/1004.5229.pdf).

    .. warning:: `np.dot` is very slow!
    """
    MV = np.max(V)
    mV = np.min(V)
    value = MV + 0.1
    tol = 1e-4
    if MV < mV + tol:
        return float('inf')
    u = np.dot(p, (1 / (value - V)))
    y = np.dot(p, np.log(value - V)) + log(u) - klMax
    print("value =", value, ", y = ", y)  # DEBUG
    _count_iteration = 0
    while _count_iteration < max_iterations and np.abs(y) > tol:
        _count_iteration += 1
        yp = u - np.dot(p, (1 / (value - V)**2)) / u  # derivative
        value -= y / yp
        print("value = ", value)  # DEBUG  # newton iteration
        if value < MV:
            value = (value + y / yp + MV) / 2  # unlikely, but not impossible
        u = np.dot(p, (1 / (value - V)))
        y = np.dot(p, np.log(value - V)) + np.log(u) - klMax
        print("value = ", value, ", y = ", y)  # DEBUG  # function
    return value


def reseqp2(p, V, klMax):
    """ Solve f(reseqp(p, V, klMax)) = klMax, using a blackbox minimizer, from scipy.optimize.

    - FIXME it does not work well yet!

    .. note:: This is a subroutine of :func:`maxEV`.

    - Reference: Eq. (4) in Section 3.2 of [Filippi, Cappé & Garivier - Allerton, 2011].

    .. warning:: `np.dot` is very slow!
    """
    MV = np.max(V)
    mV = np.min(V)
    tol = 1e-4
    value0 = mV + 0.1

    #@jit  # TODO try numba.jit() on this function
    def f(value):
        """ Function fo to minimize."""
        if MV < mV + tol:
            y = float('inf')
        else:
            u = np.dot(p, (1 / (value - V)))
            y = np.dot(p, np.log(value - V)) + log(u)
        return np.abs(y - klMax)

    res = optimize.minimize(f, value0)
    print("scipy.optimize.minimize returned", res)
    return res.x if hasattr(res, 'x') else res


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
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

    p = np.array([0.5, 0.5])
    print("\np =", p)
    V = np.array([10, 3])
    print("V =", V)
    klMax = 0.1
    print("klMax =", klMax)
    print("eta = ", reseqp(p, V, klMax))
    # print("eta 2 = ", reseqp2(p, V, klMax))
    print("Uq = ", maxEV(p, V, klMax))

    print("\np =", p)
    p = np.array([0.11794872, 0.27948718, 0.31538462, 0.14102564, 0.0974359, 0.03076923, 0.00769231, 0.01025641, 0.])
    print("V =", V)
    V = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])
    klMax = 0.0168913409484
    print("klMax =", klMax)
    print("eta = ", reseqp(p, V, klMax))
    # print("eta 2 = ", reseqp2(p, V, klMax))
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
