# -*- coding: utf-8 -*-
""" Kullback-Leibler divergence functions and klUCB utilities. Optimized version that should be compiled using Cython (http://docs.cython.org/).

- Faster implementation can be found in a C file, in ``Policies/C``, and should be compiled to speedup computations.
- However, the version here have examples, doctests.
- Cf. https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
- Reference: [Filippi, Cappé & Garivier - Allerton, 2011](https://arxiv.org/pdf/1004.5229.pdf) and [Garivier & Cappé, 2011](https://arxiv.org/pdf/1102.2490.pdf)


.. note::

    The C version is still faster, but not so much. The Cython version has the advantage of providing docstrings, and optional arguments, and is only 2 to 3 times slower than the optimal performance!
    Here are some comparisons of the two, when computing KL-UCB indexes:

    >>> r = np.random.random
    >>> import kullback_c; import kullback_cython  # fake, just for illustration

    >>> %timeit kullback_c.klucbBern(r(), r(), 1e-6)
    2.26 µs ± 21.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    >>> %timeit kullback_cython.klucbBern(r(), r())
    6.04 µs ± 46.7 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    >>> %timeit kullback_c.klucbGamma(r(), r(), 1e-6)
    2.56 µs ± 225 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    >>> %timeit kullback_cython.klucbGamma(r(), r())
    6.47 µs ± 513 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    >>> %timeit kullback_c.klucbExp(r(), r(), 1e-6)
    2.03 µs ± 92.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    >>> %timeit kullback_cython.klucbExp(r(), r())
    4.75 µs ± 444 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    >>> %timeit kullback_c.klucbGauss(r(), r(), 1e-6)  # so easy computation: no difference!
    800 ns ± 14.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    >>> %timeit kullback_cython.klucbGauss(r(), r())
    732 ns ± 71.5 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    >>> %timeit kullback_c.klucbPoisson(r(), r(), 1e-6)
    2.33 µs ± 167 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    >>> %timeit kullback_cython.klucbPoisson(r(), r())
    5.13 µs ± 521 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)


.. warning::

    This extension should be used with the ``setup.py`` script, by running::

        $ python setup.py build_ext --inplace

    You can also use [pyximport](http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html#pyximport-cython-compilation-for-developers) to import the ``kullback_cython`` module transparently:

    >>> import pyximport; pyximport.install()  # instantaneous  # doctest: +ELLIPSIS
    (None, <pyximport.pyximport.PyxImporter at 0x...>)
    >>> import kullback_cython as kullback     # takes about two seconds
    >>> # then use kullback.klucbBern or others, as if they came from the pure Python version!


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
    >>> def klGauss_vect(xs, y, sig2x=0.25) -> float:  # vectorized for first input only
    ...    return np.array([klGauss(x, y, sig2x) for x in xs])
    >>> klGauss_vect([-1, 0, 1], 0.1)  # doctest: +ELLIPSIS
    array([2.42, 0.02, 1.62])
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.9"

from libc.math cimport log, sqrt, exp


cdef float eps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]


# --- Simple Kullback-Leibler divergence for known distributions


def klBern(float x, float y) -> float:
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


def klBin(float x, float y, int n) -> float:
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


def klPoisson(float x, float y) -> float:
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


def klExp(float x, float y) -> float:
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
        return float('+inf')
    else:
        x = max(x, eps)
        y = max(y, eps)
        return x / y - 1 - log(x / y)


def klGamma(float x, float y, float a=1) -> float:
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
        return float('+inf')
    else:
        x = max(x, eps)
        y = max(y, eps)
        return a * (x / y - 1 - log(x / y))


def klNegBin(float x, float y, r=1) -> float:
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


def klGauss(float x, float y, float sig2x=0.25, float sig2y=0.25) -> float:
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
    if - eps < (sig2y - sig2x) < eps:
        return (x - y) ** 2 / (2. * sig2x)
    else:
        return (x - y) ** 2 / (2. * sig2y) + 0.5 * ((sig2x/sig2y)**2 - 1 - log(sig2x/sig2y))


# --- KL functions, for the KL-UCB policy

def klucb(float x, float d, kl,
        float upperbound,
        float lowerbound=float('-inf'),
        float precision=1e-6,
        int max_iterations=50
    ) -> float:
    """ The generic KL-UCB index computation.

    - x: value of the cum reward,
    - d: upper bound on the divergence,
    - kl: the KL divergence to be used (:func:`klBern`, :func:`klGauss`, etc),
    - upperbound, lowerbound=float('-inf') -> float: the known bound of the values x,
    - precision=1e-6: the threshold from where to stop the research,
    - max_iterations: max number of iterations of the loop (safer to bound it to reduce time complexity).

    .. note:: It uses a **bisection search**, and one call to ``kl`` for each step of the bisection search.

    For example, for :func:`klucbBern`, the two steps are to first compute an upperbound (as precise as possible) and the compute the kl-UCB index:

    >>> x, d = 0.9, 0.2   # mean x, exploration term d
    >>> upperbound = min(1., klucbGauss(x, d, sig2x=0.25))  # variance 1/4 for [0,1] bounded distributions
    >>> upperbound  # doctest: +ELLIPSIS
    1.0
    >>> klucb(x, d, klBern, upperbound, lowerbound=0, precision=1e-3, max_iterations=10)  # doctest: +ELLIPSIS
    0.9941...
    >>> klucb(x, d, klBern, upperbound, lowerbound=0, precision=1e-6, max_iterations=10)  # doctest: +ELLIPSIS
    0.994482...  # doctest: +ELLIPSIS
    >>> klucb(x, d, klBern, upperbound, lowerbound=0, precision=1e-3, max_iterations=50)  # doctest: +ELLIPSIS
    0.9941...
    >>> klucb(x, d, klBern, upperbound, lowerbound=0, precision=1e-6, max_iterations=100)  # more and more precise!  # doctest: +ELLIPSIS
    0.994489...

    .. note:: See below for more examples for different KL divergence functions.
    """
    cdef float value = max(x, lowerbound)
    cdef float u = upperbound
    cdef int _count_iteration = 0
    while _count_iteration < max_iterations and u - value > precision:
        _count_iteration += 1
        m = (value + u) / 2.
        if kl(x, m) > d:
            u = m
        else:
            value = m
    return (value + u) / 2.


def klucbBern(float x, float d, float precision=1e-6) -> float:
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
    cdef float upperbound = min(1., klucbGauss(x, d, sig2x=0.25))  # variance 1/4 for [0,1] bounded distributions
    # upperbound = min(1., klucbPoisson(x, d))  # also safe, and better ?
    return klucb(x, d, klBern, upperbound, precision)


def klucbGauss(float x, float d, float sig2x=0.25, float precision=0.) -> float:
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
    return x + sqrt(2 * sig2x * d)


def klucbPoisson(float x, float d, float precision=1e-6) -> float:
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
    cdef float upperbound = x + d + sqrt(d * d + 2 * x * d)  # looks safe, to check: left (Gaussian) tail of Poisson dev
    return klucb(x, d, klPoisson, upperbound, precision)


def klucbExp(float x, float d, float precision=1e-6) -> float:
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
    cdef float upperbound
    cdef float lowerbound
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


# FIXME this one is wrong!
def klucbGamma(float x, float d, float precision=1e-6) -> float:
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
    cdef float upperbound
    cdef float lowerbound
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
