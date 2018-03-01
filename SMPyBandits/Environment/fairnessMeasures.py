# -*- coding: utf-8 -*-
r""" Define some function to measure fairness of a vector of cumulated rewards, of shape `(nbPlayers, horizon)`.

- All functions are valued in :math:`[0, 1]`: :math:`100\%` means fully unfair (one player has :math:`0` rewards, another one has :math:`>0` rewards), and :math:`0\%` means fully fair (they all have exactly the same rewards).
- Reference: https://en.wikipedia.org/wiki/Fairness_measure and http://ica1www.epfl.ch/PS_files/LEB3132.pdf#search=%22max-min%20fairness%22.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.5"


import numpy as np


def amplitude_fairness(X, axis=0):
    r""" (Normalized) Amplitude fairness, homemade formula: :math:`1 - \min(X, axis) / \max(X, axis)`.

    Examples:

    >>> import numpy.random as rn; rn.seed(1)  # for reproductibility
    >>> X = np.cumsum(rn.rand(10, 1000))
    >>> amplitude_fairness(X)  # doctest: +ELLIPSIS
    0.999...
    >>> amplitude_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.999...
    >>> amplitude_fairness(np.log(1 + np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.959...

    >>> rn.seed(3)  # for reproductibility
    >>> X = rn.randint(0, 10, (10, 1000)); Y = np.cumsum(X, axis=1)
    >>> np.min(Y, axis=0)[0], np.max(Y, axis=0)[0]
    (3, 9)
    >>> np.min(Y, axis=0)[-1], np.max(Y, axis=0)[-1]
    (4387, 4601)
    >>> amplitude_fairness(Y, axis=0).shape
    (1000,)
    >>> list(amplitude_fairness(Y, axis=0))  # doctest: +ELLIPSIS
    [0.666..., 0.764..., ..., 0.0465...]

    >>> X[X >= 3] = 3; Y = np.cumsum(X, axis=1)
    >>> np.min(Y, axis=0)[0], np.max(Y, axis=0)[0]
    (3, 3)
    >>> np.min(Y, axis=0)[-1], np.max(Y, axis=0)[-1]
    (2353, 2433)
    >>> amplitude_fairness(Y, axis=0).shape
    (1000,)
    >>> list(amplitude_fairness(Y, axis=0))  # Less spreadout # doctest: +ELLIPSIS
    [0.0, 0.5, ..., 0.0328...]
    """
    X = np.asarray(X)
    return 1 - (np.min(X, axis=axis) / np.max(X, axis=axis))


def std_fairness(X, axis=0):
    r""" (Normalized) Standard-variation fairness, homemade formula: :math:`2 * \mathrm{std}(X, axis) / \max(X, axis)`.

    Examples:

    >>> import numpy.random as rn; rn.seed(1)  # for reproductibility
    >>> X = np.cumsum(rn.rand(10, 1000))
    >>> std_fairness(X)  # doctest: +ELLIPSIS
    0.575...
    >>> std_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.594...
    >>> std_fairness(np.sqrt(np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.470...

    >>> rn.seed(2)  # for reproductibility
    >>> X = np.cumsum(rn.randint(0, 10, (10, 100)))
    >>> std_fairness(X)  # doctest: +ELLIPSIS
    0.570...
    >>> std_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.587...
    >>> std_fairness(np.sqrt(np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.463...
    """
    X = np.asarray(X)
    return 2 * np.std(X, axis=axis) / np.max(X, axis=axis)


def rajjain_fairness(X, axis=0):
    r""" Raj Jain's fairness index: :math:`(\sum_{i=1}^{n} x_i)^2 / (n \times \sum_{i=1}^{n} x_i^2)`, projected to :math:`[0, 1]` instead of :math:`[\frac{1}{n}, 1]` as introduced in the reference article.

    - cf. https://en.wikipedia.org/wiki/Fairness_measure#Jain.27s_fairness_index.

    Examples:

    >>> import numpy.random as rn; rn.seed(1)  # for reproductibility
    >>> X = np.cumsum(rn.rand(10, 1000))
    >>> rajjain_fairness(X)  # doctest: +ELLIPSIS
    0.248...
    >>> rajjain_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.441...
    >>> rajjain_fairness(np.sqrt(np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.110...

    >>> rn.seed(2)  # for reproductibility
    >>> X = np.cumsum(rn.randint(0, 10, (10, 100)))
    >>> rajjain_fairness(X)  # doctest: +ELLIPSIS
    0.246...
    >>> rajjain_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.917...
    >>> rajjain_fairness(np.sqrt(np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.107...
    """
    X = np.asarray(X)
    n = X.shape[axis]
    if n <= 1:
        return 0
    else:
        return (n - (np.sum(X, axis=axis) ** 2) / (np.sum(X ** 2, axis=axis))) / (n - 1)


def mo_walrand_fairness(X, axis=0, alpha=2):
    r""" Mo and Walrand's family fairness index: :math:`U_{\alpha}(X)`, **NOT** projected to :math:`[0, 1]`.

    .. math::

       U_{\alpha}(X) = \begin{cases}
       \frac{1}{1 - \alpha} \sum_{i=1}^n x_i^{1 - \alpha} & \;\text{if}\; \alpha\in[0,+\infty)\setminus\{1\}, \\
       \sum_{i=1}^{n} \ln(x_i) & \;\text{otherwise}.
       \end{cases}

    Examples:

    >>> import numpy.random as rn; rn.seed(1)  # for reproductibility
    >>> X = np.cumsum(rn.rand(10, 1000))

    >>> alpha = 0
    >>> mo_walrand_fairness(X, alpha=alpha)  # doctest: +ELLIPSIS
    24972857.013...
    >>> mo_walrand_fairness(X ** 2, alpha=alpha)  # More spreadout  # doctest: +ELLIPSIS
    82933940429.039...
    >>> mo_walrand_fairness(np.sqrt(np.abs(X)), alpha=alpha)  # Less spreadout  # doctest: +ELLIPSIS
    471371.219...

    >>> alpha = 0.99999
    >>> mo_walrand_fairness(X, alpha=alpha)  # doctest: +ELLIPSIS
    1000075176.390...
    >>> mo_walrand_fairness(X ** 2, alpha=alpha)  # More spreadout  # doctest: +ELLIPSIS
    1000150358.528...
    >>> mo_walrand_fairness(np.sqrt(np.abs(X)), alpha=alpha)  # Less spreadout  # doctest: +ELLIPSIS
    1000037587.478...

    >>> alpha = 1
    >>> mo_walrand_fairness(X, alpha=alpha)  # doctest: +ELLIPSIS
    75173.509...
    >>> mo_walrand_fairness(X ** 2, alpha=alpha)  # More spreadout  # doctest: +ELLIPSIS
    150347.019...
    >>> mo_walrand_fairness(np.sqrt(np.abs(X)), alpha=alpha)  # Less spreadout  # doctest: +ELLIPSIS
    37586.754...

    >>> alpha = 1.00001
    >>> mo_walrand_fairness(X, alpha=alpha)  # doctest: +ELLIPSIS
    -999924829.359...
    >>> mo_walrand_fairness(X ** 2, alpha=alpha)  # More spreadout  # doctest: +ELLIPSIS
    -999849664.476...
    >>> mo_walrand_fairness(np.sqrt(np.abs(X)), alpha=alpha)  # Less spreadout  # doctest: +ELLIPSIS
    -999962413.957...

    >>> alpha = 2
    >>> mo_walrand_fairness(X, alpha=alpha)  # doctest: +ELLIPSIS
    -22.346...
    >>> mo_walrand_fairness(X ** 2, alpha=alpha)  # More spreadout  # doctest: +ELLIPSIS
    -9.874...
    >>> mo_walrand_fairness(np.sqrt(np.abs(X)), alpha=alpha)  # Less spreadout  # doctest: +ELLIPSIS
    -283.255...

    >>> alpha = 5
    >>> mo_walrand_fairness(X, alpha=alpha)  # doctest: +ELLIPSIS
    -8.737...
    >>> mo_walrand_fairness(X ** 2, alpha=alpha)  # More spreadout  # doctest: +ELLIPSIS
    -273.522...
    >>> mo_walrand_fairness(np.sqrt(np.abs(X)), alpha=alpha)  # Less spreadout  # doctest: +ELLIPSIS
    -2.468...
    """
    X = np.asarray(X)
    n = X.shape[axis]
    if n <= 1:
        return 0
    else:
        assert alpha >= 0, "Error: the parameter 'alpha' for mo_walrand_fairness() function has to be >= 0, but it was {} instead.".format(alpha)  # DEBUG
        if alpha == 1:
            return np.sum(np.log(X))
        else:
            oneMalpha = 1. - alpha
            return (1. / oneMalpha) * np.sum(X ** oneMalpha)


def mean_fairness(X, axis=0, methods=(amplitude_fairness, std_fairness, rajjain_fairness)):
    """ Fairness index, based on mean of the 3 fairness measures: Amplitude, STD and Raj Jain fairness.

    Examples:

    >>> import numpy.random as rn; rn.seed(1)  # for reproductibility
    >>> X = np.cumsum(rn.rand(10, 1000))
    >>> mean_fairness(X)  # doctest: +ELLIPSIS
    0.607...
    >>> mean_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.678...
    >>> mean_fairness(np.sqrt(np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.523...

    >>> rn.seed(2)  # for reproductibility
    >>> X = np.cumsum(rn.randint(0, 10, (10, 100)))
    >>> mean_fairness(X)  # doctest: +ELLIPSIS
    0.605...
    >>> mean_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.834...
    >>> mean_fairness(np.sqrt(np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.509...
    """
    X = np.asarray(X)
    fairnesses = np.array([m(X, axis=axis) for m in methods])
    return np.mean(fairnesses, axis=0)


#: Default fairness measure
fairnessMeasure = mean_fairness

#: Mapping of names of measure to their function
fairness_mapping = {
    # "amplitude_fairness":   amplitude_fairness,
    # "std_fairness":         std_fairness,
    # "rajjain_fairness":     rajjain_fairness,
    # "mean_fairness":        mean_fairness,
    # "fairnessMeasure":      fairnessMeasure,
    # "amplitude":            amplitude_fairness,
    # "std":                  std_fairness,
    # "rajjain":              rajjain_fairness,
    # "mean":                 mean_fairness,
    # "default":              fairnessMeasure,
    "Amplitude":            amplitude_fairness,
    "STD":                  std_fairness,
    "RajJain":              rajjain_fairness,
    "MoWalrand":            mo_walrand_fairness,
    "Mean":                 mean_fairness,
    "Default":              fairnessMeasure,
}


# Only export and expose the useful functions defined here
__all__ = [
    "amplitude_fairness",
    "std_fairness",
    "rajjain_fairness",
    "mo_walrand_fairness",
    "mean_fairness",
    "fairnessMeasure",
    "fairness_mapping",
]


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
