# -*- coding: utf-8 -*-
""" Define some function to measure fairness of a vector of cumulated rewards, of shape (nbPlayers, horizon).

- All functions are valued in [0, 1]: 100% means fully unfair (one player has 0 rewards, another one has >0 rewards), and 0% means fully fair (they all have exactly the same rewards).
- Reference: https://en.wikipedia.org/wiki/Fairness_measure.
"""
from __future__ import division, print_function

__author__ = "Lilian Besson"
__version__ = "0.5"


import numpy as np
import numpy.random as rn


def amplitude_fairness(X, axis=0):
    """ (Normalized) Amplitude fairness, homemade formula: 1 - min(X, axis) / max(X, axis).

    >>> rn.seed(1)
    >>> X = np.cumsum(rn.rand(10, 1000))
    >>> amplitude_fairness(X)  # doctest: +ELLIPSIS
    0.999...
    >>> amplitude_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.999...
    >>> amplitude_fairness(np.log(1 + np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.959...

    >>> rn.seed(3)
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
    return 1 - (np.min(X, axis=axis) / np.max(X, axis=axis))



def std_fairness(X, axis=0):
    """ (Normalized) Standard-variation fairness, homemade formula: 2 * std(X, axis) / max(X, axis).

    >>> rn.seed(1)
    >>> X = np.cumsum(rn.rand(10, 1000))
    >>> std_fairness(X)  # doctest: +ELLIPSIS
    0.575...
    >>> std_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.594...
    >>> std_fairness(np.sqrt(np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.470...

    >>> rn.seed(2)
    >>> X = np.cumsum(rn.randint(0, 10, (10, 100)))
    >>> std_fairness(X)  # doctest: +ELLIPSIS
    0.570...
    >>> std_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.587...
    >>> std_fairness(np.sqrt(np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.463...
    """
    return 2 * np.std(X, axis=axis) / np.max(X, axis=axis)


def rajjain_fairness(X, axis=0):
    """ Raj Jain's fairness index: (sum x)**2 / (N * sum x**2), projected to [0, 1].

    - cf. https://en.wikipedia.org/wiki/Fairness_measure#Jain.27s_fairness_index.

    >>> rn.seed(1)
    >>> X = np.cumsum(rn.rand(10, 1000))
    >>> rajjain_fairness(X)  # doctest: +ELLIPSIS
    0.248...
    >>> rajjain_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.441...
    >>> rajjain_fairness(np.sqrt(np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.110...

    >>> rn.seed(2)
    >>> X = np.cumsum(rn.randint(0, 10, (10, 100)))
    >>> rajjain_fairness(X)  # doctest: +ELLIPSIS
    0.246...
    >>> rajjain_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.917...
    >>> rajjain_fairness(np.sqrt(np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.107...
    """
    n = X.shape[axis]
    if n <= 1:
        return 0
    else:
        return (n - (np.sum(X, axis=axis) ** 2) / (np.sum(X ** 2, axis=axis))) / (n - 1)


def mean_fairness(X, axis=0, methods=(amplitude_fairness, std_fairness, rajjain_fairness)):
    """A certain measure of sortedness for the list A, based on mean of the 4 distances: manhattan, kendalltau, spearmanr, gestalt.

    >>> rn.seed(1)
    >>> X = np.cumsum(rn.rand(10, 1000))
    >>> mean_fairness(X)  # doctest: +ELLIPSIS
    0.607...
    >>> mean_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.678...
    >>> mean_fairness(np.sqrt(np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.523...

    >>> rn.seed(2)
    >>> X = np.cumsum(rn.randint(0, 10, (10, 100)))
    >>> mean_fairness(X)  # doctest: +ELLIPSIS
    0.605...
    >>> mean_fairness(X ** 2)  # More spreadout  # doctest: +ELLIPSIS
    0.834...
    >>> mean_fairness(np.sqrt(np.abs(X)))  # Less spreadout  # doctest: +ELLIPSIS
    0.509...
    """
    fairnesses = []
    for method in methods:
        fairnesses.append(method(X, axis=axis))
    return np.mean(fairnesses)


# Default fairness measure
fairnessMeasure = mean_fairness

fairness_mapping = {
    "amplitude": amplitude_fairness,
    "std": std_fairness,
    "rajjain": rajjain_fairness,
    "mean": mean_fairness,
    "default": fairnessMeasure
}


# Only export and expose the useful functions defined here
__all__ = [
    amplitude_fairness,
    std_fairness,
    rajjain_fairness,
    mean_fairness,
    fairnessMeasure,
    fairness_mapping
]


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
