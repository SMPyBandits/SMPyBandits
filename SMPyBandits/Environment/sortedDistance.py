# -*- coding: utf-8 -*-
""" sortedDistance: define function to measure of sortedness of permutations of [0..N-1].

- Cf. http://stevehanov.ca/blog/index.php?id=145 and https://stackoverflow.com/q/8206617
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

from difflib import SequenceMatcher
import numpy as np
import scipy.stats


def weightedDistance(choices, weights, n=None):
    """Relative difference between the best possible weighted choices and the actual choices.

    >>> weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    >>> choices = [8, 6, 5, 2]
    >>> weightedDistance(choices, weights)  # not a bad choice  # doctest: +ELLIPSIS
    0.8333...
    >>> choices = [8, 6, 5, 7]
    >>> weightedDistance(choices, weights)  # best choice!  # doctest: +ELLIPSIS
    1.000...
    >>> choices = [3, 2, 1, 0]
    >>> weightedDistance(choices, weights)  # worst choice!  # doctest: +ELLIPSIS
    0.3333...
    """
    choices = np.asarray(choices)
    weights = np.asarray(weights)
    if n is None:
        n = len(choices)
    bestWeights = np.sum(np.sort(weights)[-n:])
    chosenWeights = np.sum(weights[choices[-n:]])
    return chosenWeights / float(bestWeights)


def manhattan(permutation, comp=None):
    """A certain measure of sortedness for the list A, based on Manhattan distance.

    >>> perm = [0, 1, 2, 3, 4]
    >>> manhattan(perm)  # sorted  # doctest: +ELLIPSIS
    1.0...

    >>> perm = [0, 1, 2, 5, 4, 3]
    >>> manhattan(perm)  # almost sorted!  # doctest: +ELLIPSIS
    0.777...

    >>> perm = [2, 9, 6, 4, 0, 3, 1, 7, 8, 5]  # doctest: +ELLIPSIS
    >>> manhattan(perm)
    0.4

    >>> perm = [2, 1, 6, 4, 0, 3, 5, 7, 8, 9]  # better sorted!  # doctest: +ELLIPSIS
    >>> manhattan(perm)
    0.72
    """
    if comp is None:
        comp = sorted(permutation)
    return 1 - (2 * sum(abs(comp[index] - element) for index, element in enumerate(permutation))) / (len(permutation) ** 2)


def kendalltau(permutation, comp=None):
    """A certain measure of sortedness for the list A, based on Kendall Tau ranking coefficient.

    >>> perm = [0, 1, 2, 3, 4]
    >>> kendalltau(perm)  # sorted  # doctest: +ELLIPSIS
    0.98...

    >>> perm = [0, 1, 2, 5, 4, 3]
    >>> kendalltau(perm)  # almost sorted!  # doctest: +ELLIPSIS
    0.90...

    >>> perm = [2, 9, 6, 4, 0, 3, 1, 7, 8, 5]
    >>> kendalltau(perm)  # doctest: +ELLIPSIS
    0.211...

    >>> perm = [2, 1, 6, 4, 0, 3, 5, 7, 8, 9]  # better sorted!
    >>> kendalltau(perm)  # doctest: +ELLIPSIS
    0.984...
    """
    if comp is None:
        comp = sorted(permutation)
    res = 1 - scipy.stats.kendalltau(permutation, comp).pvalue
    if np.isnan(res):
        res = 0
    return res


def spearmanr(permutation, comp=None):
    """A certain measure of sortedness for the list A, based on Spearman ranking coefficient.

    >>> perm = [0, 1, 2, 3, 4]
    >>> spearmanr(perm)  # sorted  # doctest: +ELLIPSIS
    1.0...

    >>> perm = [0, 1, 2, 5, 4, 3]
    >>> spearmanr(perm)  # almost sorted!  # doctest: +ELLIPSIS
    0.92...

    >>> perm = [2, 9, 6, 4, 0, 3, 1, 7, 8, 5]
    >>> spearmanr(perm)  # doctest: +ELLIPSIS
    0.248...

    >>> perm = [2, 1, 6, 4, 0, 3, 5, 7, 8, 9]  # better sorted!
    >>> spearmanr(perm)  # doctest: +ELLIPSIS
    0.986...
    """
    if comp is None:
        comp = sorted(permutation)
    res = 1 - scipy.stats.spearmanr(permutation, comp).pvalue
    if np.isnan(res):
        res = 0
    return res


def gestalt(permutation, comp=None):
    """A certain measure of sortedness for the list A, based on Gestalt pattern matching.

    >>> perm = [0, 1, 2, 3, 4]
    >>> gestalt(perm)  # sorted  # doctest: +ELLIPSIS
    1.0...

    >>> perm = [0, 1, 2, 5, 4, 3]
    >>> gestalt(perm)  # almost sorted!  # doctest: +ELLIPSIS
    0.666...

    >>> perm = [2, 9, 6, 4, 0, 3, 1, 7, 8, 5]
    >>> gestalt(perm)  # doctest: +ELLIPSIS
    0.4...

    >>> perm = [2, 1, 6, 4, 0, 3, 5, 7, 8, 9]  # better sorted!
    >>> gestalt(perm)  # doctest: +ELLIPSIS
    0.5...

    >>> import random
    >>> random.seed(0)
    >>> ratings = [random.gauss(1200, 200) for i in range(100000)]
    >>> gestalt(ratings)  # doctest: +ELLIPSIS
    8e-05...
    """
    if comp is None:
        comp = sorted(permutation)
    return SequenceMatcher(None, permutation, comp).ratio()


def meanDistance(permutation, comp=None, methods=(manhattan, gestalt)):
    """A certain measure of sortedness for the list A, based on mean of the 2 distances: manhattan and gestalt.

    >>> perm = [0, 1, 2, 3, 4]
    >>> meanDistance(perm)  # sorted  # doctest: +ELLIPSIS
    1.0

    >>> perm = [0, 1, 2, 5, 4, 3]
    >>> meanDistance(perm)  # almost sorted!  # doctest: +ELLIPSIS
    0.722...

    >>> perm = [2, 9, 6, 4, 0, 3, 1, 7, 8, 5]  # doctest: +ELLIPSIS
    >>> meanDistance(perm)
    0.4

    >>> perm = [2, 1, 6, 4, 0, 3, 5, 7, 8, 9]  # better sorted!  # doctest: +ELLIPSIS
    >>> meanDistance(perm)
    0.61

    .. warning:: I removed :func:`kendalltau` and :func:`spearmanr` as they were giving 100% for many cases where clearly there were no reason to give 100%...
    """
    distances = []
    for method in methods:
        distances.append(method(permutation, comp=comp))
    return np.mean(distances)


# Default distance
sortedDistance = meanDistance


# Only export and expose the useful functions defined here
__all__ = [
    "weightedDistance",
    "manhattan",
    "kendalltau",
    "spearmanr",
    "gestalt",
    "meanDistance",
    "sortedDistance",
]


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
