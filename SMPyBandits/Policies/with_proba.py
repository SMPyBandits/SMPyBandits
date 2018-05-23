# -*- coding: utf-8 -*-
""" Simply defines a function :func:`with_proba` that is used everywhere.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from random import random


# --- Utility functions


def with_proba(epsilon):
    r"""Bernoulli test, with probability :math:`\varepsilon`, return `True`, and with probability :math:`1 - \varepsilon`, return `False`.

    Example:

    >>> from random import seed; seed(0)  # reproductible
    >>> with_proba(0.5)
    False
    >>> with_proba(0.9)
    True
    >>> with_proba(0.1)
    False
    >>> if with_proba(0.2):
    ...     print("This happens 20% of the time.")
    """
    assert 0 <= epsilon <= 1, "Error: for 'with_proba(epsilon)', epsilon = {:.3g} has to be between 0 and 1 to be a valid probability.".format(epsilon)  # DEBUG
    return random() < epsilon  # True with proba epsilon


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
