# -*- coding: utf-8 -*-
""" Discretely distributed arm, of finite support.

Example of creating an arm:

>>> import random; import numpy as np
>>> random.seed(0); np.random.seed(0)
>>> D3values = DiscreteArm({-1: 0.25, 0: 0.5, 1: 0.25})
>>> D3values
D({-1: 0.25, 0: 0.5, 1: 0.25})
>>> D3values.mean
0.0

- Examples of sampling from an arm:

>>> D3values.draw()
0
>>> D3values.draw_nparray(20)
array([ 0,  0,  0,  0,  0,  0,  1,  1,  0,  1,  0,  0,  1, -1, -1, -1,  1,
        1,  1,  1])

- Another example, with heavy tail:

>>> D5values = DiscreteArm({-1000: 0.001, 0: 0.5, 1: 0.25, 2:0.25, 1000: 0.001})
>>> D5values
D({-1e+03: 0.001, 0: 0.5, 1: 0.25, 2: 0.25, 1e+03: 0.001})
>>> D5values.mean
0.75

Examples of sampling from an arm:

>>> D5values.draw()
2
>>> D5values.draw_nparray(20)
array([0, 2, 0, 1, 0, 2, 1, 0, 0, 2, 0, 1, 0, 1, 1, 1, 2, 1, 0, 0])
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
from numpy.random import choice

# Local imports
try:
    from .Arm import Arm
    from .kullback import klBern
except ImportError:
    from Arm import Arm
    from kullback import klBern


class DiscreteArm(Arm):
    """ DiscreteArm distributed arm."""

    def __init__(self, values_to_proba):
        """New arm."""
        assert len(values_to_proba) > 1, "Error: DiscreteArm values_to_proba dictionnary argument cannot be empty!"
        self._values_to_proba = values_to_proba.copy()
        self._items = list(values_to_proba.items())
        self._values = np.array(list(values_to_proba.keys()))
        # Check probabilities
        self._probabilities = np.array(list(values_to_proba.values()))
        self._probabilities /= np.sum(self._probabilities)
        assert all(0 <= p <= 1 for p in self._probabilities), "Error, the probabilities (values of the 'values_to_proba' dict) for DiscreteArm class has to all be in [0, 1]."  # DEBUG
        assert np.isclose(sum(self._probabilities), 1), "Error, the total probability (sum of values of the 'values_to_proba' dict) for DiscreteArm class has to be ~= 1, but was = {:.3g} here.".format(sum(self._probabilities))  # DEBUG
        # store mean, min, max
        self._lower = min(self._values)
        self._amplitude = max(self._values) - self._lower
        self.mean = sum(v * p for v, p in self._items)  #: Mean for this DiscreteArm arm
        self.size = len(self._values)  #: Number of different values in this DiscreteArm arm

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample."""
        return choice(self._values, p=self._probabilities)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        return np.asarray(choice(self._values, p=self._probabilities, replace=True, size=shape))

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        return 0., 1.

    def __str__(self):
        return "DiscreteArm"

    def __repr__(self):
        # return "D({})".format(repr(self._values_to_proba))
        return "D({}{}{})".format("{", ", ".join("{:.3g}: {:.3g}".format(v, p) for v, p in self._items), "}")

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm.

        .. warning:: FIXME this is not correctly defined, except for the special case of having **only** 2 values, a ``DiscreteArm`` is NOT a one-dimensional distribution, and so the kl between two distributions is NOT a function of their mean!
        """
        print("WARNING: DiscreteArm.kl({:.3g}, {:.3g}) is not defined, klBern is used but this is WRONG.".format(x, y))  # DEBUG
        return klBern(x, y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for DiscreteArm arms: (mumax - mu) / KL(mu, mumax).

        .. warning:: FIXME this is not correctly defined, except for the special case of having **only** 2 values, a ``DiscreteArm`` is NOT a one-dimensional distribution, and so the kl between two distributions is NOT a function of their mean!
        """
        print("WARNING: DiscreteArm.oneLR({:.3g}, {:.3g}) is not defined, klBern is used but this is WRONG.".format(mumax, mu))  # DEBUG
        return (mumax - mu) / klBern(mu, mumax)


# Only export and expose the class defined here
__all__ = ["DiscreteArm"]


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
