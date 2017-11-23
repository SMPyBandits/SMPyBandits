# -*- coding: utf-8 -*-
""" Bernoulli distributed arm.

Example of creating an arm:

>>> import random; import numpy as np
>>> random.seed(0); np.random.seed(0)
>>> B03 = Bernoulli(0.3)
>>> B03
B(0.3)
>>> B03.mean
0.3

Examples of sampling from an arm:

>>> B03.draw()
0.0
>>> B03.draw_nparray(20)
array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,
        1.,  0.,  0.,  0.,  1.,  1.,  1.])
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
from numpy.random import binomial

from .Arm import Arm
from .kullback import klBern


class Bernoulli(Arm):
    """ Bernoulli distributed arm."""

    def __init__(self, probability):
        """New arm."""
        assert 0 <= probability <= 1, "Error, the parameter probability for Bernoulli class has to be in [0, 1]."  # DEBUG
        self.probability = probability  #: Parameter p for this Bernoulli arm
        self.mean = probability  #: Mean for this Bernoulli arm

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample."""
        return np.asarray(binomial(1, self.probability), dtype=float)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        return np.asarray(binomial(1, self.probability, shape), dtype=float)

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        return 0., 1.

    def __str__(self):
        return "Bernoulli"

    def __repr__(self):
        return "B({:.3g})".format(self.probability)

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        return klBern(x, y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Bernoulli arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klBern(mu, mumax)


# Only export and expose the class defined here
__all__ = ["Bernoulli"]
