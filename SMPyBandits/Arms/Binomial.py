# -*- coding: utf-8 -*-
""" Binomial distributed arm.

Example of creating an arm:

>>> import random; import numpy as np
>>> random.seed(0); np.random.seed(0)
>>> B03_10 = Binomial(0.3, 10)
>>> B03_10
Bin(0.3, 10)
>>> B03_10.mean
3.0

Examples of sampling from an arm:

>>> B03_10.draw()
3.0
>>> B03_10.draw_nparray(20)
array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,
        1.,  0.,  0.,  0.,  1.,  1.,  1.])
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.5"

# from random import random
import numpy as np
from numpy.random import binomial as npbinomial

from .Arm import Arm
from .kullback import klBin


class Binomial(Arm):
    """ Binomial distributed arm."""

    def __init__(self, probability, draws=1):
        """New arm."""
        assert 0 <= probability <= 1, "Error, the parameter probability for Binomial class has to be in [0, 1]."  # DEBUG
        assert isinstance(draws, int) and 1 <= draws, "Error, the parameter draws for Binomial class has to be an integer >= 1."  # DEBUG
        self.probability = probability  #: Parameter p for this Binomial arm
        self.draws = draws  #: Parameter n for this Binomial arm
        self.mean = probability * draws  #: Mean for this Binomial arm

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        return np.asarray(npbinomial(self.draws, self.probability), dtype=float)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        return np.asarray(npbinomial(self.draws, self.probability, shape), dtype=float)

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        return 0., self.draws

    def __str__(self):
        return "Binomial"

    def __repr__(self):
        return "Bin({:.3g}, {})".format(self.probability, self.draws)

    # --- Lower bound

    def kl(self, x, y):
        """ The kl(x, y) to use for this arm."""
        return klBin(x, y, self.draws)

    def oneLR(self, mumax, mu):
        """ One term of the Lai & Robbins lower bound for Binomial arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klBin(mu, mumax, self.draws)


# Only export and expose the class defined here
__all__ = ["Binomial"]
