# -*- coding: utf-8 -*-
""" Gaussian distributed arm."""

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.5"

from random import gauss
from numpy.random import standard_normal
import numpy as np

from .Arm import Arm
from .kullback import klGauss

# oo = float('+inf')  # Nice way to write +infinity

VARIANCE = 0.05


class Gaussian(Arm):
    """ Gaussian distributed arm, possibly truncated.

    - Default is to truncate into [0, 1] (so Gaussian.draw() is in [0, 1]).
    """

    # def __init__(self, mu, sigma=VARIANCE, mini=-oo, maxi=oo):
    def __init__(self, mu, sigma=VARIANCE, mini=0, maxi=1):
        self.mu = mu
        self.expectation = mu
        assert sigma > 0, "Error, the parameter 'sigma' for Gaussian class has to be > 0."
        self.sigma = sigma
        assert mini <= maxi, "Error, the parameter 'trunc' for Exponential class has to a tuple with mini < maxi."
        self.min = mini
        self.max = maxi

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def mean(self):
        return self.expectation

    # --- Random samples

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        return min(max(gauss(self.mu, self.sigma), self.min), self.max)

    def draw_nparray(self, shape=(1,)):
        """ The parameter t is ignored in this Arm."""
        return np.minimum(np.maximum(self.mu + self.sigma * standard_normal(shape), self.min), self.max)

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def lower_amplitude(self):
        return (self.min, self.max - self.min)

    def __str__(self):
        return "Gaussian"

    def __repr__(self):
        return "G({:.3g}, {:.3g})".format(self.mu, self.sigma)

    # --- Lower bound

    # @staticmethod
    def kl(self, x, y):
        # return klGauss(x, y, VARIANCE)
        return klGauss(x, y, self.sigma)

    # @staticmethod
    def oneLR(self, mumax, mu):
        """ One term of the Lai & Robbins lower bound for Gaussian arms: (mumax - mu) / KL(mu, mumax). """
        # return (mumax - mu) / klGauss(mu, mumax, VARIANCE)
        return (mumax - mu) / klGauss(mu, mumax, self.sigma)
