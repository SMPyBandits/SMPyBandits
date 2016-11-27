# -*- coding: utf-8 -*-
""" Gaussian distributed arm."""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.4 $"

from random import gauss
from .Arm import Arm


class Gaussian(Arm):
    """ Gaussian distributed arm, possibly truncated.

    - Default is to truncate into [0, 1] (so Gaussian.draw() is in [0, 1]).
    """

    def __init__(self, mu, sigma, trunc=[0, 1]):
        assert sigma > 0, "Error, the parameter 'sigma' for Gaussian class has to be > 0."
        self.sigma = sigma
        self.mu = mu
        self.expectation = mu
        assert trunc[0] <= trunc[1], "Error, the parameter 'trunc' for Exponential class has to a tuple with trunc[0] < trunc[1]."
        self.trunc = trunc

    def __str__(self):
        return "Gaussian"

    def mean(self):
        return self.expectation

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        return min(max(self.mu + self.sigma * gauss(0, 1), self.trunc[0]), self.trunc[1])

    def __repr__(self):
        # return "<" + self.__class__.__name__ + ": " + repr(self.mu) + ", " + repr(self.sigma) + ">"
        return "G({}, {})".format(self.mu, self.sigma)
