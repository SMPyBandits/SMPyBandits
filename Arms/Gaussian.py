# -*- coding: utf-8 -*-
""" Gaussian distributed arm."""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.4 $"

from random import gauss
from .Arm import Arm

# oo = float("+inf")  # Nice way to write +infinity


class Gaussian(Arm):
    """ Gaussian distributed arm, possibly truncated.

    - Default is to truncate into [0, 1] (so Gaussian.draw() is in [0, 1]).
    """

    # def __init__(self, mu, sigma=0.1, trunc=[-oo, oo]):
    def __init__(self, mu, sigma=0.1, trunc=[0, 1]):
        self.mu = mu
        self.expectation = mu
        assert sigma > 0, "Error, the parameter 'sigma' for Gaussian class has to be > 0."
        self.sigma = sigma
        assert trunc[0] <= trunc[1], "Error, the parameter 'trunc' for Exponential class has to a tuple with trunc[0] < trunc[1]."
        self.min = trunc[0]
        self.max = trunc[1]

    def __str__(self):
        return "Gaussian"

    def mean(self):
        return self.expectation

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        return min(max(self.mu + self.sigma * gauss(0, 1), self.min), self.max)

    def __repr__(self):
        return "G({}, {})".format(self.mu, self.sigma)
