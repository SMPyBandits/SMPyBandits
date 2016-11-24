# -*- coding: utf-8 -*-
""" Poisson distributed arm, possibly truncated."""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.6 $"

from math import isinf, exp
from scipy.stats import poisson
from .Arm import Arm


class Poisson(Arm):
    """ Poisson distributed arm, possibly truncated."""

    def __init__(self, p, trunc=float('inf')):
        self.p = p
        self.trunc = trunc
        if isinf(trunc):
            self.expectation = p
        else:
            q = exp(-p)
            sq = q
            self.expectation = 0
            for k in range(1, self.trunc):
                q = q * p / k
                self.expectation += k * q
                sq += q
            self.expectation += self.trunc * (1 - sq)

    def __str__(self):
        return "Poisson"

    def mean(self):
        return self.expectation

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        return min(poisson.rvs(self.p), self.trunc)

    def __repr__(self):
        if isinf(self.trunc):
            # return "<" + self.__class__.__name__ + ": " + repr(self.p) + ">"
            return "P({})".format(self.p)
        else:
            # return "<" + self.__class__.__name__ + ": " + repr(self.p) + ", " + repr(self.trunc) + ">"
            return "P({}, {})".format(self.p, self.trunc)
