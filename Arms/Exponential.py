# -*- coding: utf-8 -*-
""" Exponentially distributed arm."""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.5 $"

from math import isinf, exp, log
from random import random

from .Arm import Arm
from .kullback import klExp


class Exponential(Arm):
    """ Exponentially distributed arm, possibly truncated.

    - Default is to truncate to 1 (so Exponential.draw() is in [0, 1]).
    """

    # def __init__(self, p, trunc=float('+inf')):
    def __init__(self, p, trunc=1):
        self.p = p
        assert p > 0, "Error, the parameter 'p' for Exponential class has to be > 0."
        self.trunc = trunc
        assert trunc > 0, "Error, the parameter 'trunc' for Exponential class has to be > 0."
        if isinf(trunc):
            self.expectation = 1. / p
        else:
            self.expectation = (1. - exp(-p * trunc)) / p

    # --- Random samples

    def mean(self):
        return self.expectation

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        return min(-1. / self.p * log(random()), self.trunc)

    # --- Printing

    def __str__(self):
        return "Exponential"

    def __repr__(self):
        if isinf(self.trunc):
            # return "<" + self.__class__.__name__ + ": " + repr(self.p) + ">"
            return "Exp({})".format(self.p)
        else:
            # return "<" + self.__class__.__name__ + ": " + repr(self.p) + ", " + repr(self.trunc) + ">"
            return "Exp({}, {})".format(self.p, self.trunc)

    # --- Lower bound

    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Exponential arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klExp(mu, mumax)
