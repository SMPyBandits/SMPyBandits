# -*- coding: utf-8 -*-
""" Exponentially distributed arm."""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.5 $"

from math import isinf, exp, log
from random import random


class Exponential():
    """Exponentially distributed arm, possibly truncated."""

    def __init__(self, p, trunc=float('inf')):
        self.p = p
        self.trunc = trunc
        if isinf(trunc):
            self.expectation = 1. / p
        else:
            self.expectation = (1. - exp(-p * trunc)) / p

    def __str__(self):
        return "Exponential"

    def mean(self):
        return self.expectation

    def draw(self):
        return min(-1. / self.p * log(random()), self.trunc)

    def __repr__(self):
        if isinf(self.trunc):
            # return "<" + self.__class__.__name__ + ": " + repr(self.p) + ">"
            return "Exp({})".format(self.p)
        else:
            # return "<" + self.__class__.__name__ + ": " + repr(self.p) + ", " + repr(self.trunc) + ">"
            return "Exp({}, {})".format(self.p, self.trunc)
