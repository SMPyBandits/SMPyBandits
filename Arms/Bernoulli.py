# -*- coding: utf-8 -*-
""" Bernoulli distributed arm."""

__author__ = "Lilian Besson"
__version__ = "0.1"

from random import random
from .Arm import Arm


class Bernoulli(Arm):
    """ Bernoulli distributed arm."""

    def __init__(self, probability):
        assert 0 <= probability <= 1, "Error, the parameter probability for Bernoulli class has to be in [0, 1]."
        self.probability = probability

    def __str__(self):
        return "Bernoulli"

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        return float(random() < self.probability)

    def mean(self):
        return self.probability

    def __repr__(self):
        # return "<" + self.__class__.__name__ + ": " + repr(self.probability) + ">"
        return "B({})".format(self.probability)
