# -*- coding: utf-8 -*-
""" Bernoulli distributed arm."""

__author__ = "Lilian Besson"
__version__ = "0.1"

from random import random

from .Arm import Arm
from .kullback import klBern


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

    def lowerbound(self, means):
        """ Compute the Lai & Robbins lower bounds for a list of Bernoulli arms. """
        bestMean = max(means)
        return sum(oneLR(bestMean, mean) for mean in means if mean != bestMean)


def oneLR(mumax, mu):
    """ One term of the Lai & Robbins lower bound for Bernoulli arms: (mumax - mu) / KL(mu, mumax). """
    return (mumax - mu) / klBern(mu, mumax)
