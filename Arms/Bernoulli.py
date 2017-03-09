# -*- coding: utf-8 -*-
""" Bernoulli distributed arm."""

__author__ = "Lilian Besson"
__version__ = "0.1"

from random import random
from numpy.random import random as nprandom

from .Arm import Arm
from .kullback import klBern


class Bernoulli(Arm):
    """ Bernoulli distributed arm."""

    def __init__(self, probability):
        assert 0 <= probability <= 1, "Error, the parameter probability for Bernoulli class has to be in [0, 1]."
        self.probability = probability

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def mean(self):
        return self.probability

    # --- Random samples

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        return float(random() <= self.probability)

    def draw_nparray(self, shape=(1,)):
        """ The parameter t is ignored in this Arm."""
        return 1.0 * (nprandom(shape) <= self.probability)

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def lower_amplitude(self):
        return (0., 1.)

    def __str__(self):
        return "Bernoulli"

    def __repr__(self):
        return "B({:.3g})".format(self.probability)

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        return klBern(x, y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Bernoulli arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klBern(mu, mumax)
