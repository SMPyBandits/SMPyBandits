# -*- coding: utf-8 -*-
""" Gaussian distributed arm."""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.4 $"

from random import gauss


class Gaussian():
    """ Gaussian distributed arm."""

    def __init__(self, mu, sigma):
        self.sigma = sigma
        self.mu = mu
        self.expectation = mu

    def __str__(self):
        return "Gaussian"

    def mean(self):
        return self.expectation

    def draw(self):
        return self.mu + self.sigma * gauss(0, 1)

    def __repr__(self):
        # return "<" + self.__class__.__name__ + ": " + repr(self.mu) + ", " + repr(self.sigma) + ">"
        return "G({}, {})".format(self.mu, self.sigma)
