# -*- coding: utf-8 -*-
"""Bernoulli distributed arm."""

from random import random


class Bernoulli(object):
    """Bernoulli distributed arm."""

    def __init__(self, probability):
        self.probability = probability

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        return float(random() < self.probability)

    def __repr__(self):
        return "<" + self.__class__.__name__ + ": " + repr(self.probability) + ">"
