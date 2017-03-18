# -*- coding: utf-8 -*-
""" Exponentially distributed arm."""

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.5"

from math import isinf, exp, log
from random import random
import numpy as np
from numpy.random import random as nprandom
from scipy.optimize import minimize

# Local imports
from .Arm import Arm
from .kullback import klExp


def p_of_expectation(expectation, trunc=1):
    if isinf(trunc):
        def expp(p):
            return 1. / p
    else:
        def expp(p):
            return (1. - exp(-p * trunc)) / p

    def objective(p):
        return abs(expectation - expp(p))

    return minimize(objective, 1).x[0]


class Exponential(Arm):
    """ Exponentially distributed arm, possibly truncated.

    - Default is to truncate to 1 (so Exponential.draw() is in [0, 1]).
    """

    # def __init__(self, p, trunc=float('+inf')):
    def __init__(self, p, trunc=1):
        self.p = p
        assert p > 0, "Error, the parameter 'p' for Exponential arm has to be > 0."
        self.trunc = trunc
        assert trunc > 0, "Error, the parameter 'trunc' for Exponential arm has to be > 0."
        if isinf(trunc):
            self.mean = 1. / p
        else:
            self.mean = (1. - exp(-p * trunc)) / p

    # --- Random samples

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        return min((-1. / self.p) * log(random()), self.trunc)

    def draw_nparray(self, shape=(1,)):
        """ The parameter t is ignored in this Arm."""
        return np.minimum((-1. / self.p) * np.log(nprandom(shape)), self.trunc)

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def lower_amplitude(self):
        return 0., self.trunc

    def __str__(self):
        return "Exponential"

    def __repr__(self):
        return "Exp({:.3g}{})".format(self.p, '' if isinf(self.trunc) else ', {:.3g}'.format(self.trunc))

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        return klExp(x, y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Exponential arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klExp(mu, mumax)

    def oneHOI(self, mumax, mu):
        return 1 - (mumax - mu) / self.trunc


class ExponentialFromMean(Exponential):
    """ Exponentially distributed arm, possibly truncated, defined by its mean and not its parameter.

    - Default is to truncate to 1 (so Exponential.draw() is in [0, 1]).
    """

    def __init__(self, mean, trunc=1):
        p = p_of_expectation(mean)
        super(ExponentialFromMean, self).__init__(p, trunc=trunc)


class UnboundedExponential(Exponential):
    """ Exponential distributed arm, not truncated, ie. trunc =  oo."""

    def __init__(self, mu):
        super(UnboundedExponential, self).__init__(mu, trunc=float('+inf'))
