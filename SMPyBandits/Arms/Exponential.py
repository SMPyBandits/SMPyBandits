# -*- coding: utf-8 -*-
""" Exponentially distributed arm.

Example of creating an arm:

>>> import random; import numpy as np
>>> random.seed(0); np.random.seed(0)
>>> Exp03 = ExponentialFromMean(0.3)
>>> Exp03
\mathrm{Exp}(3.2, 1)
>>> Exp03.mean  # doctest: +ELLIPSIS
0.3000...

Examples of sampling from an arm:

>>> Exp03.draw()  # doctest: +ELLIPSIS
0.052...
>>> Exp03.draw_nparray(20)  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
array([0.18..., 0.10..., 0.15..., 0.18..., 0.26...,
       0.13..., 0.25..., 0.03..., 0.01..., 0.29... ,
       0.07..., 0.19..., 0.17..., 0.02... , 0.82... ,
       0.76..., 1.     , 0.05..., 0.07..., 0.04...])
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.5"

from math import isinf, exp, log
from random import random
import numpy as np
from numpy.random import random as nprandom
from scipy.optimize import minimize

# Local imports
try:
    from .Arm import Arm
    from .kullback import klExp
except ImportError:
    from Arm import Arm
    from kullback import klExp


def p_of_expectation(expectation, trunc=1):
    """Use a numerical solver (:func:`scipy.optimize.minimize`) to find the value p giving an arm Exp(p) of a given expectation."""
    if isinf(trunc):
        def expp(p):
            """ mean = expectation(p)."""
            return 1. / p
    else:
        def expp(p):
            """ mean = expectation(p)."""
            return (1. - exp(-p * trunc)) / p

    def objective(p):
        """ Objective function to minimize."""
        return abs(expectation - expp(p))

    return minimize(objective, 1).x[0]


class Exponential(Arm):
    """ Exponentially distributed arm, possibly truncated.

    - Default is to truncate to 1 (so Exponential.draw() is in [0, 1]).
    """

    # def __init__(self, p, trunc=float('+inf')):
    def __init__(self, p, trunc=1):
        """New arm."""
        self.p = p  #: Parameter p for Exponential arm
        assert p > 0, "Error, the parameter 'p' for Exponential arm has to be > 0."
        self.trunc = trunc  #: Max value of reward
        assert trunc > 0, "Error, the parameter 'trunc' for Exponential arm has to be > 0."
        if isinf(trunc):
            self.mean = 1. / p  #: Mean of Exponential arm
        else:
            self.mean = (1. - exp(-p * trunc)) / p

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        return min((-1. / self.p) * log(random()), self.trunc)

    def draw_nparray(self, shape=(1,)):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        return np.minimum((-1. / self.p) * np.log(nprandom(shape)), self.trunc)

    def set_mean_param(self, p_inv):
        self.p = 1 / p_inv
        if isinf(self.trunc):
            self.mean = 1. / self.p  #: Mean of Exponential arm
        else:
            self.mean = (1. - exp(-self.p * self.trunc)) / self.p

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        return 0., self.trunc

    def __str__(self):
        return "Exponential"

    def __repr__(self):
        return r"{}({:.3g}{})".format(r'\mathrm{Exp}', self.p, '' if isinf(self.trunc) else ', {:.3g}'.format(self.trunc))

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        return klExp(x, y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Exponential arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klExp(mu, mumax)

    def oneHOI(self, mumax, mu):
        """ One term for the HOI factor for this arm."""
        return 1 - (mumax - mu) / self.trunc


class ExponentialFromMean(Exponential):
    """ Exponentially distributed arm, possibly truncated, defined by its mean and not its parameter.

    - Default is to truncate to 1 (so Exponential.draw() is in [0, 1]).
    """

    def __init__(self, mean, trunc=1):
        """New arm."""
        p = p_of_expectation(mean)
        super(ExponentialFromMean, self).__init__(p, trunc=trunc)


class UnboundedExponential(Exponential):
    """ Exponential distributed arm, not truncated, ie. trunc =  oo."""

    def __init__(self, mu):
        """New arm."""
        super(UnboundedExponential, self).__init__(mu, trunc=float('+inf'))



# Only export and expose the class defined here
__all__ = ["Exponential", "ExponentialFromMean", "UnboundedExponential"]


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
