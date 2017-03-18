# -*- coding: utf-8 -*-
""" Poisson distributed arm, possibly truncated."""

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.5"

from math import isinf, exp
from scipy.stats import poisson

from .Arm import Arm
from .kullback import klPoisson


class Poisson(Arm):
    """ Poisson distributed arm, possibly truncated.

    - Default is to not truncate.
    - Warning: the draw() method is QUITE inefficient! (15 seconds for 200000 draws, 62 µs for 1).
    """

    def __init__(self, p, trunc=1):
        assert p >= 0, "Error, the parameter 'p' for Poisson arm has to be >= 0."
        self.p = p
        self.trunc = trunc
        if isinf(trunc):
            self.mean = p
        else:  # Warning: this is very slow if self.trunc is large!
            q = exp(-p)
            sq = q
            self.mean = 0
            for k in range(1, self.trunc):
                q *= p / k
                self.mean += k * q
                sq += q
            self.mean += self.trunc * (1 - sq)

    # --- Random samples

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        return min(poisson.rvs(self.p), self.trunc)

    def draw_nparray(self, shape=(1,)):
        """ The parameter t is ignored in this Arm."""
        return min(poisson.rvs(self.p, size=shape), self.trunc)

    # --- Printing

    def __str__(self):
        return "Poisson"

    def __repr__(self):
        if isinf(self.trunc):
            return "P({:.3g})".format(self.p)
        else:
            return "P({:.3g}, {:.3g})".format(self.p, self.trunc)

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        return klPoisson(x, y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Poisson arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klPoisson(mu, mumax)


class UnboundedPoisson(Poisson):
    """ Poisson distributed arm, not truncated, ie. trunc =  oo."""

    def __init__(self, p):
        super(UnboundedPoisson, self).__init__(p, trunc=float('+inf'))
