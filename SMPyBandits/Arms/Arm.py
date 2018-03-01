# -*- coding: utf-8 -*-
""" Base class for an arm class."""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"


class Arm(object):
    """ Base class for an arm class."""

    def __init__(self, lower=0., amplitude=1.):
        """ Base class for an arm class."""
        self.lower = lower  #: Lower value of rewards
        self.amplitude = amplitude  #: Amplitude of value of rewards
        self.min = lower  #: Lower value of rewards
        self.max = lower + amplitude  #: Higher value of rewards

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        if hasattr(self, 'lower') and hasattr(self, 'amplitude'):
            return self.lower, self.amplitude
        elif hasattr(self, 'min') and hasattr(self, 'max'):
            return self.min, self.max - self.min
        else:
            raise NotImplementedError("This method lower_amplitude() has to be implemented in the class inheriting from Arm.")

    # --- Printing

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dir__)

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample."""
        raise NotImplementedError("This method draw(t) has to be implemented in the class inheriting from Arm.")

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        raise NotImplementedError("This method draw_nparray(t) has to be implemented in the class inheriting from Arm.")

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        raise NotImplementedError("This method kl(x, y) has to be implemented in the class inheriting from Arm.")

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Gaussian arms: (mumax - mu) / KL(mu, mumax). """
        raise NotImplementedError("This method oneLR(mumax, mu) has to be implemented in the class inheriting from Arm.")

    @staticmethod
    def oneHOI(mumax, mu):
        """ One term for the HOI factor for this arm."""
        return 1 - (mumax - mu)
