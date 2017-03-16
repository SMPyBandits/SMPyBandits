# -*- coding: utf-8 -*-
""" Gamma distributed arm."""

__author__ = "Lilian Besson"
__version__ = "0.6"

from random import gammavariate
from numpy.random import gamma
import numpy as np

from .Arm import Arm
from .kullback import klGamma

oo = float('+inf')  # Nice way to write +infinity

SCALE = 1.


class Gamma(Arm):
    """ Gamma distributed arm, possibly truncated.

    - Default is to truncate into [0, 1] (so Gamma.draw() is in [0, 1]).
    - Cf. http://chercheurs.lille.inria.fr/ekaufman/NIPS13 Figure 1
    """

    # def __init__(self, shape, scale=SCALE, mini=-oo, maxi=oo):
    def __init__(self, shape, scale=SCALE, mini=0, maxi=1):
        assert shape > 0, "Error, the parameter 'shape' for Gamma arm has to be > 0."
        self.shape = shape
        assert scale > 0, "Error, the parameter 'scale' for Gamma arm has to be > 0."
        self.scale = scale
        self.mean = shape * scale
        assert mini <= maxi, "Error, the parameter 'trunc' for Gamma arm has to a tuple with mini < maxi."
        self.min = mini
        self.max = maxi

    # --- Random samples

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        return min(max(gammavariate(self.shape, self.scale), self.min), self.max)

    def draw_nparray(self, shape=(1,)):
        """ The parameter t is ignored in this Arm."""
        return np.minimum(np.maximum(gamma(self.shape, self.scale, size=shape), self.min), self.max)

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def lower_amplitude(self):
        return self.min, self.max - self.min

    def __str__(self):
        return "Gamma"

    def __repr__(self):
        return "Gamma({:.3g}, {:.3g})".format(self.shape, self.scale)

    # --- Lower bound

    def kl(self, x, y):
        # FIXME if x, y are means (ie self.shape), shouldn't we divide them by self.scale ?
        return klGamma(x, y, self.scale)

    def oneLR(self, mumax, mu):
        """ One term of the Lai & Robbins lower bound for Gaussian arms: (mumax - shape) / KL(shape, mumax). """
        return (mumax - mu) / klGamma(mu, mumax, self.scale)
