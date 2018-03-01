# -*- coding: utf-8 -*-
""" Gamma distributed arm.

Example of creating an arm:

>>> import random; import numpy as np
>>> random.seed(0); np.random.seed(0)
>>> Gamma03 = GammaFromMean(0.3)
>>> Gamma03
\Gamma(0.3, 1)
>>> Gamma03.mean
0.3

Examples of sampling from an arm:

>>> Gamma03.draw()  # doctest: +ELLIPSIS
1
>>> Gamma03.draw_nparray(20)  # doctest: +ELLIPSIS
array([  1.35...e-01,   1.84...e-01,   5.71...e-02,
         6.36...e-02,   4.94...e-01,   1.51...e-01,
         1.48...e-04,   2.25...e-06,   4.56...e-01,
         1.00...e+00,   7.59...e-02,   8.12...e-04,
         1.54...e-03,   1.14...e-01,   1.18...e-02,
         7.30...e-02,   1.76...e-06,   1.94...e-01,
         1.00...e+00,   3.30...e-02])
"""
from __future__ import division, print_function  # Python 2 compatibility

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

    # def __init__(self, shape, scale=SCALE, mini=-oo, maxi=oo):  # XXX Non truncated!
    def __init__(self, shape, scale=SCALE, mini=0, maxi=1):
        """New arm."""
        assert shape > 0, "Error, the parameter 'shape' for Gamma arm has to be > 0."
        self.shape = shape  #: Shape parameter for this Gamma arm
        assert scale > 0, "Error, the parameter 'scale' for Gamma arm has to be > 0."
        self.scale = scale  #: Scale parameter for this Gamma arm
        self.mean = shape * scale  #: Mean for this Gamma arm
        assert mini <= maxi, "Error, the parameter 'mini' for Gamma arm has to a tuple with > 'maxi'."  # DEBUG
        self.min = mini  #: Lower value of rewards
        self.max = maxi  #: Larger value of rewards

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        return min(max(gammavariate(self.shape, self.scale), self.min), self.max)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        return np.minimum(np.maximum(gamma(self.shape, self.scale, size=shape), self.min), self.max)

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        return self.min, self.max - self.min

    def __str__(self):
        return "Gamma"

    def __repr__(self):
        return r"\Gamma({:.3g}, {:.3g})".format(self.shape, self.scale)

    # --- Lower bound

    def kl(self, x, y):
        """ The kl(x, y) to use for this arm."""
        # FIXME if x, y are means (ie self.shape), shouldn't we divide them by self.scale ?
        return klGamma(x, y, self.scale)

    def oneLR(self, mumax, mu):
        """ One term of the Lai & Robbins lower bound for Gaussian arms: (mumax - shape) / KL(shape, mumax). """
        return (mumax - mu) / klGamma(mu, mumax, self.scale)

    def oneHOI(self, mumax, mu):
        """ One term for the HOI factor for this arm."""
        return 1 - (mumax - mu) / self.max


class GammaFromMean(Gamma):
    """ Gamma distributed arm, possibly truncated, defined by its mean and not its scale parameter."""

    def __init__(self, mean, scale=SCALE, mini=0, maxi=1):
        """As mean = scale * shape, shape = mean / scale is used."""
        shape = mean / scale
        super(GammaFromMean, self).__init__(shape, scale=scale, mini=mini, maxi=maxi)


class UnboundedGamma(Gamma):
    """ Gamma distributed arm, not truncated, ie. supported in (-oo,  oo)."""

    def __init__(self, shape, scale=SCALE):
        """New arm."""
        super(UnboundedGamma, self).__init__(shape, scale=scale, mini=-oo, maxi=oo)


# Only export and expose the class defined here
__all__ = ["Gamma", "GammaFromMean", "UnboundedGamma"]
