# -*- coding: utf-8 -*-
""" Gaussian distributed arm.

Example of creating an arm:

>>> import random; import numpy as np
>>> random.seed(0); np.random.seed(0)
>>> Gauss03 = Gaussian(0.3, 0.05)  # small variance
>>> Gauss03
N(0.3, 0.05)
>>> Gauss03.mean
0.3

Examples of sampling from an arm:

>>> Gauss03.draw()  # doctest: +ELLIPSIS
0.3470...
>>> Gauss03.draw_nparray(20)  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
array([0.388..., 0.320..., 0.348... , 0.412..., 0.393... ,
       0.251..., 0.347..., 0.292..., 0.294..., 0.320...,
       0.307..., 0.372..., 0.338..., 0.306..., 0.322...,
       0.316..., 0.374..., 0.289..., 0.315..., 0.257...])
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.9"

from random import gauss
from numpy.random import standard_normal
import numpy as np
from scipy.special import erf

# Local imports
try:
    from .Arm import Arm
    from .kullback import klGauss
except ImportError:
    from Arm import Arm
    from kullback import klGauss

oo = float('+inf')  # Nice way to write +infinity

#: Default value for the variance of a [0, 1] Gaussian arm
VARIANCE = 0.05


class Gaussian(Arm):
    """ Gaussian distributed arm, possibly truncated.

    - Default is to truncate into [0, 1] (so Gaussian.draw() is in [0, 1]).
    """

    def __init__(self, mu, sigma=VARIANCE, mini=0, maxi=1):
        """New arm."""
        self.mu = self.mean = mu  #: Mean of Gaussian arm
        assert sigma > 0, "Error, the parameter 'sigma' for Gaussian arm has to be > 0."
        self.sigma = sigma  #: Variance of Gaussian arm
        assert mini <= maxi, "Error, the parameter 'mini' for Gaussian arm has to < 'maxi'."  # DEBUG
        self.min = mini  #: Lower value of rewards
        self.max = maxi  #: Higher value of rewards
        # XXX if needed, compute the true mean : Cf. https://en.wikipedia.org/wiki/Truncated_normal_distribution#Moments
        # real_mean = mu + sigma * (phi(mini) - phi(maxi)) / (Phi(maxi) - Phi(mini))

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        return min(max(gauss(self.mu, self.sigma), self.min), self.max)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        return np.minimum(np.maximum(self.mu + self.sigma * standard_normal(shape), self.min), self.max)

    def set_mean_param(self, mean):
        self.mu = self.mean = mean

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        return self.min, self.max - self.min

    def __str__(self):
        return "Gaussian"

    def __repr__(self):
        return "N({:.3g}, {:.3g})".format(self.mu, self.sigma)

    # --- Lower bound

    def kl(self, x, y):
        """ The kl(x, y) to use for this arm."""
        return klGauss(x, y, self.sigma)

    def oneLR(self, mumax, mu):
        """ One term of the Lai & Robbins lower bound for Gaussian arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klGauss(mu, mumax, self.sigma)

    def oneHOI(self, mumax, mu):
        """ One term for the HOI factor for this arm."""
        return 1 - (mumax - mu) / self.max


class Gaussian_0_1(Gaussian):
    """ Gaussian distributed arm, truncated to [0, 1]."""
    def __init__(self, mu, sigma=0.05, mini=0, maxi=1):
        super(Gaussian_0_1, self).__init__(mu, sigma=sigma, mini=mini, maxi=maxi)


class Gaussian_0_2(Gaussian):
    """ Gaussian distributed arm, truncated to [0, 2]."""
    def __init__(self, mu, sigma=0.1, mini=0, maxi=2):
        super(Gaussian_0_2, self).__init__(mu, sigma=sigma, mini=mini, maxi=maxi)


class Gaussian_0_5(Gaussian):
    """ Gaussian distributed arm, truncated to [0, 5]."""
    def __init__(self, mu, sigma=0.5, mini=0, maxi=5):
        super(Gaussian_0_5, self).__init__(mu, sigma=sigma, mini=mini, maxi=maxi)


class Gaussian_0_10(Gaussian):
    """ Gaussian distributed arm, truncated to [0, 10]."""
    def __init__(self, mu, sigma=1, mini=0, maxi=10):
        super(Gaussian_0_10, self).__init__(mu, sigma=sigma, mini=mini, maxi=maxi)


class Gaussian_0_100(Gaussian):
    """ Gaussian distributed arm, truncated to [0, 100]."""
    def __init__(self, mu, sigma=5, mini=0, maxi=100):
        super(Gaussian_0_100, self).__init__(mu, sigma=sigma, mini=mini, maxi=maxi)


class Gaussian_m1_1(Gaussian):
    """ Gaussian distributed arm, truncated to [-1, 1]."""
    def __init__(self, mu, sigma=0.1, mini=-1, maxi=1):
        super(Gaussian_m1_1, self).__init__(mu, sigma=sigma, mini=mini, maxi=maxi)


class Gaussian_m2_2(Gaussian):
    """ Gaussian distributed arm, truncated to [-2, 2]."""
    def __init__(self, mu, sigma=0.25, mini=-2, maxi=2):
        super(Gaussian_m2_2, self).__init__(mu, sigma=sigma, mini=mini, maxi=maxi)


class Gaussian_m5_5(Gaussian):
    """ Gaussian distributed arm, truncated to [-5, 5]."""
    def __init__(self, mu, sigma=1, mini=-5, maxi=5):
        super(Gaussian_m5_5, self).__init__(mu, sigma=sigma, mini=mini, maxi=maxi)


class Gaussian_m10_10(Gaussian):
    """ Gaussian distributed arm, truncated to [-10, 10]."""
    def __init__(self, mu, sigma=2, mini=-10, maxi=10):
        super(Gaussian_m10_10, self).__init__(mu, sigma=sigma, mini=mini, maxi=maxi)


class Gaussian_m100_100(Gaussian):
    """ Gaussian distributed arm, truncated to [-100, 100]."""
    def __init__(self, mu, sigma=10, mini=-100, maxi=100):
        super(Gaussian_m100_100, self).__init__(mu, sigma=sigma, mini=mini, maxi=maxi)


#: Default value for the variance of an unbounded Gaussian arm
UNBOUNDED_VARIANCE = 1


class UnboundedGaussian(Gaussian):
    """ Gaussian distributed arm, not truncated, ie. supported in (-oo,  oo)."""

    def __init__(self, mu, sigma=UNBOUNDED_VARIANCE):
        """New arm."""
        super(UnboundedGaussian, self).__init__(mu, sigma=sigma, mini=-oo, maxi=oo)

    # def __str__(self):
    #     return "UnboundedGaussian"

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        return gauss(self.mu, self.sigma)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        return self.mu + self.sigma * standard_normal(shape)

    def __repr__(self):
        return "N({:.3g}, {:.3g})".format(self.mu, self.sigma)


def phi(xi):
    r"""The :math:`\phi(\xi)` function, defined by:

    .. math:: \phi(\xi) := \frac{1}{\sqrt{2 \pi}} \exp\left(- \frac12 \xi^2 \right)

    It is the probability density function of the standard normal distribution, see https://en.wikipedia.org/wiki/Standard_normal_distribution.
    """
    return np.exp(- 0.5 * xi**2) / np.sqrt(2. * np.pi)


def Phi(x):
    r"""The :math:`\Phi(x)` function, defined by:

    .. math:: \Phi(x) := \frac{1}{2} \left(1 + \mathrm{erf}\left( \frac{x}{\sqrt{2}} \right) \right).

    It is the probability density function of the standard normal distribution, see https://en.wikipedia.org/wiki/Cumulative_distribution_function
    """
    return (1. + erf(x / np.sqrt(2.))) / 2.


# Only export and expose the classes defined here
__all__ = [
    "Gaussian",
    "Gaussian_0_1",
    "Gaussian_0_2",
    "Gaussian_0_5",
    "Gaussian_0_10",
    "Gaussian_0_100",
    "Gaussian_m1_1",
    "Gaussian_m2_2",
    "Gaussian_m5_5",
    "Gaussian_m10_10",
    "Gaussian_m100_100",
    "UnboundedGaussian"
]


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
