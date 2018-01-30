# -*- coding: utf-8 -*-
""" Uniformly distributed arm in [0, 1], or [lower, lower + amplitude].

Example of creating an arm:

>>> import random; import numpy as np
>>> random.seed(0); np.random.seed(0)
>>> Unif01 = UniformArm(0, 1)
>>> Unif01
U(0, 1)
>>> Unif01.mean
0.5

Examples of sampling from an arm:

>>> Unif01.draw()  # doctest: +ELLIPSIS
0.4049...
>>> Unif01.draw_nparray(20)  # doctest: +ELLIPSIS
array([ 0.5488...,  0.7151...,  0.6027...,  0.5448...,  0.4236...,
        0.6458...,  0.4375...,  0.8917...,  0.9636...,  0.3834...,
        0.7917...,  0.5288...,  0.5680...,  0.9255...,  0.0710...,
        0.0871...,  0.0202...,  0.8326...,  0.7781...,  0.8700...])
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

from random import random
from numpy.random import random as nprandom

from .Arm import Arm
from .kullback import klBern


class UniformArm(Arm):
    """ Uniformly distributed arm, default in [0, 1],

    - default to (mini, maxi),
    - or [lower, lower + amplitude], if (lower=lower, amplitude=amplitude) is given.

    >>> arm_0_1 = UniformArm()
    >>> arm_0_10 = UniformArm(0, 10)  # maxi = 10
    >>> arm_2_4 = UniformArm(2, 4)
    >>> arm_m10_10 = UniformArm(-10, 10)  # also UniformArm(lower=-10, amplitude=20)
    """

    def __init__(self, mean=None, mini=0., maxi=1., lower=0., amplitude=1.):
        """New arm."""
        mini = max(mini, lower)
        maxi = min(maxi, lower + amplitude)
        if mean is not None:
            assert mini <= mean <= maxi, "Error: 'mean' = {} argument for UniformArm has to be between 'mini' = {} and 'maxi' = {}...".format(mean, mini, maxi)  # DEBUG
            gap = min(mean - mini, maxi - mean)
            assert mini <= mean - gap <= mean + gap <= maxi, "Error: computing 'gap' = {} was wrong...".format(gap)  # DEBUG
            mini = mean - gap
            maxi = mean + gap
        assert mini >= lower, "Error: 'mini' = {} argument for UniformArm has to be >= 'lower' = {}...".format(mini, lower)  # DEBUG
        self.lower = mini  #: Lower value of rewards
        assert maxi <= lower + amplitude, "Error: 'maxi' = {} argument for UniformArm has to be >= 'lower + amplitude' = {}...".format(maxi, lower + amplitude)
        self.amplitude = maxi - mini  #: Amplitude of rewards
        # self.mean = (mini + maxi) / 2.0
        self.mean = self.lower + (self.amplitude / 2.0)  #: Mean for this UniformArm arm

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        return self.lower + (random() * self.amplitude)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        return self.lower + (nprandom(shape) * self.amplitude)

    # --- Printing

    def __str__(self):
        return "UniformArm"

    def __repr__(self):
        return "U({:.3g}, {:.3g})".format(self.lower, self.lower + self.amplitude)

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        return klBern(x, y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for UniformArm arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klBern(mu, mumax)


__all__ = ["UniformArm"]
