# -*- coding: utf-8 -*-
""" Arm with a constant reward. Useful for debugging.

Example of creating an arm:

>>> C013 = Constant(0.13)
>>> C013
Constant(0.13)
>>> C013.mean
0.13

Examples of sampling from an arm:

>>> C013.draw()
0.13
>>> C013.draw_nparray(3)
array([ 0.13,  0.13,  0.13])
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np

from .Arm import Arm


class Constant(Arm):
    """ Arm with a constant reward. Useful for debugging.

    - `constant_reward` is the constant reward,
    - `lower`, `amplitude` default to `floor(constant_reward)`, `1` (so the )

    >>> arm_0_5 = Constant(0.5)
    >>> arm_0_5.draw()
    0.5
    >>> arm_0_5.draw_nparray((3, 2))
    array([[ 0.5,  0.5],
           [ 0.5,  0.5],
           [ 0.5,  0.5]])
    """

    def __init__(self, constant_reward=0.5, lower=0., amplitude=1.):
        """ New arm."""
        constant_reward = float(constant_reward)
        self.constant_reward = constant_reward  #: Constant value of rewards
        lower = min(lower, np.floor(constant_reward))
        self.lower = lower  #: Known lower value of rewards
        self.amplitude = amplitude  #: Known amplitude of rewards
        self.mean = constant_reward  #: Mean for this Constant arm

    # --- Random samples

    def draw(self, t=None):
        """ Draw one constant sample. The parameter t is ignored in this Arm."""
        return self.constant_reward

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of constant samples, of a certain shape."""
        return np.full(shape, self.constant_reward)

    # --- Printing

    def __str__(self):
        return "Constant"

    def __repr__(self):
        return "Constant({:.3g})".format(self.constant_reward)

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        """ The `kl(x, y) = abs(x - y)` to use for this arm."""
        return abs(x - y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Constant arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / abs(mumax - mu)


__all__ = ["Constant"]
