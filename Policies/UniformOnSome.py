# -*- coding: utf-8 -*-
""" UniformOnSome: a fully uniform policy who selects randomly (uniformly) an arm among a fix set, at each step (stupid).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.1"

from random import choice

from .Uniform import Uniform


class UniformOnSome(Uniform):
    """ UniformOnSome: a fully uniform policy who selects randomly (uniformly) an arm among a fix set, at each step (stupid).
    """

    def __init__(self, nbArms, armIndexes=None, lower=0., amplitude=1.):
        self.nbArms = nbArms  #: Number of arms
        if armIndexes is None:
            armIndexes = list(range(nbArms))
        self.armIndexes = armIndexes  #: Arms from where to uniformly sample

    def __str__(self):
        return "UniformOnSome({})".format(self.armIndexes)

    def choice(self):
        """Uniform choice from armIndexes."""
        return choice(self.armIndexes)
