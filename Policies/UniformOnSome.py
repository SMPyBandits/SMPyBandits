# -*- coding: utf-8 -*-
""" UniformOnSome: a fully uniform policy who selects randomly (uniformly) an arm among a fix set, at each step (stupid).
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from random import choice

from .Uniform import Uniform


class UniformOnSome(Uniform):
    """ UniformOnSome: a fully uniform policy who selects randomly (uniformly) an arm among a fix set, at each step (stupid).
    """

    def __init__(self, nbArms, armIndexes=None):
        self.nbArms = nbArms
        if armIndexes is None:
            armIndexes = list(range(nbArms))
        self.armIndexes = armIndexes

    def __str__(self):
        return "UniformOnSome({})".format(self.armIndexes)

    def choice(self):
        return choice(self.armIndexes)
