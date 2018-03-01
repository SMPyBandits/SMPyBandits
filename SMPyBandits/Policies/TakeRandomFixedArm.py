# -*- coding: utf-8 -*-
""" TakeRandomFixedArm: always select a fixed arm.
This is the perfect static policy if armIndex = bestArmIndex (not realistic, for test only).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
import numpy.random as rn
from .TakeFixedArm import TakeFixedArm


class TakeRandomFixedArm(TakeFixedArm):
    """ TakeRandomFixedArm: first selects a random sub-set of arms, then always select from it. """

    def __init__(self, nbArms, lower=0., amplitude=1., nbArmIndexes=None):
        self.nbArms = nbArms  #: Number of arms
        #: Get the number of arms, randomly!
        if nbArmIndexes is None:
            nbArmIndexes = rn.randint(low=1, high=1 + int(nbArms / 2.))
        #: Fix the set of arms
        self.armIndexes = list(rn.choice(np.arange(nbArms), size=nbArmIndexes, replace=False))

    def __str__(self):
        return "TakeRandomFixedArm({})".format(self.armIndexes)

    def choice(self):
        """Uniform choice from armIndexes."""
        return rn.choice(self.armIndexes)
