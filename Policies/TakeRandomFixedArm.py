# -*- coding: utf-8 -*-
""" TakeRandomFixedArm: always select a fixed arm.
This is the perfect static policy if armIndex = bestArmIndex (not realistic, for test only).
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np
import numpy.random as rd
from .TakeFixedArm import TakeFixedArm


class TakeRandomFixedArm(TakeFixedArm):
    """ TakeRandomFixedArm: first selects a random sub-set of arms, then always select from it. """

    def __init__(self, nbArms):
        self.nbArms = nbArms
        # Get the number of arms, randomly!
        nbArmIndexes = rd.randint(low=1, high=1 + int(nbArms / 2.))
        # Fix the set of arms
        self.armIndexes = list(rd.choice(np.arange(nbArms), size=nbArmIndexes, replace=False))

    def __str__(self):
        return "TakeRandomFixedArm({})".format(self.armIndexes)

    def choice(self):
        return rd.choice(self.armIndexes)
