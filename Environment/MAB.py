# -*- coding: utf-8 -*-
""" MAB.MAB class to wrap the arms."""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.26 $"

import numpy as np


class MAB:
    """ Multi-armed Bandit environment."""

    def __init__(self, configuration):
        arm_type = configuration["arm_type"]
        probabilities = configuration["probabilities"]
        self.arms = [arm_type(probability) for probability in probabilities]
        self.nbArms = len(self.arms)
        self.maxArm = np.max(probabilities)

    def __repr__(self):
        return '<' + self.__class__.__name__ + repr(self.__dict__) + '>'
