# -*- coding: utf-8 -*-
""" Result.Result class to wrap the simulation results."""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.26 $"

import numpy as np


class Result:
    """ Result accumulators"""

    def __init__(self, nbArms, horizon):
        self.choices = np.zeros(horizon)
        self.rewards = np.zeros(horizon)
        self.pulls = np.zeros(nbArms)

    def store(self, time, choice, reward):
        self.choices[time] = choice
        self.rewards[time] = reward
        self.pulls[choice] += 1
