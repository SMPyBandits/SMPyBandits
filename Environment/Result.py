# -*- coding: utf-8 -*-
""" Result.Result class to wrap the simulation results."""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.26 $"

import numpy as np


class Result(object):
    """ Result accumulators"""

    def __init__(self, nbArms, horizon):
        self.choices = np.zeros(horizon, dtype=int)
        self.rewards = np.zeros(horizon)         # To compute the mean
        self.rewardsSquared = np.zeros(horizon)  # To compute the variance
        self.pulls = np.zeros(nbArms, dtype=int)

    def store(self, time, choice, reward):
        self.choices[time] = choice
        self.rewards[time] = reward
        self.rewardsSquared[time] = reward ** 2
        self.pulls[choice] += 1
