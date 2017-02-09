# -*- coding: utf-8 -*-
""" Result.Result class to wrap the simulation results."""

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.5"

import numpy as np


class Result(object):
    """ Result accumulators"""

    def __init__(self, nbArms, horizon, delta_t_save=1):
        """Create ResultMultiPlayers."""
        duration = int(horizon / delta_t_save)
        self.delta_t_save = delta_t_save
        self.choices = np.zeros(duration, dtype=int)
        self.rewards = np.zeros(duration)         # To compute the mean
        # self.rewardsSquared = np.zeros(duration)  # To compute the variance  # XXX uncomment if needed
        self.pulls = np.zeros(nbArms, dtype=int)

    def store(self, time, choice, reward):
        """Store results."""
        time = int(time / self.delta_t_save)
        self.choices[time] = choice
        self.rewards[time] = reward
        # self.rewardsSquared[time] = reward ** 2  # XXX uncomment if needed
        self.pulls[choice] += 1
        # FIXME find a way to store the result while learning?
