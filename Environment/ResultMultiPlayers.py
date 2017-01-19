# -*- coding: utf-8 -*-
""" ResultMultiPlayers.ResultMultiPlayers class to wrap the simulation results, for the multi-players case."""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np


class ResultMultiPlayers(object):
    """ ResultMultiPlayers accumulators, for the multi-players case. """

    def __init__(self, nbArms, horizon, nbPlayers, delta_t_save=1):
        duration = int(horizon / delta_t_save)
        self.delta_t_save = delta_t_save
        self.choices = np.zeros((nbPlayers, duration), dtype=int)
        self.rewards = np.zeros((nbPlayers, duration))         # To compute the mean
        # self.rewardsSquared = np.zeros((nbPlayers, duration))  # To compute the variance
        self.pulls = np.zeros((nbPlayers, nbArms), dtype=int)
        self.allPulls = np.zeros((nbPlayers, nbArms, duration), dtype=int)
        self.collisions = np.zeros((nbArms, duration), dtype=int)

    def store(self, time, choices, rewards, pulls, collisions):
        time = int(time / self.delta_t_save)
        self.choices[:, time] = choices
        self.rewards[:, time] = rewards
        # self.rewardsSquared[:, time] = rewards ** 2
        self.pulls += pulls
        self.allPulls[:, :, time] = pulls
        self.collisions[:, time] = collisions
