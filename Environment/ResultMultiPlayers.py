# -*- coding: utf-8 -*-
""" ResultMultiPlayers.ResultMultiPlayers class to wrap the simulation results, for the multi-players case."""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np


class ResultMultiPlayers(object):
    """ ResultMultiPlayers accumulators, for the multi-players case. """

    def __init__(self, nbArms, horizon, nbPlayers):
        self.choices = np.zeros((nbPlayers, horizon), dtype=int)
        self.rewards = np.zeros((nbPlayers, horizon))         # To compute the mean
        self.rewardsSquared = np.zeros((nbPlayers, horizon))  # To compute the variance
        self.pulls = np.zeros((nbPlayers, nbArms), dtype=int)
        self.allPulls = np.zeros((nbPlayers, nbArms, horizon), dtype=int)
        self.collisions = np.zeros((nbArms, horizon), dtype=int)

    def store(self, time, choices, rewards, pulls, collisions):
        self.choices[:, time] = choices
        self.rewards[:, time] = rewards
        self.rewardsSquared[:, time] = rewards ** 2
        self.pulls += pulls
        self.allPulls[:, :, time] = pulls
        self.collisions[:, time] = collisions
