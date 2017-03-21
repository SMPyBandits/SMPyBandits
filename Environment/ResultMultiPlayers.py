# -*- coding: utf-8 -*-
""" ResultMultiPlayers.ResultMultiPlayers class to wrap the simulation results, for the multi-players case."""

__author__ = "Lilian Besson"
__version__ = "0.5"

import numpy as np


class ResultMultiPlayers(object):
    """ ResultMultiPlayers accumulators, for the multi-players case. """

    def __init__(self, nbArms, horizon, nbPlayers, delta_t_save=1):
        """Create ResultMultiPlayers."""
        horizon = horizon
        self.delta_t_save = delta_t_save
        self.choices = np.zeros((nbPlayers, horizon), dtype=int)
        self.rewards = np.zeros((nbPlayers, horizon))         # To compute the mean
        # self.rewardsSquared = np.zeros((nbPlayers, horizon))  # To compute the variance  # XXX uncomment if needed
        self.pulls = np.zeros((nbPlayers, nbArms), dtype=int)
        self.allPulls = np.zeros((nbPlayers, nbArms, horizon), dtype=int)
        self.collisions = np.zeros((nbArms, horizon), dtype=int)

    def store(self, time, choices, rewards, pulls, collisions):
        """Store results."""
        self.choices[:, time] = choices
        self.rewards[:, time] = rewards
        # self.rewardsSquared[:, time] = rewards ** 2  # XXX uncomment if needed
        self.pulls += pulls
        self.allPulls[:, :, time] = pulls
        self.collisions[:, time] = collisions

    def saveondisk(self, filepath='/tmp/saveondisk.hdf5', delta_t_save=None):
        """Save the content of the result files into a HDF5 file on the disk."""
        # FIXME write it !
        if delta_t_save is None:
            delta_t_save = self.delta_t_save
        raise ValueError("FIXME finish to write this function saveondisk() for ResultMultiPlayers!")
