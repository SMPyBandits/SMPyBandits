# -*- coding: utf-8 -*-
""" ResultMultiPlayers.ResultMultiPlayers class to wrap the simulation results, for the multi-players case."""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np


class ResultMultiPlayers(object):
    """ ResultMultiPlayers accumulators, for the multi-players case. """

    # , delta_t_save=1
    def __init__(self, nbArms, horizon, nbPlayers, means=None):
        """ Create ResultMultiPlayers."""
        self._means = means  # Keep the means for DynamicMAB cases
        self.choices = np.zeros((nbPlayers, horizon), dtype=int)  #: Store all the choices of all the players
        self.rewards = np.zeros((nbPlayers, horizon))  #: Store all the rewards of all the players, to compute the mean
        # self.rewardsSquared = np.zeros((nbPlayers, horizon))  #: Store all the rewards**2 of all the players, to compute the variance  # XXX uncomment if needed
        self.pulls = np.zeros((nbPlayers, nbArms), dtype=int)  #: Store the pulls of all the players
        self.allPulls = np.zeros((nbPlayers, nbArms, horizon), dtype=int)  #: Store all the pulls of all the players
        self.collisions = np.zeros((nbArms, horizon), dtype=int)  #: Store the collisions on all the arms

    def store(self, time, choices, rewards, pulls, collisions):
        """ Store results."""
        self.choices[:, time] = choices
        self.rewards[:, time] = rewards
        # self.rewardsSquared[:, time] = rewards ** 2  # XXX uncomment if needed
        self.pulls += pulls
        self.allPulls[:, :, time] = pulls
        self.collisions[:, time] = collisions

    # def saveondisk(self, filepath='/tmp/saveondisk.hdf5', delta_t_save=None):
    #     """ Save the content of the result files into a HDF5 file on the disk."""
    #     if delta_t_save is None:
    #         delta_t_save = self.delta_t_save
    #     raise ValueError("FIXME finish to write this function saveondisk() for ResultMultiPlayers!")
