# -*- coding: utf-8 -*-
""" Result.Result class to wrap the simulation results."""

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.5"

import numpy as np


class Result(object):
    """ Result accumulators"""

    def __init__(self, nbArms, horizon, delta_t_save=1):
        """ Create ResultMultiPlayers."""
        self.delta_t_save = delta_t_save  #: Sample rate for saving
        self.choices = np.zeros(horizon, dtype=int)  #: Store all the choices of all the players
        self.rewards = np.zeros(horizon)  #: Store all the rewards of all the players, to compute the mean
        self.pulls = np.zeros(nbArms, dtype=int)  #: Store the pulls of all the players

    def store(self, time, choice, reward):
        """ Store results."""
        self.choices[time] = choice
        self.rewards[time] = reward
        self.pulls[choice] += 1

    def saveondisk(self, filepath='/tmp/saveondisk.hdf5', delta_t_save=None):
        """ Save the content of the result files into a HDF5 file on the disk."""
        # FIXME write it !
        if delta_t_save is None:
            delta_t_save = self.delta_t_save
        raise ValueError("FIXME finish to write this function saveondisk() for ResultMultiPlayers!")
