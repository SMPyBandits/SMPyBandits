# -*- coding: utf-8 -*-
""" Result.Result class to wrap the simulation results."""

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np


class Result(object):
    """ Result accumulators."""

    # , delta_t_save=1):
    def __init__(self, nbArms, horizon, index_bestarm=-1):
        """ Create ResultMultiPlayers."""
        # self.delta_t_save = delta_t_save  #: Sample rate for saving
        self.choices = np.zeros(horizon, dtype=int)  #: Store all the choices of all the players
        self.rewards = np.zeros(horizon)  #: Store all the rewards of all the players, to compute the mean
        self.pulls = np.zeros(nbArms, dtype=int)  #: Store the pulls of all the players
        self.indeces_bestarm = np.full(horizon, index_bestarm)  #: Store also the position of the best arm, XXX in case of dynamically switching environment.

    def store(self, time, choice, reward):
        """ Store results."""
        self.choices[time] = choice
        self.rewards[time] = reward
        self.pulls[choice] += 1

    def change_in_arms(self, time, index_bestarm):
        """ Store the position of the best arm from this list of arm.

        - From that time t **and after**, the index of the best arm is stored as ``index_bestarm``.
        """
        self.indeces_bestarm[time:] = index_bestarm

    # def saveondisk(self, filepath='/tmp/saveondisk.hdf5', delta_t_save=None):
    #     """ Save the content of the Result object into a HDF5 file on the disk."""
    #     # FIXME write it !
    #     if delta_t_save is None:
    #         delta_t_save = self.delta_t_save
    #     raise ValueError("FIXME finish to write this function saveondisk() for Result!")
