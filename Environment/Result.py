# -*- coding: utf-8 -*-
""" Result.Result class to wrap the simulation results."""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.7"

import numpy as np


class Result(object):
    """ Result accumulators."""

    # , delta_t_save=1):
    def __init__(self, nbArms, horizon, indexes_bestarm=-1, means=None):
        """ Create ResultMultiPlayers."""
        self._means = means  # Keep the means for DynamicMAB cases
        # self.delta_t_save = delta_t_save  #: Sample rate for saving
        self.choices = np.zeros(horizon, dtype=int)  #: Store all the choices
        self.rewards = np.zeros(horizon)  #: Store all the rewards, to compute the mean
        self.pulls = np.zeros(nbArms, dtype=int)  #: Store the pulls
        if means is not None:
            indexes_bestarm = np.nonzero(np.isclose(means, np.max(means)))[0]
        indexes_bestarm = np.asarray(indexes_bestarm)
        if np.size(indexes_bestarm) == 1:
            indexes_bestarm = np.asarray([indexes_bestarm])
        self.indexes_bestarm = [ indexes_bestarm for _ in range(horizon)]  #: Store also the position of the best arm, XXX in case of dynamically switching environment.

    def store(self, time, choice, reward):
        """ Store results."""
        self.choices[time] = choice
        self.rewards[time] = reward
        self.pulls[choice] += 1

    def change_in_arms(self, time, indexes_bestarm):
        """ Store the position of the best arm from this list of arm.

        - From that time t **and after**, the index of the best arm is stored as ``indexes_bestarm``.
        """
        for t in range(time, len(self.indexes_bestarm)):
            self.indexes_bestarm[t] = indexes_bestarm

    # def saveondisk(self, filepath='/tmp/saveondisk.hdf5', delta_t_save=None):
    #     """ Save the content of the Result object into a HDF5 file on the disk."""
    #     if delta_t_save is None:
    #         delta_t_save = self.delta_t_save
    #     raise ValueError("FIXME finish to write this function saveondisk() for Result!")
