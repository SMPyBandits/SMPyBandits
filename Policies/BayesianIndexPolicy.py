# -*- coding: utf-8 -*-
""" Basic Bayesian index policy. By default, it uses a Beta posterior. """

__author__ = "Lilian Besson"
__version__ = "0.3"

from .IndexPolicy import IndexPolicy
from .Beta import Beta


class BayesianIndexPolicy(IndexPolicy):
    """ Basic Bayesian index policy. By default, it uses a Beta posterior. """

    def __init__(self, nbArms, posterior=Beta, lower=0., amplitude=1.):
        super(BayesianIndexPolicy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.posterior = [None] * nbArms  # List instead of dict, quicker access
        for arm in range(self.nbArms):
            self.posterior[arm] = posterior()

    def startGame(self):
        self.t = 0
        for arm in range(self.nbArms):
            self.posterior[arm].reset()

    def getReward(self, arm, reward):
        self.posterior[arm].update((reward - self.lower) / self.amplitude)
        self.t += 1

    def computeIndex(self, arm):
        raise NotImplementedError("This method computeIndex(arm) has to be implemented in the child class inheriting from BayesianIndexPolicy.")
