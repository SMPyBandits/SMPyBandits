# -*- coding: utf-8 -*-
""" Basic Bayesian index policy. By default, it uses a Beta posterior. """

__author__ = "Lilian Besson"
__version__ = "0.3"

from .IndexPolicy import IndexPolicy
from .Posterior import Beta


class BayesianIndexPolicy(IndexPolicy):
    """ Basic Bayesian index policy. By default, it uses a Beta posterior. """

    def __init__(self, nbArms, posterior=Beta, lower=0., amplitude=1., *args, **kwargs):
        super(BayesianIndexPolicy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self._posterior_name = str(posterior.__class__.__name__)
        self.posterior = [None] * nbArms  #: Posterior for each arm. List instead of dict, quicker access
        for arm in range(self.nbArms):
            self.posterior[arm] = posterior(*args, **kwargs)

    def __str__(self):
        """ -> str"""
        if self._posterior_name == "Beta":
            return "{}".format(self.__class__.__name__)
        else:
            return "{}({})".format(self.__class__.__name__, self._posterior_name)

    def startGame(self):
        self.t = 0
        for arm in range(self.nbArms):
            self.posterior[arm].reset()

    def getReward(self, arm, reward):
        self.posterior[arm].update((reward - self.lower) / self.amplitude)
        self.t += 1

    def computeIndex(self, arm):
        raise NotImplementedError("This method computeIndex(arm) has to be implemented in the child class inheriting from BayesianIndexPolicy.")
