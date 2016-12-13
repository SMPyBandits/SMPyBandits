# -*- coding: utf-8 -*-
""" Base class for a posterior. """

__author__ = "Lilian Besson"
__version__ = "0.3"


class Posterior(object):
    """ Manipulate posteriors experiments."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This method __init__(self, *args, **kwargs) has to be implemented in the child class inheriting from Posterior.")

    def reset(self, *args, **kwargs):
        raise NotImplementedError("This method reset(self, *args, **kwargs) has to be implemented in the child class inheriting from Posterior.")

    def sample(self):
        raise NotImplementedError("This method sample(self) has to be implemented in the child class inheriting from Posterior.")

    def quantile(self, p):
        raise NotImplementedError("This method quantile(self, p) has to be implemented in the child class inheriting from Posterior.")

    def mean(self):
        raise NotImplementedError("This method mean(self) has to be implemented in the child class inheriting from Posterior.")

    def forget(self, obs):
        raise NotImplementedError("This method forget(self, obs) has to be implemented in the child class inheriting from Posterior.")

    def update(self, obs):
        raise NotImplementedError("This method update(self, obs) has to be implemented in the child class inheriting from Posterior.")
