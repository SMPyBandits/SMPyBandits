# -*- coding: utf-8 -*-
""" Base class for a posterior. Cf. http://chercheurs.lille.inria.fr/ekaufman/NIPS13 Fig.1 for a list of posteriors. """
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"


class Posterior(object):
    """ Manipulate posteriors experiments."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This method __init__(self, *args, **kwargs) has to be implemented in the child class inheriting from Posterior.")

    def reset(self, *args, **kwargs):
        """Reset posterior, new experiment."""
        raise NotImplementedError("This method reset(self, *args, **kwargs) has to be implemented in the child class inheriting from Posterior.")

    def sample(self):
        """Sample from the posterior."""
        raise NotImplementedError("This method sample(self) has to be implemented in the child class inheriting from Posterior.")

    def quantile(self, p):
        """p quantile from the posterior."""
        raise NotImplementedError("This method quantile(self, p) has to be implemented in the child class inheriting from Posterior.")

    def mean(self):
        """Mean of the posterior."""
        raise NotImplementedError("This method mean(self) has to be implemented in the child class inheriting from Posterior.")

    def forget(self, obs):
        """Forget last observation (never used)."""
        raise NotImplementedError("This method forget(self, obs) has to be implemented in the child class inheriting from Posterior.")

    def update(self, obs):
        """Update posterior with this observation."""
        raise NotImplementedError("This method update(self, obs) has to be implemented in the child class inheriting from Posterior.")
