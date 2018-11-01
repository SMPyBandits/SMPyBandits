# -*- coding: utf-8 -*-
""" Basic Bayesian index policy. By default, it uses a Beta posterior. """
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

try:
    from .IndexPolicy import IndexPolicy
    from .Posterior import Beta
except ImportError:
    from IndexPolicy import IndexPolicy
    from Posterior import Beta


class BayesianIndexPolicy(IndexPolicy):
    """ Basic Bayesian index policy.

    - By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
    - Use ``*args`` and ``**kwargs`` if you want to give parameters to the underlying posteriors.
    - Or use ``params_for_each_posterior`` as a *list* of parameters (as a dictionary) to give a different set of parameters for each posterior.
    """

    def __init__(self, nbArms,
            posterior=Beta,
            lower=0., amplitude=1.,
            *args, **kwargs
        ):
        """ Create a new Bayesian policy, by creating a default posterior on each arm."""
        super(BayesianIndexPolicy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.posterior = [None] * nbArms  #: Posterior for each arm. List instead of dict, quicker access
        if 'params_for_each_posterior' in kwargs:
            params = kwargs['params_for_each_posterior']
            print("'params_for_each_posterior' is in kwargs, so using params =\n{}\nas a list of parameters to give to each posterior.".format(params))  # DEBUG
            for arm in range(self.nbArms):
                print("Creating posterior for arm {}, with params = {}.".format(arm, params[arm]))  # DEBUG
                self.posterior[arm] = posterior(**params[arm])
        else:
            for arm in range(self.nbArms):
                # print("Creating posterior for arm {}, with args = {} and kwargs = {}.".format(arm, args, kwargs))  # DEBUG
                self.posterior[arm] = posterior(*args, **kwargs)
        self._posterior_name = str(self.posterior[0].__class__.__name__)

    def __str__(self):
        """ -> str"""
        if self._posterior_name == "Beta":
            return "{}".format(self.__class__.__name__)
        else:
            return "{}({})".format(self.__class__.__name__, self._posterior_name)

    def startGame(self):
        """ Reset the posterior on each arm."""
        self.t = 0
        for arm in range(self.nbArms):
            self.posterior[arm].reset()
        # print("Policy {} reinitialized with posteriors: {}".format(self, [str(p) for p in self.posterior])) # DEBUG

    def getReward(self, arm, reward):
        """ Update the posterior on each arm, with the normalized reward."""
        self.posterior[arm].update((reward - self.lower) / self.amplitude)
        self.t += 1

    def computeIndex(self, arm):
        raise NotImplementedError("This method computeIndex(arm) has to be implemented in the child class inheriting from BayesianIndexPolicy.")
