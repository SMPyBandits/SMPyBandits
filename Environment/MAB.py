# -*- coding: utf-8 -*-
""" MAB.MAB class to wrap the arms."""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.26 $"


class MAB:
    """ Multi-armed Bandit environment.

    - configuration has to be a dict with 'arm_type' and 'params' keys.
    - 'arm_type' is a class from the Arms module
    - 'params' is a dict, used as a list/tuple/iterable of named parameters given to 'arm_type'.

    Example:

        configuration = {
            'arm_type': Bernoulli,
            'params':   [0.1, 0.5, 0.9]
        }

    It will create three Bernoulli arms, of parameters (means) 0.1, 0.5 and 0.9.
    """

    def __init__(self, configuration):
        print("Creating a new MAB problem ...")  # DEBUG
        if isinstance(configuration, dict):
            print("  Reading arms of this MAB problem from a dictionnary 'configuration' = {} ...".format(configuration))  # DEBUG
            arm_type = configuration["arm_type"]
            print(" - with 'arm_type' =", arm_type)  # DEBUG
            params = configuration["params"]
            print(" - with 'params' =", params)  # DEBUG
            # Each 'param' could be one value (eg. 'mean' = probability for a Bernoulli) or a tuple (eg. '(mu, sigma)' for a Gaussian)
            # XXX manually detect if the parameters are iterable or not
            self.arms = []
            for param in params:
                try:
                    self.arms.append(arm_type(*param))
                except TypeError:
                    self.arms.append(arm_type(param))
            # self.arms = [arm_type(*param) for param in params]
        else:
            print("  Taking arms of this MAB problem from a list of arms 'configuration' = {} ...".format(configuration))  # DEBUG
            self.arms = []
            for arm in configuration:
                self.arms.append(arm)
        print(" - with 'arms' =", self.arms)  # DEBUG
        self.nbArms = len(self.arms)
        print(" - with 'nbArms' =", self.nbArms)  # DEBUG
        self.maxArm = max([arm.mean() for arm in self.arms])
        print(" - with 'maxArm' =", self.maxArm)  # DEBUG

    def __repr__(self):
        return '<' + self.__class__.__name__ + repr(self.__dict__) + '>'

    def complexity(self):
        """ Compute the [Lai & Robbins] lower bound for this MAB problem (complexity), using functions from kullback.py or kullback.so. """
        raise NotImplementedError("Error, the method complexity() of the MAB class is not implemented yet. FIXME do it!")
