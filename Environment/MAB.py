# -*- coding: utf-8 -*-
""" MAB.MAB class to wrap the arms."""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.26 $"


class MAB:
    """ Multi-armed Bandit environment."""

    def __init__(self, configuration):
        arm_type = configuration["arm_type"]
        params = configuration["params"]
        # Each 'param' could be one value (eg. 'mean' = probability for a Bernoulli) or a tuple (eg. '(mu, sigma)' for a Gaussian)
        self.arms = [arm_type(param) for param in params]
        self.nbArms = len(self.arms)
        self.maxArm = max([arm.mean() for arm in self.arms])

    def __repr__(self):
        return '<' + self.__class__.__name__ + repr(self.__dict__) + '>'
