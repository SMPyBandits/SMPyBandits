# -*- coding: utf-8 -*-
""" Base class for any centralized policy, for the multi-players setting."""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"


class BaseCentralizedPolicy(object):
    """ Base class for any centralized policy, for the multi-players setting."""

    def __init__(self, nbArms):
        """ New policy"""
        self.nbArms = nbArms

    def __str__(self):
        return self.__class__.__name__

    def startGame(self):
        """ Start the simulation."""
        raise NotImplementedError("This method startGame() has to be implemented in the child class inheriting from BaseCentralizedPolicy.")

    def getReward(self, arm, reward):
        """ Get a reward from that arm."""
        raise NotImplementedError("This method getReward(arm, reward) has to be implemented in the child class inheriting from BaseCentralizedPolicy.")

    def choice(self):
        """ Chose an arm."""
        raise NotImplementedError("This method choice() has to be implemented in the child class inheriting from BaseCentralizedPolicy.")
