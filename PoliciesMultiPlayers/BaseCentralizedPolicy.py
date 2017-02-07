# -*- coding: utf-8 -*-
""" Base class for any centralized policy, for the multi-players setting."""

__author__ = "Lilian Besson"
__version__ = "0.3"


class BaseCentralizedPolicy(object):
    """ Base class for any centralized policy, for the multi-players setting."""

    def __init__(self, nbArms):
        self.nbArms = nbArms

    def __str__(self):
        return self.__class__.__name__

    def startGame(self):
        raise NotImplementedError("This method startGame() has to be implemented in the child class inheriting from BaseCentralizedPolicy.")

    def getReward(self, arm, reward):
        raise NotImplementedError("This method getReward(arm, reward) has to be implemented in the child class inheriting from BaseCentralizedPolicy.")

    def choice(self):
        raise NotImplementedError("This method choice() has to be implemented in the child class inheriting from BaseCentralizedPolicy.")
