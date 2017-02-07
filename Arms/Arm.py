# -*- coding: utf-8 -*-
""" Base class for an arm class."""

__author__ = "Lilian Besson"
__version__ = "0.1"


class Arm(object):
    """ Base class for an arm class."""

    # --- Printing

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dir__)

    # --- Random samples

    def draw(self, t=None):
        raise NotImplementedError("This method draw(t) has to be implemented in the class inheriting from Arm.")

    def draw_nparray(self, shape=(1,)):
        raise NotImplementedError("This method draw_nparray(t) has to be implemented in the class inheriting from Arm.")

    def mean(self):
        raise NotImplementedError("This method mean() has to be implemented in the class inheriting from Arm.")

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        raise NotImplementedError("This method kl(x, y) has to be implemented in the class inheriting from Arm.")

    @staticmethod
    def oneLR(mumax, mu):
        raise NotImplementedError("This method oneLR(mumax, mu) has to be implemented in the class inheriting from Arm.")
