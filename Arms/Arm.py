# -*- coding: utf-8 -*-
""" Basis class for an arm."""

__author__ = "Lilian Besson"
__version__ = "0.1"


class Arm(object):
    """ Basis class for an arm."""

    def __init__(self):
        pass

    def __str__(self):
        return "Arm"

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        raise NotImplementedError("This method draw() has to be implemented in the child class.")

    def mean(self):
        raise NotImplementedError("This method mean() has to be implemented in the child class.")

    def __repr__(self):
        return self.__class__.__name__ + "(" + repr(self.__dir__) + ")"
