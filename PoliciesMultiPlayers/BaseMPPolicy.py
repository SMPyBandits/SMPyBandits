# -*- coding: utf-8 -*-
""" Base class for any multi-players policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""

__author__ = "Lilian Besson"
__version__ = "0.3"


class BaseMPPolicy(object):
    """ Base class for any multi-players policy."""

    def __init__(self):
        pass

    def __str__(self):
        return self.__class__.__name__
