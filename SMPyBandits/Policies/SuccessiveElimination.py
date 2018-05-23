# -*- coding: utf-8 -*-
""" Generic policy based on successive elimination, mostly useless except to maintain a clear hierarchy of inheritance.
"""

__author__ = "Lilian Besson"
__version__ = "0.9"

from numpy import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .IndexPolicy import IndexPolicy
except ImportError:
    from IndexPolicy import IndexPolicy


class SuccessiveElimination(IndexPolicy):
    """ Generic policy based on successive elimination, mostly useless except to maintain a clear hierarchy of inheritance.
    """

    def choice(self):
        r""" In policy based on successive elimination, choosing an arm is the same as choosing an arm from the set of active arms (``self.activeArms``) with method ``choiceFromSubSet``.
        """
        return self.choiceFromSubSet(self.activeArms)
