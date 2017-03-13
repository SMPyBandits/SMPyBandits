# -*- coding: utf-8 -*-
""" The naive Empirical Means policy for bounded bandits: like UCB but without a bias correction term."""

__author__ = "Lilian Besson"
__version__ = "0.1"

from .IndexPolicy import IndexPolicy


class EmpiricalMeans(IndexPolicy):
    """ The naive Empirical Means policy for bounded bandits: like UCB but without a bias correction term."""

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            return self.rewards[arm] / self.pulls[arm]
