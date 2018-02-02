# -*- coding: utf-8 -*-
""" The MOSS-Experimental policy for bounded bandits, without knowing the horizon (and no doubling trick).
Reference: [Degenne & Perchet, 2016](http://proceedings.mlr.press/v48/degenne16.pdf).

.. warning:: Nothing was proved for this heuristic!
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from numpy import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .MOSS import MOSS


class MOSSExperimental(MOSS):
    """ The MOSS-Experimental policy for bounded bandits, without knowing the horizon (and no doubling trick).
    Reference: [Degenne & Perchet, 2016](http://proceedings.mlr.press/v48/degenne16.pdf).
    """

    def __str__(self):
        return "MOSS-Experimental"

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, if there is K arms:

        .. math::

            I_k(t) &= \frac{X_k(t)}{N_k(t)} + \sqrt{ \max\left(0, \frac{\log\left(\frac{t}{\hat{H}(t)}\right)}{N_k(t)}\right)},\\
            \text{where}\;\; \hat{H}(t) &:= \begin{cases}
                \sum\limits_{j=1, N_j(t) < \sqrt{t}}^{K} N_j(t) & \;\text{if it is}\; > 0,\\
                K N_k(t) & \;\text{otherwise}\;
            \end{cases}

        .. note:: In the article, the authors do not explain this subtlety, and I don't see an argument to justify that at anytime, :math:`\hat{H}(t) > 0` ie to justify that there is always some arms :math:`j` such that :math:`0 < N_j(t) < \sqrt{t}`.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            pulls_of_suboptimal_arms = np.sum(self.pulls[self.pulls < np.sqrt(self.t)])
            if pulls_of_suboptimal_arms > 0:
                return (self.rewards[arm] / self.pulls[arm]) + np.sqrt(0.5 * max(0, np.log(self.t / pulls_of_suboptimal_arms)) / self.pulls[arm])
            else:
                return (self.rewards[arm] / self.pulls[arm]) + np.sqrt(0.5 * max(0, np.log(self.t / (self.nbArms * self.pulls[arm]))) / self.pulls[arm])

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        pulls_of_suboptimal_arms = np.sum(self.pulls[self.pulls < np.sqrt(self.t)])
        if pulls_of_suboptimal_arms > 0:
            indexes = (self.rewards / self.pulls) + np.sqrt(0.5 * np.maximum(0, np.log(self.t / pulls_of_suboptimal_arms)) / self.pulls)
        else:
            indexes = (self.rewards / self.pulls) + np.sqrt(0.5 * np.maximum(0, np.log(self.t / (self.nbArms * self.pulls))) / self.pulls)
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
