# -*- coding: utf-8 -*-
""" The Empirical KL-UCB algorithm non-parametric policy.
Reference: [Maillard, Munos & Stoltz - COLT, 2011], [Cappé, Garivier,  Maillard, Munos & Stoltz, 2012].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.1"

import numpy as np

from .kullback import maxEV   # XXX Not detected as in the kullback.py file ?
from .IndexPolicy import IndexPolicy


class KLempUCB(IndexPolicy):
    """ The Empirical KL-UCB algorithm non-parametric policy.
    References: [Maillard, Munos & Stoltz - COLT, 2011], [Cappé, Garivier,  Maillard, Munos & Stoltz, 2012].
    """

    def __init__(self, nbArms, maxReward=1., lower=0., amplitude=1.):
        super(KLempUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.c = 1  #: Parameter c
        self.maxReward = maxReward  #: Known upper bound on the rewards
        self.pulls = np.zeros(self.nbArms, dtype=int)  #: Keep track of pulls of each arm
        #: UNBOUNDED dictionnary for each arm: keep track of how many observation of each rewards were seen.
        #: Warning: KLempUCB works better for *discrete* distributions!
        self.obs = [dict()] * self.nbArms

    def startGame(self):
        """ Initialize the policy for a new game."""
        self.t = 0
        self.pulls.fill(0)
        for arm in range(self.nbArms):
            self.obs[arm] = {self.maxReward: 0}

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k."""
        if self.pulls[arm] < 1:
            return float('+infinity')
        else:
            return self._KLucb(self.obs[arm], self.c * np.log(self.t) / self.pulls[arm])

    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update count of observations for that arm."""
        self.t += 1
        self.pulls[arm] += 1
        self.obs[arm][reward] = 1 + self.obs[arm].get(reward, 0)

    # FIXME this does not work apparently...
    @staticmethod
    def _KLucb(obs, klMax, debug=False):
        """ Optimization method."""
        p = np.array(list(obs.values()), dtype=float)
        p /= np.sum(p)
        v = np.array(list(obs.keys()), dtype=float)
        if debug:
            print("Calling maxEV(", p, ", ", v, ", ", klMax, ") ...")
        q = maxEV(p, v, klMax)
        # if debug:
        #     q2 = kbp.maxEV(p, v, klMax)
        #     if max(abs(q - q2)) > 1e-8:
        #         print("ERROR: for p=", p, " ,v = ", v, " and klMax = ", klMax, " : ")
        #         print("q = ", q)
        #         print("q2 = ", q2)
        #         print("_____________________________")
        #         print("q = ", q)
        return np.dot(q, v)
