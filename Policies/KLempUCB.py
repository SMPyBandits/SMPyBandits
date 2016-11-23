# -*- coding: utf-8 -*-
""" The Empirical KL-UCB algorithm non-parametric policy.
Reference: [Maillard, Munos & Stoltz - COLT, 2011], [Cappé, Garivier,  Maillard, Munos & Stoltz, 2012].
"""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.8 $"

from math import log
import numpy as np

from .kullback import maxEV
from .IndexPolicy import IndexPolicy


class KLempUCB(IndexPolicy):
    """ The Empirical KL-UCB algorithm non-parametric policy.
    References: [Maillard, Munos & Stoltz - COLT, 2011], [Cappé, Garivier,  Maillard, Munos & Stoltz, 2012].
    """

    def __init__(self, nbArms, maxReward=1.):
        super(KLempUCB, self).__init__(nbArms)
        self.c = 1
        self.maxReward = maxReward
        self.obs = [None] * nbArms  # List instead of dict, quicker access
        self.params = 'maxReward: ' + repr(maxReward)

    def startGame(self):
        self.t = 0
        self.pulls = np.zeros(self.nbArms)
        for arm in range(self.nbArms):
            self.obs[arm] = {self.maxReward: 0}

    def computeIndex(self, arm):
        if self.pulls[arm] > 0:
            return self._KLucb(self.obs[arm], self.c * log(self.t) / self.pulls[arm])
        else:
            return float('+infinity')

    def getReward(self, arm, reward):
        self.t += 1
        self.pulls[arm] += 1
        self.obs[arm][reward] = 1 + self.obs[arm].get(reward, 0)

    # FIXME this does not work apparently...
    def _KLucb(self, obs, klMax, debug=False):
        p = np.array(list(obs.values()))
        p = p / np.sum(p)
        v = np.array(obs.keys())
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
