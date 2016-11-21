# -*- coding: utf-8 -*-
""" The Emipirical KL-UCB algorithm non-parametric policy.
Reference: [Maillard, Munos & Stoltz - COLT, 2011], [Cappé, Garivier,  Maillard, Munos & Stoltz, 2012].
"""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.8 $"

from math import log
import numpy as np
from .kullback import maxEV
from .IndexPolicy import IndexPolicy


class KLempUCB(IndexPolicy):
    """ The Emipirical KL-UCB algorithm non-parametric policy.
    References: [Maillard, Munos & Stoltz - COLT, 2011], [Cappé, Garivier,  Maillard, Munos & Stoltz, 2012].
    """

    def __init__(self, nbArms, maxReward=1.):
        self.c = 1
        self.nbArms = nbArms
        self.maxReward = maxReward
        self.nbpulls = np.zeros(nbArms)
        self.obs = dict()

    def startGame(self):
        self.t = 1
        self.nbpulls = np.zeros(self.nbArms)
        for arm in range(self.nbArms):
            self.obs[arm] = {self.maxReward: 0}

    def computeIndex(self, arm):
        if self.nbpulls[arm] > 0:
            return self.KLucb(self.obs[arm], self.c * log(self.t) / self.nbpulls[arm])
        else:
            return float('+infinity')

    def getReward(self, arm, reward):
        self.t += 1
        self.nbpulls[arm] += 1
        # if self.obs[arm].has_key(reward):
        #     self.obs[arm][reward] += 1
        # else:
        #     self.obs[arm][reward] = 1
        self.obs[arm][reward] = 1 + self.obs[arm].get(reward, 0)

    def KLucb(self, obs, klMax):
        p = (np.array(obs.values()) + 0.) / sum(obs.values())
        v = np.array(obs.keys())
        # print("calling maxEV(", p, ", ", v, ", ", klMax, ")")
        q = maxEV(p, v, klMax)
        # q2 = kbp.maxEV(p, v, klMax)
        # if max(abs(q-q2))>1e-8:
        #    print("ERROR: for p=", p, " ,v = ", v, " and klMax = ", klMax, " : ")
        #    print("q = ", q))
        #    print("q2 = ", q2))
        #    print("_____________________________")
        # print("q = ", q))
        return np.dot(q, v)
