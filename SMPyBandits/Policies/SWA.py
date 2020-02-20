# -*- coding: utf-8 -*-
"""
author : Julien Seznec
Sliding Window Average policy for rotting bandits.

Reference: [Levine et al., 2017, https://papers.nips.cc/paper/6900-rotting-bandits.pdf].
Advances in Neural Information Processing Systems 30 (NIPS 2017)
Nir Levine, Koby Crammer, Shie Mannor
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Julien Seznec"
__version__ = "0.1"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .IndexPolicy import IndexPolicy
except ImportError:
    from IndexPolicy import IndexPolicy


class SWA(IndexPolicy):
    """ The Sliding Window Average policy for rotting bandits.
    Reference: [Levine et al., 2017, https://papers.nips.cc/paper/6900-rotting-bandits.pdf].
    """
    def __init__(self, nbArms, horizon=1, subgaussian=1, maxDecrement=1,alpha=0.2, doublingTrick=False):
        super(SWA, self).__init__(nbArms)
        self.t = 0
        self.nbArms = nbArms
        self.armSet = set(range(nbArms))
        self.horizon = horizon
        self.starting_horizon = horizon
        self.alpha = alpha if alpha is not None else (2*maxDecrement)**(-2/3)
        self.subgaussian = subgaussian
        self.h = self.setWindow()
        self.arms_history = {arm: np.full(self.h, np.inf) for arm in range(nbArms)}
        self.doubling = doublingTrick

    def setWindow(self):
        return int(np.ceil(self.alpha * (16 * self.subgaussian**2 * self.horizon**2 * np.log(np.sqrt(2)*self.horizon)/ len(self.armSet)**2)**(1/3)))

    def getReward(self, arm, reward):
        super(SWA, self).getReward(arm, reward)
        if self.h >1:
            self.arms_history[arm][1:] = self.arms_history[arm][:-1] + reward
        self.arms_history[arm][0] = reward

    def computeIndex(self, arm):
        """ Compute the mean of the h last value """
        return self.arms_history[arm][-1]

    def startGame(self, resetHorizon = True):
        super(SWA, self).startGame()
        self.arms_history = {arm: np.full(self.h, np.inf) for arm in range(self.nbArms)}
        if resetHorizon:
            self.horizon = self.starting_horizon


class wSWA(SWA):
    """ SWA with doubling trick
    Reference: [Levine et al., 2017, https://papers.nips.cc/paper/6900-rotting-bandits.pdf].
    """
    def __init__(self, nbArms, firstHorizon=1, subgaussian=1, maxDecrement=1, alpha=0.2):
        super(wSWA, self).__init__(nbArms, firstHorizon, subgaussian, maxDecrement, alpha)

    def __str__(self):
        return r"wSWA($\alpha={:.3g}$)".format(self.alpha)

    def doublingTrick(self):
        self.horizon *= 2
        self.h = self.setWindow()
        self.t=0
        self.startGame(resetHorizon=False)

    def getReward(self, arm, reward):
        super(wSWA, self).getReward(arm,reward)
        if self.t >= self.horizon:
            self.doublingTrick()

# --- Debugging
if __name__ == "__main__":
    # Code for debugging purposes.
    reward = {0:0, 1:0.2, 2:0.4, 3:0.6, 4:0.8}
    policy = wSWA(5)
    for t in range(1000):
        choice = policy.choice()
        policy.getReward(choice, reward[choice])
    print(policy.pulls)