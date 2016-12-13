# -*- coding: utf-8 -*-
""" The AdBandits bandit algorithm
Reference: https://github.com/flaviotruzzi/AdBandits/
"""
from __future__ import print_function

__author__ = "Flavio Truzzi et al."
__version__ = "0.1"

import random as rn
import numpy as np
from .Beta import Beta


class AdBandits(object):
    """ The AdBandits bandit algorithm
    Reference: https://github.com/flaviotruzzi/AdBandits/
    """

    def __str__(self):
        return "AdBandits ({})".format(self.params)

    def __init__(self, nbArms, horizon, alpha, posterior=Beta):
        self.nbArms = nbArms
        self.alpha = alpha
        self.horizon = horizon
        self.rewards = np.zeros(nbArms)
        self.pulls = np.zeros(nbArms, dtype=int)
        self.posterior = dict()
        for arm in range(self.nbArms):
            self.posterior[arm] = posterior()
        # self.params = 'alpha:' + repr(self.alpha) + ', horizon:' + repr(self.horizon)
        self.params = 'alpha:' + repr(self.alpha)
        # self.startGame()  # XXX do not call it here!
        self.t = -1

    def startGame(self):
        self.t = 0
        self.rewards.fill(0)
        self.pulls.fill(0)
        for arm in range(self.nbArms):
            self.posterior[arm].reset()

    def getReward(self, arm, reward):
        self.posterior[arm].update(reward)
        self.rewards[arm] += reward
        self.pulls[arm] += 1
        self.t += 1

    def computeIndex(self, arm):
        return self.posterior[arm].sample()

    def choice(self):
        # Thompson Exploration
        if rn.random() > 1.0 * self.t / (self.horizon * self.alpha):
            # XXX if possible, this part should also use numpy arrays to be faster?
            upperbounds = [self.computeIndex(i) for i in range(self.nbArms)]
            maxIndex = max(upperbounds)
            bestArms = [arm for (arm, index) in enumerate(upperbounds) if index == maxIndex]
            arm = rn.choice(bestArms)
        # UCB-Bayes
        else:
            expectations = (1.0 + self.rewards) / (2.0 + self.pulls)
            upperbounds = [self.posterior[arm].quantile(1. - 1. / self.t) for arm in range(self.nbArms)]
            regret = np.max(upperbounds) - expectations
            remin = np.min(regret)
            admissible = np.where(regret == remin)[0]
            arm = rn.choice(admissible)
        return arm
