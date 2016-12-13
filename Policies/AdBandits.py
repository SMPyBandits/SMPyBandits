# -*- coding: utf-8 -*-
""" The AdBandits bandit algorithm
Reference: [AdBandit: A New Algorithm For Multi-Armed Bandits, F.S.Truzzi, V.F.da Silva, A.H.R.Costa, F.G.Cozman](http://sites.poli.usp.br/p/fabio.cozman/Publications/Article/truzzi-silva-costa-cozman-eniac2013.pdf)
Code from: https://github.com/flaviotruzzi/AdBandits/
"""
from __future__ import print_function

__author__ = "Flavio Truzzi et al."
__version__ = "0.1"

import random as rn
import numpy as np
from .Beta import Beta


class AdBandits(object):
    """ The AdBandits bandit algorithm
    Reference: [AdBandit: A New Algorithm For Multi-Armed Bandits, F.S.Truzzi, V.F.da Silva, A.H.R.Costa, F.G.Cozman](http://sites.poli.usp.br/p/fabio.cozman/Publications/Article/truzzi-silva-costa-cozman-eniac2013.pdf)
    Code from: https://github.com/flaviotruzzi/AdBandits/
    """

    def __str__(self):
        # return "AdBandits (alpha: {}, horizon: {})".format(self.alpha, self.horizon)
        return "AdBandits (alpha: {})".format(self.alpha)

    def __init__(self, nbArms, horizon, alpha, posterior=Beta):
        self.nbArms = nbArms
        self.alpha = alpha
        self.horizon = horizon
        self.rewards = np.zeros(nbArms)
        self.pulls = np.zeros(nbArms, dtype=int)
        self.posterior = [None] * self.nbArms  # Faster with a list
        for arm in range(self.nbArms):
            self.posterior[arm] = posterior()
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
            admissible = np.where(regret == np.min(regret))[0]
            arm = rn.choice(admissible)
        return arm

    # def choiceWithRank(self, rank=1):
    #     """ FIXME I should do it directly, here."""
    #     return self.choice()
