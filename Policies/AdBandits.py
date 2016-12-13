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
from .BasePolicy import BasePolicy


class AdBandits(BasePolicy):
    """ The AdBandits bandit algorithm
    Reference: [AdBandit: A New Algorithm For Multi-Armed Bandits, F.S.Truzzi, V.F.da Silva, A.H.R.Costa, F.G.Cozman](http://sites.poli.usp.br/p/fabio.cozman/Publications/Article/truzzi-silva-costa-cozman-eniac2013.pdf)
    Code from: https://github.com/flaviotruzzi/AdBandits/
    """

    def __init__(self, nbArms, horizon, alpha, posterior=Beta, lower=0., amplitude=1.):
        super(AdBandits, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.alpha = alpha
        self.horizon = horizon
        self.posterior = [None] * self.nbArms  # List instead of dict, quicker access
        for arm in range(self.nbArms):
            self.posterior[arm] = posterior()

    def __str__(self):
        # return "AdBandits(alpha: {}, horizon: {})".format(self.alpha, self.horizon)
        return "AdBandits(alpha: {})".format(self.alpha)

    def startGame(self):
        super(AdBandits, self).startGame()
        for arm in range(self.nbArms):
            self.posterior[arm].reset()

    def getReward(self, arm, reward):
        super(AdBandits, self).getReward(arm, reward)
        reward = (reward - self.lower) / self.amplitude
        self.posterior[arm].update(reward)

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
