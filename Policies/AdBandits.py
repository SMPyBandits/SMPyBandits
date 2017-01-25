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
        # return r"AdBandits($\alpha={:.3g}$, $T={:.3g}$)".format(self.alpha, self.horizon)
        return r"AdBandits($\alpha={:.3g}$)".format(self.alpha)

    def startGame(self):
        super(AdBandits, self).startGame()
        for arm in range(self.nbArms):
            self.posterior[arm].reset()

    def getReward(self, arm, reward):
        super(AdBandits, self).getReward(arm, reward)
        reward = (reward - self.lower) / self.amplitude
        self.posterior[arm].update(reward)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):
        return float(self.t / (self.horizon * self.alpha))

    def choice(self):
        # Thompson Exploration
        if rn.random() > self.epsilon:
            upperbounds = [self.posterior[i].sample() for i in range(self.nbArms)]
            maxIndex = max(upperbounds)
            bestArms = [arm for (arm, index) in enumerate(upperbounds) if index == maxIndex]
            arm = rn.choice(bestArms)
        # UCB-Bayes
        else:
            expectations = (1.0 + self.rewards) / (2.0 + self.pulls)
            upperbounds = [self.posterior[arm].quantile(1. - 1. / self.t) for arm in range(self.nbArms)]
            regret = np.max(upperbounds) - expectations
            admissible = np.nonzero(regret == np.min(regret))[0]
            arm = rn.choice(admissible)
        return arm

    def choiceWithRank(self, rank=1):
        if rank == 1:
            return self.choice()
        else:
            assert rank >= 1, "Error: for AdBandits = {}, in choiceWithRank(rank={}) rank has to be >= 1.".format(self, rank)
            # Thompson Exploration
            if rn.random() > self.epsilon:
                indexes = [self.posterior[i].sample() for i in range(self.nbArms)]
            # UCB-Bayes
            else:
                expectations = (1.0 + self.rewards) / (2.0 + self.pulls)
                upperbounds = [self.posterior[arm].quantile(1. - 1. / self.t) for arm in range(self.nbArms)]
                indexes = expectations - np.max(upperbounds)
            # We computed the indexes, OK let's use them
            sortedRewards = np.sort(indexes)  # XXX What happens here if two arms has the same index, being the max?
            chosenIndex = sortedRewards[-rank]
            # Uniform choice among the rank-th best arms
            return np.random.choice(np.nonzero(indexes == chosenIndex)[0])
