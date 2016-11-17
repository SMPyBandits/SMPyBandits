# -*- coding: utf-8 -*-
""" The Aggregated bandit algorithm

- Reference: FIXME write it!
"""

from __future__ import print_function
import numpy as np
import numpy.random as rn
from .Beta import Beta

__author__ = "Lilian Besson, Emilie Kaufmann"
__version__ = "0.1"


class Aggr:
    def __str__(self):
        return "Aggr"

    def __init__(self, nbArms, learningRate, policies,
                 prior=None, posterior=Beta):
        self.nbArms = nbArms
        self.learningRate = learningRate
        self.policies = []
        self.posterior = dict()
        for arm in range(self.nbArms):
            self.posterior[arm] = posterior()
        # Create all child policies
        self.nbPolicies = len(policies)
        for i in range(self.nbPolicies):
            self.policies.append(policies[i]['archtype'](nbArms, **policies[i]['params']))
        if prior is not None and prior != 'uniform':
            self.trusts = prior  # Has to be an array of the good size
        else:   # Assume uniform prior if not given or if = 'uniform'
            self.trusts = np.ones(self.nbPolicies) / self.nbPolicies
        self.rewards = np.zeros(self.nbArms)
        self.pulls = np.zeros(self.nbArms)
        self.params = 'policies:' + repr(self.policies)
        self.startGame()

    def startGame(self):
        self.rewards[:] = 0
        self.pulls[:] = 0
        self.t = 1
        for arm in range(self.nbArms):
            self.posterior[arm].reset()
        # Start all child policies
        for i in range(self.nbPolicies):
            self.policies[i].startGame()

    def getReward(self, arm, reward):
        self.posterior[arm].update(reward)
        self.rewards[arm] += reward
        self.pulls[arm] += 1
        self.t += 1
        # Give reward to all child policies
        for i in range(self.nbPolicies):
            self.policies[i].getReward(arm, reward)
            if self.choices[i] == arm:  # this child's choice was chosen
                # 3. increase self.trusts for the childs who were true
                self.trusts[i] *= np.exp(reward * self.learningRate)
        # 4. renormalize self.trusts to make it a proba dist
        self.trusts = self.trusts / np.sum(self.trusts)
        # print("self.trusts =", self.trusts)  # DEBUG

    def choice(self):
        # TODO :
        # 1. make vote every child policies
        self.choices = [-1] * self.nbPolicies
        for i in range(self.nbPolicies):
            self.choices[i] = self.policies[i].choice()
        # print("self.choices =", self.choices)  # DEBUG
        # 2. select the vote to trust
        # trustOnArms = [0] * self.nbArms
        # for i in range(self.nbPolicies):
        #     trustOnArms[self.choices[i]] += self.trusts[i]
        # print("trustOnArms =", trustOnArms)  # DEBUG
        return rn.choice(self.choices, p=self.trusts)
