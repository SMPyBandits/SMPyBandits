# -*- coding: utf-8 -*-
""" The Aggregated bandit algorithm
Reference: FIXME write it!
"""
from __future__ import print_function

__author__ = "Lilian Besson, Emilie Kaufmann"
__version__ = "0.1"

import numpy as np
import numpy.random as rn

from .Beta import Beta


class Aggr:
    """ The Aggregated bandit algorithm
    Reference: FIXME write it!
    """

    def __init__(self, nbArms, learningRate, children,
                 prior='uniform', posterior=Beta):
        self.nbArms = nbArms
        self.learningRate = learningRate
        self.children = []
        # Create all child children
        self.nbChildren = len(children)
        for i in range(self.nbChildren):
            self.children.append(children[i]['archtype'](nbArms, **children[i]['params']))
        if prior is not None and prior != 'uniform':
            self.trusts = prior  # Has to be an array of the good size
        else:   # Assume uniform prior if not given or if = 'uniform'
            self.trusts = np.ones(self.nbChildren) / float(self.nbChildren)
        self.rewards = np.zeros(self.nbArms)
        self.pulls = np.zeros(self.nbArms)
        self.params = "children:" + repr(self.children)
        self.startGame()

    def __str__(self):
        return "Aggr"

    def startGame(self):
        self.rewards[:] = 0
        self.pulls[:] = 0
        self.t = 1
        # Start all child children
        for i in range(self.nbChildren):
            self.children[i].startGame()

    def getReward(self, arm, reward):
        self.rewards[arm] += reward
        self.pulls[arm] += 1
        self.t += 1
        # Give reward to all child children
        for i in range(self.nbChildren):
            self.children[i].getReward(arm, reward)
            if self.choices[i] == arm:  # this child's choice was chosen
                # 3. increase self.trusts for the children who were true
                self.trusts[i] *= np.exp(reward * self.learningRate)
            else:
                # 3. XXX decrease self.trusts for the children who were wrong
                self.trusts[i] *= np.exp(- reward * self.learningRate)
            # FIXME test both
        # 4. renormalize self.trusts to make it a proba dist
        # In practice, it also decreases the self.trusts for the children who were wrong
        # print("  The most trusted child policy is the {}th with confidence {}.".format(1 + np.argmax(self.trusts), np.max(self.trusts)))  # DEBUG
        self.trusts = self.trusts / float(np.sum(self.trusts))
        # print("self.trusts =", self.trusts)  # DEBUG

    def choice(self):
        # 1. make vote every child children
        self.choices = [-1] * self.nbChildren
        for i in range(self.nbChildren):
            self.choices[i] = self.children[i].choice()
        # print("self.choices =", self.choices)  # DEBUG
        # 2. select the vote to trust, randomly
        return rn.choice(self.choices, p=self.trusts)
