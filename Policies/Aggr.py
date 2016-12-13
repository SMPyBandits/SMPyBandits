# -*- coding: utf-8 -*-
""" The Aggregated bandit algorithm
Reference: https://github.com/Naereen/AlgoBandits
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np
import numpy.random as rn


# Default values for the parameters
update_all_children = False


class Aggr(object):
    """ The Aggregated bandit algorithm
    Reference: https://github.com/Naereen/AlgoBandits
    """

    def __init__(self, nbArms, learningRate, children,
                 decreaseRate=None,
                 update_all_children=update_all_children, prior='uniform'):
        # Attributes
        self.nbArms = nbArms
        self.learningRate = learningRate
        self.decreaseRate = decreaseRate
        self.update_all_children = update_all_children
        self.nbChildren = len(children)
        self.t = -1
        # Internal object memory
        self.children = []
        for childId, child in enumerate(children):
            if isinstance(child, dict):
                print("  Creating this child player from a dictionnary 'children[{}]' = {} ...".format(childId, child))  # DEBUG
                self.children.append(child['archtype'](nbArms, **child['params']))
            else:
                print("  Using this already created player 'children[{}]' = {} ...".format(childId, child))  # DEBUG
                self.children.append(child)
        # Initialize the arrays
        if prior is not None and prior != 'uniform':
            assert len(prior) == self.nbChildren, "Error: the 'prior' argument given to Aggr.Aggr has to be an array of the good size ({}).".format(self.nbChildren)
            self.trusts = prior
        else:   # Assume uniform prior if not given or if = 'uniform'
            self.trusts = np.ones(self.nbChildren) / self.nbChildren
        # Internal vectorial memory
        self.choices = (-1) * np.ones(self.nbChildren, dtype=int)

    def __str__(self):
        return "Aggr (nb: {}, rate: {})".format(self.nbChildren, self.learningRate)

    # @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof)
    def startGame(self):
        self.t = 0
        # Start all child children
        for i in range(self.nbChildren):
            self.children[i].startGame()
        self.choices.fill(-1)

    # @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof)
    def getReward(self, arm, reward):
        self.t += 1
        if self.decreaseRate is None:
            learningRate = self.learningRate
        else:
            # DONE I tried to reduce the learning rate (geometrically) when t increase: it does not improve much
            learningRate = self.learningRate * np.exp(- self.t / self.decreaseRate)
        # Give reward to all child children
        for i in range(self.nbChildren):
            self.children[i].getReward(arm, reward)
        scalingConstant = np.exp(reward * learningRate)
        # 3. increase self.trusts for the children who were true
        self.trusts[self.choices == arm] *= scalingConstant
        # DONE test both, by changing the option self.update_all_children
        if self.update_all_children:
            self.trusts[self.choices != arm] /= scalingConstant
        # 4. renormalize self.trusts to make it a proba dist
        # In practice, it also decreases the self.trusts for the children who were wrong
        self.trusts = self.trusts / np.sum(self.trusts)
        # print("  The most trusted child policy is the {}th with confidence {}.".format(1 + np.argmax(self.trusts), np.max(self.trusts)))  # DEBUG
        # print("self.trusts =", self.trusts)  # DEBUG

    # @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof)
    def choice(self):
        # 1. make vote every child children
        for i in range(self.nbChildren):
            self.choices[i] = self.children[i].choice()
            # Could we be faster here? Idea: first sample according to self.trusts, then make it decide
            # XXX No: in fact, we need to vector self.choices to update the self.trusts probabilities!
        # print("self.choices =", self.choices)  # DEBUG
        # 2. select the vote to trust, randomly
        return rn.choice(self.choices, p=self.trusts)

    # @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof)
    def choiceWithRank(self, rank=1):
        # 1. make vote every child children
        for i in range(self.nbChildren):
            self.choices[i] = self.children[i].choiceWithRank(rank)
            # Could we be faster here? Idea: first sample according to self.trusts, then make it decide
            # XXX No: in fact, we need to vector self.choices to update the self.trusts probabilities!
        # print("self.choices =", self.choices)  # DEBUG
        # 2. select the vote to trust, randomly
        return rn.choice(self.choices, p=self.trusts)
