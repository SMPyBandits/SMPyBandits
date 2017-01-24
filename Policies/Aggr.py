# -*- coding: utf-8 -*-
""" The Aggregated bandit algorithm, similar to Exp4.
Reference: https://github.com/Naereen/AlgoBandits
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np
import numpy.random as rn
from .BasePolicy import BasePolicy


# Default values for the parameters
update_all_children = False

# self.unbiased is a flag to know if the rewards are used as biased estimator,
# ie just r_t, or unbiased estimators, r_t / p_t
unbiased = False
unbiased = True


class Aggr(BasePolicy):
    """ The Aggregated bandit algorithm, similar to Exp4.
    Reference: https://github.com/Naereen/AlgoBandits
    """

    def __init__(self, nbArms, learningRate, children,
                 decreaseRate=None, unbiased=unbiased, horizon=None, lower=0., amplitude=1.,
                 update_all_children=update_all_children, prior='uniform'):
        # Attributes
        self.nbArms = nbArms
        self.lower = lower
        self.amplitude = amplitude
        self.learningRate = learningRate
        self.decreaseRate = decreaseRate
        self.unbiased = unbiased
        self.horizon = horizon
        # If possible, pre compute the learning rate
        if horizon is not None and self.decreaseRate == 'auto':
            self.learningRate = np.sqrt(2 * np.log(self.nbChildren) / (self.horizon * self.nbArms))
            self.decreaseRate = None
        self.update_all_children = update_all_children
        self.nbChildren = len(children)
        self.t = -1
        # Internal object memory
        self.children = []
        for childId, child in enumerate(children):
            if isinstance(child, dict):
                print("  Creating this child player from a dictionnary 'children[{}]' = {} ...".format(childId, child))  # DEBUG
                self.children.append(child['archtype'](nbArms, lower=lower, amplitude=amplitude, **child['params']))
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
        self.choices = (-10000) * np.ones(self.nbChildren, dtype=int)

    # Print, different output according to the learning rate
    def __str__(self):
        if self.decreaseRate == 'auto':
            if self.horizon:
                return "Aggr(nb: {}, dRate: {}, horizon: {})".format(self.nbChildren, self.decreaseRate, self.horizon)
            else:
                return "Aggr(nb: {}, dRate: {})".format(self.nbChildren, self.decreaseRate)
        elif self.decreaseRate is not None:
            return "Aggr(nb: {}, rate: {}, dRate: {})".format(self.nbChildren, self.learningRate, self.decreaseRate)
        else:
            return "Aggr(nb: {}, rate: {})".format(self.nbChildren, self.learningRate)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def rate(self):
        """ Learning rate, can be constant if self.decreaseRate is None, or decreasing.

        - if horizon is known, use the formula which uses it,
        - if horizon is not known, use the formula which uses current time t,
        - else, if decreaseRate is a number, use an exponentionally decreasing learning rate, rate = learningRate * exp(- t / decreaseRate). Bad.
        """
        if self.decreaseRate is None:  # Constant learning rate
            return self.learningRate
        elif self.decreaseRate == 'auto':
            # DONE Implement the two smart values given in Theorem 4.2 from [Bubeck & Cesa-Bianchi, 2012]
            if self.horizon is None:
                return np.sqrt(np.log(self.nbChildren) / (self.t * self.nbArms))
            else:
                return np.sqrt(2 * np.log(self.nbChildren) / (self.horizon * self.nbArms))
        else:
            # DONE I tried to reduce the learning rate (geometrically) when t increase: it does not improve much
            return self.learningRate * np.exp(- self.t / self.decreaseRate)

    def startGame(self):
        self.t = 0
        # Start all children
        for i in range(self.nbChildren):
            self.children[i].startGame()
        self.choices.fill(-1)

    def getReward(self, arm, reward):
        self.t += 1
        # First, give reward to all child children
        for i in range(self.nbChildren):
            self.children[i].getReward(arm, reward)
        # Then compute the new learning rate
        trusts = self.trusts
        rate = self.rate
        reward = (reward - self.lower) / self.amplitude
        # reward = reward - 1  # FIXME try this trick of receiving a loss instead of a reward ?
        # FIXED compute the proba that we observed this arm, p_t
        if self.unbiased:
            proba_of_observing_arm = np.sum(trusts[self.choices == arm])
            # print("  Observing arm", arm, "with reward", reward, "and the estimated proba of observing it was", proba_of_observing_arm)  # DEBUG
            reward /= proba_of_observing_arm
        scalingConstant = np.exp(reward * rate)
        # 3. increase self.trusts for the children who were true
        trusts[self.choices == arm] *= scalingConstant
        # DONE test both, by changing the option self.update_all_children
        if self.update_all_children:
            trusts[self.choices != arm] /= scalingConstant
        # 4. renormalize self.trusts to make it a proba dist
        # In practice, it also decreases the self.trusts for the children who were wrong
        self.trusts = trusts / np.sum(trusts)
        # print("  The most trusted child policy is the {}th with confidence {}.".format(1 + np.argmax(self.trusts), np.max(self.trusts)))  # DEBUG
        # print("self.trusts =", self.trusts)  # DEBUG

    def makeChildrenChose(self):
        """ Convenience method to make every children chose their best arm."""
        for i in range(self.nbChildren):
            self.choices[i] = self.children[i].choice()
            # Could we be faster here? Idea: first sample according to self.trusts, then make it decide
            # XXX No: in fact, we need to vector self.choices to update the self.trusts probabilities!
        # print("self.choices =", self.choices)  # DEBUG

    def choice(self):
        # 1. make vote every child children
        self.makeChildrenChose()
        # 2. select the vote to trust, randomly
        return rn.choice(self.choices, p=self.trusts)

    def choiceWithRank(self, rank=1):
        if rank == 1:
            return self.choice()
        else:
            for i in range(self.nbChildren):
                self.choices[i] = self.children[i].choiceWithRank(rank)
            return rn.choice(self.choices, p=self.trusts)

    def choiceFromSubSet(self, availableArms='all'):
        if (availableArms == 'all') or (len(availableArms) == self.nbArms):
            return self.choice()
        else:
            for i in range(self.nbChildren):
                self.choices[i] = self.children[i].choiceFromSubSet(availableArms)
            return rn.choice(self.choices, p=self.trusts)

    def choiceMultiple(self, nb=1):
        if nb == 1:
            return self.choice()
        else:
            for i in range(self.nbChildren):
                self.choices[i] = self.children[i].choiceMultiple(nb)
            return rn.choice(self.choices, size=nb, replace=False, p=self.trusts)
            # XXX is there something more to do??
