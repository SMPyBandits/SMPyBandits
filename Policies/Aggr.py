# -*- coding: utf-8 -*-
""" My Aggregated bandit algorithm, similar to Exp4 but not exactly equivalent.
Reference: https://github.com/Naereen/AlgoBandits
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.5"

import numpy as np
import numpy.random as rn
from .BasePolicy import BasePolicy


# Default values for the parameters
update_all_children = False

# self.unbiased is a flag to know if the rewards are used as biased estimator,
# ie just r_t, or unbiased estimators, r_t / p_t
unbiased = False
unbiased = True    # XXX Better

# Flag to know if we should update the trusts proba like in Exp4 or like in my initial Aggr proposal
update_like_exp4 = False    # trusts^(t+1) <-- trusts^t * exp(rate_t * estimate reward at time t)
update_like_exp4 = True     # trusts^(t+1) = exp(rate_t * estimated rewards upto time t)  # XXX Better

# Non parametric flag to know if the Exp4-like update uses losses or rewards
USE_LOSSES = False
USE_LOSSES = True


class Aggr(BasePolicy):
    """ My Aggregated bandit algorithm, similar to Exp4 but not exactly equivalent.
    Reference: https://github.com/Naereen/AlgoBandits
    """

    def __init__(self, nbArms, children,
                 learningRate=None, decreaseRate=None, horizon=None,
                 update_all_children=update_all_children, update_like_exp4=update_like_exp4,
                 unbiased=unbiased, prior='uniform',
                 lower=0., amplitude=1.,
                 ):
        # Attributes
        self.nbArms = nbArms
        self.lower = lower
        self.amplitude = amplitude
        self.learningRate = learningRate
        self.decreaseRate = decreaseRate
        self.unbiased = unbiased
        self.horizon = horizon
        self.update_all_children = update_all_children
        self.nbChildren = len(children)
        self.t = -1
        self.update_like_exp4 = update_like_exp4
        # If possible, pre compute the learning rate
        if horizon is not None and decreaseRate == 'auto':
            self.learningRate = np.sqrt(2 * np.log(self.nbChildren) / (self.horizon * self.nbArms))
            self.decreaseRate = None
        elif learningRate is None:
            self.decreaseRate = 'auto'
        # Internal object memory
        self.children = []
        for i, child in enumerate(children):
            if isinstance(child, dict):
                print("  Creating this child player from a dictionnary 'children[{}]' = {} ...".format(i, child))  # DEBUG
                self.children.append(child['archtype'](nbArms, lower=lower, amplitude=amplitude, **child['params']))
            else:
                print("  Using this already created player 'children[{}]' = {} ...".format(i, child))  # DEBUG
                self.children.append(child)
        # Initialize the arrays
        if prior is not None and prior != 'uniform':
            assert len(prior) == self.nbChildren, "Error: the 'prior' argument given to Aggr.Aggr has to be an array of the good size ({}).".format(self.nbChildren)
            self.trusts = prior
        else:   # Assume uniform prior if not given or if = 'uniform'
            self.trusts = np.ones(self.nbChildren) / self.nbChildren
        # Internal vectorial memory
        self.choices = (-10000) * np.ones(self.nbChildren, dtype=int)
        if self.update_like_exp4:
            self.children_cumulated_losses = np.zeros(self.nbChildren)

    # Print, different output according to the parameters
    def __str__(self):
        exp4 = ", Exp4" if self.update_like_exp4 else ""
        if self.decreaseRate == 'auto':
            if self.horizon:
                return r"Aggr($N={}${}, $T={}$)".format(self.nbChildren, exp4, self.horizon)
            else:
                return r"Aggr($N={}${})".format(self.nbChildren, exp4)
        elif self.decreaseRate is not None:
            return r"Aggr($N={}${}, $\eta={:.3g}$, $dRate={:.3g}$)".format(self.nbChildren, exp4, self.learningRate, self.decreaseRate)
        else:
            return r"Aggr($N={}${}, $\eta={:.3g}$)".format(self.nbChildren, exp4, self.learningRate)

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
        for child in self.children:
            child.getReward(arm, reward)
        # Then compute the new learning rate
        trusts = self.trusts
        rate = self.rate
        reward = (reward - self.lower) / self.amplitude  # Normalize it to [0, 1]
        # DONE compute the proba that we observed this arm, p_t, if unbiased
        if self.unbiased:
            proba_of_observing_arm = np.sum(trusts[self.choices == arm])
            # print("  Observing arm", arm, "with reward", reward, "and the estimated proba of observing it was", proba_of_observing_arm)  # DEBUG
            reward /= proba_of_observing_arm
        # 3. Compute the new trust proba, like in Exp4
        if self.update_like_exp4:
            if USE_LOSSES:
                # FIXME try this trick of receiving a loss instead of a reward ?
                loss = 1 - reward
                # Update estimated cumulated rewards for each player
                self.children_cumulated_losses[self.choices == arm] += loss
                trusts = np.exp(- rate * self.children_cumulated_losses)
            else:
                # Update estimated cumulated rewards for each player
                self.children_cumulated_losses[self.choices == arm] += reward
                trusts = np.exp(rate * self.children_cumulated_losses)
        # 3'. increase self.trusts for the children who were true
        else:
            scaling = np.exp(rate * reward)
            trusts[self.choices == arm] *= scaling
            # DONE test both, by changing the option self.update_all_children
            if self.update_all_children:
                trusts[self.choices != arm] /= scaling
        # 4. renormalize self.trusts to make it a proba dist
        # In practice, it also decreases the self.trusts for the children who were wrong
        self.trusts = trusts / np.sum(trusts)
        # FIXME experiment dynamic resetting of proba, put this as a parameter
        # if self.t % 2000 == 0:
        #     print("   => t % 2000 == 0 : reinitializing the trust proba ...")  # DEBUG
        #     self.trusts = np.ones(self.nbChildren) / self.nbChildren
        # print("  The most trusted child policy is the {}th with confidence {}.".format(1 + np.argmax(self.trusts), np.max(self.trusts)))  # DEBUG
        # print("self.trusts =", self.trusts)  # DEBUG

    def makeChildrenChose(self):
        """ Convenience method to make every children chose their best arm."""
        for i, child in enumerate(self.children):
            self.choices[i] = child.choice()
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
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceWithRank(rank)
            return rn.choice(self.choices, p=self.trusts)

    def choiceFromSubSet(self, availableArms='all'):
        if (availableArms == 'all') or (len(availableArms) == self.nbArms):
            return self.choice()
        else:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceFromSubSet(availableArms)
            return rn.choice(self.choices, p=self.trusts)

    def choiceMultiple(self, nb=1):
        if nb == 1:
            return self.choice()
        else:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceMultiple(nb)
            return rn.choice(self.choices, size=nb, replace=False, p=self.trusts)
