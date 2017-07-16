# -*- coding: utf-8 -*-
""" My Aggregated bandit algorithm, similar to Exp4 but not exactly equivalent.

The algorithm is a master A, managing several "slave" algorithms, A1, .., AN.

- At every step, the prediction of every slave is gathered, and a vote is done to decide A's decision.
- The vote is simply a majority vote, weighted by a trust probability. If Ai decides arm Ii, then the probability of selecting k is the sum of trust probabilities, Pi, of every Ai for which Ii = k.
- The trust probabilities are first uniform, Pi = 1/N, and then at every step, after receiving the feedback for *one* arm k (the reward), the trust in each slave Ai is updated: Pi increases if Ai advised k (Ii = k), or decreases if Ai advised another arm.

- The detail about how to increase or decrease the probabilities are specified below.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.5"

import numpy as np
import numpy.random as rn
from .BasePolicy import BasePolicy


# Default values for the parameters

#: self.unbiased is a flag to know if the rewards are used as biased estimator,
#: ie just r_t, or unbiased estimators, r_t / p_t, if p_t is the probability of selecting that arm at time t.
#: It seemed to work better with unbiased estimators (of course).
unbiased = False
unbiased = True    # Better

#: Flag to know if we should update the trusts proba like in Exp4 or like in my initial Aggr proposal
#:
#: - First choice: like Exp4, trusts are fully recomputed, trusts^(t+1) = exp(rate_t * estimated mean rewards upto time t),
#: - Second choice: my proposal, trusts are just updated multiplicatively, trusts^(t+1) <-- trusts^t * exp(rate_t * estimate instant reward at time t).
#:
#: Both choices seem fine, and anyway the trusts are renormalized to be a probability distribution, so it doesn't matter much.
update_like_exp4 = True
update_like_exp4 = False  # Better

#: Non parametric flag to know if the Exp4-like update uses losses or rewards.
#: Losses are 1 - reward, in which case the rate_t is negative.
USE_LOSSES = True
USE_LOSSES = False

#: Should all trusts be updated, or only the trusts of slaves Ai who advised the decision Aggr[A1..AN] followed.
update_all_children = False


class Aggr(BasePolicy):
    """ My Aggregated bandit algorithm, similar to Exp4 but not exactly equivalent."""

    def __init__(self, nbArms, children=None,
                 learningRate=None, decreaseRate=None, horizon=None,
                 update_all_children=update_all_children, update_like_exp4=update_like_exp4,
                 unbiased=unbiased, prior='uniform',
                 lower=0., amplitude=1.,
                 ):
        # Attributes
        self.nbArms = nbArms  #: Number of arms
        self.lower = lower  #: Lower values for rewards
        self.amplitude = amplitude  #: Larger values for rewards
        self.learningRate = learningRate  #: Value of the learning rate (can be decreasing in time)
        self.decreaseRate = decreaseRate  #: Value of the constant used in the decreasing of the learning rate
        self.unbiased = unbiased or update_like_exp4  #: Flag, see above.
        # XXX If we use the Exp4 update rule, it's better to be unbiased
        # XXX If we use my update rule, it seems to be better to be "biased"
        self.horizon = horizon  #: Horizon T, if given and not None, can be used to compute a "good" constant learning rate, sqrt(2 log(N) / (T K)) for N slaves, K arms (heuristic).
        self.update_all_children = update_all_children  #: Flag, see above.
        self.nbChildren = len(children)  #: Number N of slave algorithms.
        self.t = -1  #: Internal time
        self.update_like_exp4 = update_like_exp4  #: Flag, see above.
        # If possible, pre compute the learning rate
        if horizon is not None and decreaseRate == 'auto':
            self.learningRate = np.sqrt(2 * np.log(self.nbChildren) / (self.horizon * self.nbArms))
            self.decreaseRate = None
        elif learningRate is None:
            self.decreaseRate = 'auto'
        # Internal object memory
        self.children = []  #: List of slave algorithms.
        for i, child in enumerate(children):
            if isinstance(child, dict):
                print("  Creating this child player from a dictionnary 'children[{}]' = {} ...".format(i, child))  # DEBUG
                localparams = {'lower': lower, 'amplitude': amplitude}
                localparams.update(child['params'])
                self.children.append(child['archtype'](nbArms, **localparams))
            elif isinstance(child, type):
                print("  Using this not-yet created player 'children[{}]' = {} ...".format(i, child))  # DEBUG
                self.children.append(child(nbArms, lower=lower, amplitude=amplitude))  # Create it here!
            else:
                print("  Using this already created player 'children[{}]' = {} ...".format(i, child))  # DEBUG
                self.children.append(child)
        # Initialize the arrays
        if prior is not None and prior != 'uniform':
            assert len(prior) == self.nbChildren, "Error: the 'prior' argument given to Aggr has to be an array of the good size ({}).".format(self.nbChildren)  # DEBUG
            self.trusts = prior  #: Initial trusts in the slaves. Default to uniform, but a prior can also be given.
        else:   # Assume uniform prior if not given or if = 'uniform'
            self.trusts = np.ones(self.nbChildren) / self.nbChildren
        # Internal vectorial memory
        self.choices = (-10000) * np.ones(self.nbChildren, dtype=int)  #: Keep track of the last choices of each slave, to know whom to update if update_all_children is false.
        if self.update_like_exp4:
            self.children_cumulated_losses = np.zeros(self.nbChildren)  #: Keep track of the cumulated loss (empirical mean)

    # Print, different output according to the parameters
    def __str__(self):
        """ Nicely print the name of the algorithm with its relevant parameters."""
        exp4 = ", Exp4" if self.update_like_exp4 else ""
        all_children = ", updateAll" if self.update_all_children else ""
        if self.decreaseRate == 'auto':
            if self.horizon:
                return r"Aggr($N={}${}{}, $T={}$)".format(self.nbChildren, exp4, all_children, self.horizon)
            else:
                return r"Aggr($N={}${}{})".format(self.nbChildren, exp4, all_children)
        elif self.decreaseRate is not None:
            return r"Aggr($N={}${}{}, $\eta={:.3g}$, $dRate={:.3g}$)".format(self.nbChildren, exp4, all_children, self.learningRate, self.decreaseRate)
        else:
            return r"Aggr($N={}${}{}, $\eta={:.3g}$)".format(self.nbChildren, exp4, all_children, self.learningRate)

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
            # DONE Implement the two smart values given in Theorem 4.2 from [Bubeck & Cesa-Bianchi, 2012](http://sbubeck.com/SurveyBCB12.pdf)
            if self.horizon is None:
                return np.sqrt(np.log(self.nbChildren) / (self.t * self.nbArms))
            else:
                return np.sqrt(2 * np.log(self.nbChildren) / (self.horizon * self.nbArms))
        else:
            # DONE I tried to reduce the learning rate (geometrically) when t increase: it does not improve much
            return self.learningRate * np.exp(- self.t / self.decreaseRate)

    # --- Start and get a reward

    def startGame(self):
        """ Start the game for each child."""
        self.t = 0
        # Start all children
        for i in range(self.nbChildren):
            self.children[i].startGame()
        self.choices.fill(-1)

    def getReward(self, arm, reward):
        """ Give reward for each child, and then update the trust probabilities."""
        self.t += 1
        # First, give reward to all children
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

    # --- Internal method

    def _makeChildrenChoose(self):
        """ Convenience method to make every children chose their best arm, and store their decision in ``self.choices``."""
        for i, child in enumerate(self.children):
            self.choices[i] = child.choice()
            # Could we be faster here? Idea: first sample according to self.trusts, then make it decide
            # XXX No: in fact, we need to vector self.choices to update the self.trusts probabilities!
        # print("self.choices =", self.choices)  # DEBUG

    # --- Choice of arm methods

    def choice(self):
        """ Make each child vote, then sample the decision by importance sampling on their votes with the trust probabilities."""
        # 1. make vote every child
        self._makeChildrenChoose()
        # 2. select the vote to trust, randomly
        return rn.choice(self.choices, p=self.trusts)

    def choiceWithRank(self, rank=1):
        """ Make each child vote, with rank, then sample the decision by importance sampling on their votes with the trust probabilities."""
        if rank == 1:
            return self.choice()
        else:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceWithRank(rank)
            return rn.choice(self.choices, p=self.trusts)

    def choiceFromSubSet(self, availableArms='all'):
        """ Make each child vote, on subsets of arms, then sample the decision by importance sampling on their votes with the trust probabilities."""
        if (availableArms == 'all') or (len(availableArms) == self.nbArms):
            return self.choice()
        else:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceFromSubSet(availableArms)
            return rn.choice(self.choices, p=self.trusts)

    def choiceMultiple(self, nb=1):
        """ Make each child vote, multiple times, then sample the decision by importance sampling on their votes with the trust probabilities."""
        if nb == 1:
            return self.choice()
        else:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceMultiple(nb)
            return rn.choice(self.choices, size=nb, replace=False, p=self.trusts)

    def choiceIMP(self, nb=1):
        """ Make each child vote, multiple times (with IMP scheme), then sample the decision by importance sampling on their votes with the trust probabilities."""
        if nb == 1:
            return self.choice()
        else:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceIMP(nb)
            return rn.choice(self.choices, size=nb, replace=False, p=self.trusts)

    def estimatedOrder(self):
        """ Make each child vote for their estimate order of the arms, then randomly select an ordering by importance sampling with the trust probabilities.
        Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means."""
        alltrusts = self.trusts
        orders = []
        trusts = []
        for i, child in enumerate(self.children):
            if hasattr(child, 'estimatedOrder'):
                orders.append(child.estimatedOrder())
                trusts.append(alltrusts[i])
        trusts = np.asarray(trusts)
        trusts /= np.sum(trusts)
        chosenOrder = int(rn.choice(len(orders), size=1, replace=False, p=trusts))
        return orders[chosenOrder]
