# -*- coding: utf-8 -*-
r""" The CORRAL aggregation bandit algorithm, similar to Exp4 but not exactly equivalent.

The algorithm is a master A, managing several "slave" algorithms, :math:`A_1, ..., A_N`.

- At every step, one slave algorithm is selected, by a random selection from a trust distribution on :math:`[1,...,N]`.
- Then its decision is listen to, played by the master algorithm, and a feedback reward is received.
- The reward is reweighted by the trust of the listened algorithm, and given back to it.
- The other slaves receive, whose decision was not even asked, a zero reward, or no reward at all.
- The trust probabilities are first uniform, :math:`P_i = 1/N`, and then at every step, after receiving the feedback for *one* arm k (the reward), the trust in each slave Ai is updated: :math:`P_i` by the reward received.
- The detail about how to increase or decrease the probabilities are specified in the reference article.

.. note:: Reference: [["Corralling a Band of Bandit Algorithms", by A. Agarwal, H. Luo, B. Neyshabur, R.E. Schapire, 01.2017]](https://arxiv.org/abs/1612.06246v2).
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
import numpy.random as rn
from scipy.optimize import minimize_scalar
from .BasePolicy import BasePolicy


# --- Log-Barrier-OMD

def log_Barrier_OMB(trusts, losses, steps):
    r""" A step of the mirror barrier descent, updating the trusts in log-space:

    - Find :math:`\lambda \in [\min_i l_{t,i}, \max_i l_{t,i}]` such that :math:`\sum_i \frac{1}{1/p_{t,i} + \eta_{t,i}(l_{t,i} - \lambda)} = 1`.
    - Return :math:`\mathbf{p}_{t+1,i}` such that :math:`\frac{1}{p_{t+1,i}} = \frac{1}{p_{t,i}} + \eta_{t,i}(l_{t,i} - \lambda)`.

    - Note: uses `scipy.optimize.minimize_scalar` for the optimization.
    """
    min_loss, max_loss = np.min(losses), np.max(losses)
    def objective(best_loss):
        lhs = np.sum(1. / ((1. / trusts) + steps * (losses - best_loss)))
        rhs = 1.
        return (lhs - rhs) ** 2
    result = minimize_scalar(objective, bounds=(min_loss, max_loss))
    best_loss = result[0]
    return 1. / ((1. / trusts) + steps * (losses - best_loss))


# --- Parameters for the CORRAL algorithm

# Default values for the parameters

#: self.unbiased is a flag to know if the rewards are used as biased estimator,
#: ie just r_t, or unbiased estimators, r_t / p_t, if p_t is the probability of selecting that arm at time t.
#: It seemed to work better with unbiased estimators (of course).
unbiased = False
unbiased = True  # Better

#: Non parametric flag to know if the Exp4-like update uses losses or rewards.
#: Losses are 1 - reward, in which case the rate_t is negative.
use_losses = False
use_losses = True  # Better

#: Default for the initial value of the constant eta.
RATE = 1.


# --- CORRAL algorithm


class CORRAL(BasePolicy):
    """ The CORRAL aggregation bandit algorithm, similar to Exp4 but not exactly equivalent."""

    def __init__(self, nbArms, children,
                 learningRate=None, horizon=None, rate=RATE,
                 unbiased=unbiased, prior='uniform',
                 lower=0., amplitude=1.,
                 ):
        # Attributes
        self.nbArms = nbArms  #: Number of arms
        self.lower = lower  #: Lower values for rewards
        self.amplitude = amplitude  #: Larger values for rewards
        self.unbiased = unbiased  #: Flag, see above.
        # XXX If we use the Exp4 update rule, it's better to be unbiased
        # XXX If we use my update rule, it seems to be better to be "biased"

        self.horizon = horizon
        self.gamma = 1 / horizon  #: Constant :math:`\gamma = 1 / T`.
        self.beta = np.exp(1 / np.log(horizon))  #: Constant :math:`\beta = \exp(1 / \log(T))`.

        self.nbChildren = len(children)  #: Number N of slave algorithms.
        self.rates = np.full(self.nbChildren, learningRate)  #: Value of the learning rate (will be decreasing in time)

        # Internal object memory
        self.t = -1  #: Internal time
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
            assert len(prior) == self.nbChildren, "Error: the 'prior' argument given to CORRAL has to be an array of the good size ({}).".format(self.nbChildren)  # DEBUG
            self.trusts = prior  #: Initial trusts in the slaves. Default to uniform, but a prior can also be given.
        else:   # Assume uniform prior if not given or if = 'uniform'
            self.trusts = np.ones(self.nbChildren) / self.nbChildren
        self.bar_trusts = np.copy(self.trusts)
        # Internal memory not in Aggr
        self.last_choice = None
        self.losses = np.zeros(self.nbChildren)
        self.rhos = np.full(self.nbChildren, 2 * self.nbChildren)

    def __str__(self):
        """ Nicely print the name of the algorithm with its relevant parameters."""
        return r"CORRAL($N={}$, $\gamma={:.3g}$, $\beta={:.3g}$, $\rho={}$, $\eta={}$)".format(self.nbChildren, self.gamma, self.beta, list(self.rhos), list(self.rates))

    # --- Start the game

    def startGame(self):
        """ Start the game for each child."""
        self.t = 0
        # Start all children
        for i in range(self.nbChildren):
            self.children[i].startGame()

    # --- Get a reward

    def getReward(self, arm, reward):
        """ Give reward for each child, and then update the trust probabilities."""
        self.t += 1

        # 1. First, give rewards to all children
        for child in self.children:
            if child == self.last_choice:
                # Give reward, biased or not
                if self.unbiased:
                    child.getReward(arm, reward / self.bar_trusts[self.last_choice])
                else:
                    child.getReward(arm, reward)
            else:  # give 0 reward to all other children
                child.getReward(arm, 0)

        # 2. Then reinitialize this array of losses
        if self.unbiased:
            reward /= self.bar_trusts[self.last_choice]
        reward = (reward - self.lower) / self.amplitude  # Normalize it to [0, 1]

        loss = 1 - reward
        self.losses  = loss / self.bar_trusts

        # 3. Compute the new trust proba, with a log-barrier step
        trusts = log_Barrier_OMB(self.trusts, self.losses, self.rates)
        # 4. renormalize self.trusts to make it a proba dist
        # In practice, it also decreases the self.trusts for the children who were wrong
        self.trusts = trusts / np.sum(trusts)

        # add uniform mixing of proportion gamma
        bar_trusts = (1 - self.gamma) * self.trusts + self.gamma / self.nbChildren
        self.bar_trusts = bar_trusts / np.sum(bar_trusts)

        # 5. Compare trusts with the self.rhos values to compute the new learning rates and rhos
        for i in range(self.nbChildren):
            if (1 / self.bar_trusts[i]) > self.rhos[i]:
                self.rhos[i] = 2 / self.bar_trusts[i]
                # increase the rate for this guy
                self.rates[i] *= self.beta
            # else:  # nothing to do
            #     self.rhos[i] = self.rhos[i]
            #     self.rates[i] = self.rates[i]

        print("  The most trusted child policy is the {}th with confidence {}...".format(1 + np.argmax(self.bar_trusts), np.max(self.bar_trusts)))  # DEBUG
        assert np.isclose(np.sum(self.bar_trusts), 1), "Error: bar_trusts don't sum to 1."  # DEBUG
        print("self.bar_trusts =", self.bar_trusts)  # DEBUG
        assert np.isclose(np.sum(self.trusts), 1), "Error: trusts don't sum to 1."  # DEBUG
        print("self.trusts =", self.trusts)  # DEBUG

    # --- Choice of arm methods

    def choice(self):
        """ Trust one of the slave and listen to his choice."""
        # 1. first decide who to listen to
        trusted_one = rn.choice(self.nbChildren, p=self.bar_trusts)
        self.last_choice = trusted_one
        # 2. then listen to him
        his_choice = self.children[trusted_one].choice()
        return his_choice

    def choiceWithRank(self, rank=1):
        """ Trust one of the slave and listen to his choiceWithRank."""
        # 1. first decide who to listen to
        trusted_one = rn.choice(self.nbChildren, p=self.bar_trusts)
        self.last_choice = trusted_one
        # 2. then listen to him
        his_choice = self.children[trusted_one].choiceWithRank(rank=rank)
        return his_choice

    def choiceFromSubSet(self, availableArms='all'):
        """ Trust one of the slave and listen to his choiceFromSubSet."""
        # 1. first decide who to listen to
        trusted_one = rn.choice(self.nbChildren, p=self.bar_trusts)
        self.last_choice = trusted_one
        # 2. then listen to him
        his_choice = self.children[trusted_one].choiceFromSubSet(availableArms=availableArms)
        return his_choice

    def choiceMultiple(self, nb=1):
        """ Trust one of the slave and listen to his choiceMultiple."""
        # 1. first decide who to listen to
        trusted_one = rn.choice(self.nbChildren, p=self.bar_trusts)
        self.last_choice = trusted_one
        # 2. then listen to him
        his_choices = self.children[trusted_one].choiceMultiple(nb=nb)
        return his_choices

    def choiceIMP(self, nb=1):
        """ Trust one of the slave and listen to his choiceIMP."""
        # 1. first decide who to listen to
        trusted_one = rn.choice(self.nbChildren, p=self.bar_trusts)
        self.last_choice = trusted_one
        # 2. then listen to him
        his_choices = self.children[trusted_one].choiceIMP(nb=nb)
        return his_choices

    def estimatedOrder(self):
        r""" Trust one of the slave and listen to his estimated order.

        - Return the estimate order of the arms, as a permutation on :math:`[0,...,K-1]` that would order the arms by increasing means.
        """
        # 1. first decide who to listen to
        trusted_one = rn.choice(self.nbChildren, p=self.bar_trusts)
        self.last_choice = trusted_one
        # 2. then listen to him
        his_order = self.children[trusted_one].estimatedOrder()
        return his_order
