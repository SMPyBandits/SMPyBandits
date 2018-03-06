# -*- coding: utf-8 -*-
r""" The LearnExp aggregation bandit algorithm, similar to Exp4 but not equivalent.

The algorithm is a master A, managing several "slave" algorithms, :math:`A_1, ..., A_N`.

- At every step, one slave algorithm is selected, by a random selection from a trust distribution on :math:`[1,...,N]`.
- Then its decision is listen to, played by the master algorithm, and a feedback reward is received.
- The reward is reweighted by the trust of the listened algorithm, and given back to it *with* a certain probability.
- The other slaves, whose decision was not even asked, receive nothing.
- The trust probabilities are first uniform, :math:`P_i = 1/N`, and then at every step, after receiving the feedback for *one* arm k (the reward), the trust in each slave Ai is updated: :math:`P_i` by the reward received.
- The detail about how to increase or decrease the probabilities are specified in the reference article.

.. note:: Reference: [[Learning to Use Learners' Advice, A.Singla, H.Hassani & A.Krause, 2017](https://arxiv.org/abs/1702.04825)].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.7"

import numpy as np
import numpy.random as rn
from .BasePolicy import BasePolicy


# --- Utility

def with_probability(p):
    """Use it like this:

    .. code:: python

        if with_probability(0.2):
            print("This happens only 20% of the times!")
    """
    assert 0 <= p <= 1, "Error: for 'with_probability(p)', p = {:.3g} has to be between 0 and 1 to be a valid probability."  # DEBUG
    return bool(rn.random() <= p)


# --- Renormalize function

def renormalize_reward(reward, lower=0., amplitude=1., trust=1., unbiased=True, mintrust=None):
    r"""Renormalize the reward to `[0, 1]`:

    - divide by (`trust/mintrust`) if `unbiased` is `True`.
    - simply project to `[0, 1]` if `unbiased` is `False`,

    .. warning:: If `mintrust` is unknown, the unbiased estimator CANNOT be projected back to a bounded interval.
    """
    if unbiased:
        if mintrust is not None:
            return (reward - lower) / (amplitude * (trust / mintrust))
        else:
            return (reward - lower) / (amplitude * trust)
    else:
        return (reward - lower) / amplitude


def unnormalize_reward(reward, lower=0., amplitude=1.):
    r"""Project back reward to `[lower, lower + amplitude]`."""
    return lower + (reward * amplitude)


# --- Parameters for the LearnExp algorithm

# Default values for the parameters

#: self.unbiased is a flag to know if the rewards are used as biased estimator,
#: i.e., just :math:`r_t`, or unbiased estimators, :math:`r_t / p_t`, if :math:`p_t` is the probability of selecting that arm at time :math:`t`.
#: It seemed to work better with unbiased estimators (of course).
UNBIASED = False
UNBIASED = True  # Better


#: Default value for the constant Eta in (0, 1]
ETA = 0.5


# --- LearnExp algorithm

class LearnExp(BasePolicy):
    """ The LearnExp aggregation bandit algorithm, similar to Exp4 but not equivalent."""

    def __init__(self, nbArms, children=None,
                 unbiased=UNBIASED, eta=ETA, prior='uniform',
                 lower=0., amplitude=1.
                 ):
        # Attributes
        self.nbArms = nbArms  #: Number of arms.
        self.lower = lower  #: Lower values for rewards.
        self.amplitude = amplitude  #: Larger values for rewards.
        self.unbiased = unbiased  #: Flag, see above.
        assert 0 < eta < 1, "Error: parameter 'eta' for a LearnExp player was expected to be in (0, 1) but = {:.3g}...".format(eta)  # DEBUG
        if eta is None:
            eta = 1. / nbArms
        self.eta = eta  #: Constant parameter :math:`\eta`.

        self.nbChildren = nbChildren = len(children)  #: Number N of slave algorithms.
        self.rate = eta / nbChildren  #: Constant :math:`\eta / N`, faster computations if it is stored once.
        assert self.rate > 0, "Error: parameter 'rate' for a LearnExp player was expected to be > 0, but = {:.3g}...".format(self.rate)  # DEBUG

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

        self.last_choice = None  #: Remember the index of the last child trusted for a decision.
        # Initialize the arrays
        # Assume uniform prior if not given or if = 'uniform'
        self.trusts = np.full(nbChildren, 1. / nbChildren)  #: Initial trusts in the slaves :math:`p_j^t`. Default to uniform, but a prior can also be given.
        if prior is not None and prior != 'uniform':
            assert len(prior) == nbChildren, "Error: the 'prior' argument given to LearnExp has to be an array of the good size ({}).".format(nbChildren)  # DEBUG
            self.trusts = prior

        self.weights = (self.trusts - self.rate) / (1 - self.eta)  #: Weights :math:`w_j^t`.

    def __str__(self):
        """ Nicely print the name of the algorithm with its relevant parameters."""
        is_unbiased = "" if self.unbiased else ", biased"
        return r"LearnExp($N={}${}, $\eta={:.3g}$)".format(self.nbChildren, is_unbiased, self.eta)

    # --- Start the game

    def startGame(self):
        """ Start the game for each child."""
        # Start all children
        for i in range(self.nbChildren):
            self.children[i].startGame()

    # --- Get a reward

    def getReward(self, arm, reward):
        """ Give reward for each child, and then update the trust probabilities."""
        reward = float(reward)
        new_reward = renormalize_reward(reward, lower=self.lower, amplitude=self.amplitude, unbiased=False)
        # print("  A LearnExp player {} received a reward = {:.3g} on arm {} and trust = {:.3g} on that choice = {}, giving {:.3g} ...".format(self, reward, arm, self.trusts[self.last_choice], self.last_choice, new_reward))  # DEBUG

        # 1. First, give rewards to that slave, with probability rate / trusts
        probability = self.rate / self.trusts[self.last_choice]
        assert 0 <= probability <= 1, "Error: 'probability' = {:.3g} = rate = {:.3g} / trust_j^t = {:.3g} should have been in [0, 1]...".format(probability, self.rate, self.trusts[self.last_choice])  # DEBUG
        if with_probability(probability):
            self.children[self.last_choice].getReward(arm, reward)

        # 2. Then reinitialize this array of losses
        assert 0 <= new_reward <= 1, "Error: the normalized reward {:.3g} was NOT in [0, 1] ...".format(new_reward)  # DEBUG
        loss = (1 - new_reward)
        if self.unbiased:
            loss /= self.trusts[self.last_choice]

        # 3. Update weight of that slave
        self.weights[self.last_choice] *= np.exp(- self.rate * loss)

        # 4. Recomputed the trusts from the weights
        # add uniform mixing of proportion rate=eta/N
        self.trusts = (1 - self.eta) * (self.weights / np.sum(self.weights)) + self.rate
        # self.trusts = trusts / np.sum(trusts)  # XXX maybe this isn't necessary...

        # print("  The most trusted child policy is the {}th with confidence {}...".format(1 + np.argmax(self.trusts), np.max(self.trusts)))  # DEBUG
        assert np.isclose(np.sum(self.trusts), 1), "Error: 'trusts' do not sum to 1 but to {:.3g} instead...".format(np.sum(self.trusts))  # DEBUG
        # print("self.trusts =", self.trusts)  # DEBUG

    # --- Choice of arm methods

    def choice(self):
        """ Trust one of the slave and listen to his `choice`."""
        # 1. first decide who to listen to
        self.last_choice = rn.choice(self.nbChildren, p=self.trusts)
        # 2. then listen to him
        return self.children[self.last_choice].choice()

    def choiceWithRank(self, rank=1):
        """ Trust one of the slave and listen to his `choiceWithRank`."""
        # 1. first decide who to listen to
        self.last_choice = rn.choice(self.nbChildren, p=self.trusts)
        # 2. then listen to him
        return self.children[self.last_choice].choiceWithRank(rank=rank)

    def choiceFromSubSet(self, availableArms='all'):
        """ Trust one of the slave and listen to his `choiceFromSubSet`."""
        # 1. first decide who to listen to
        self.last_choice = rn.choice(self.nbChildren, p=self.trusts)
        # 2. then listen to him
        return self.children[self.last_choice].choiceFromSubSet(availableArms=availableArms)

    def choiceMultiple(self, nb=1):
        """ Trust one of the slave and listen to his `choiceMultiple`."""
        # 1. first decide who to listen to
        self.last_choice = rn.choice(self.nbChildren, p=self.trusts)
        # 2. then listen to him
        return self.children[self.last_choice].choiceMultiple(nb=nb)

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        """ Trust one of the slave and listen to his `choiceIMP`."""
        # 1. first decide who to listen to
        self.last_choice = rn.choice(self.nbChildren, p=self.trusts)
        # 2. then listen to him
        return self.children[self.last_choice].choiceIMP(nb=nb)

    def estimatedOrder(self):
        r""" Trust one of the slave and listen to his `estimatedOrder`.

        - Return the estimate order of the arms, as a permutation on :math:`[0,...,K-1]` that would order the arms by increasing means.
        """
        # 1. first decide who to listen to
        self.last_choice = rn.choice(self.nbChildren, p=self.trusts)
        # 2. then listen to him
        return self.children[self.last_choice].estimatedOrder()

    def estimatedBestArms(self, M=1):
        """ Return a (non-necessarily sorted) list of the indexes of the M-best arms. Identify the set M-best."""
        assert 1 <= M <= self.nbArms, "Error: the parameter 'M' has to be between 1 and K = {}, but it was {} ...".format(self.nbArms, M)  # DEBUG
        order = self.estimatedOrder()
        return order[-M:]
