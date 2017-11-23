# -*- coding: utf-8 -*-
r""" The CORRAL aggregation bandit algorithm, similar to Exp4 but not exactly equivalent.

The algorithm is a master A, managing several "slave" algorithms, :math:`A_1, ..., A_N`.

- At every step, one slave algorithm is selected, by a random selection from a trust distribution on :math:`[1,...,N]`.
- Then its decision is listen to, played by the master algorithm, and a feedback reward is received.
- The reward is reweighted by the trust of the listened algorithm, and given back to it.
- The other slaves, whose decision was not even asked, receive a zero reward, or no reward at all.
- The trust probabilities are first uniform, :math:`P_i = 1/N`, and then at every step, after receiving the feedback for *one* arm k (the reward), the trust in each slave Ai is updated: :math:`P_i` by the reward received.
- The detail about how to increase or decrease the probabilities are specified in the reference article.

.. note:: Reference: [["Corralling a Band of Bandit Algorithms", by A. Agarwal, H. Luo, B. Neyshabur, R.E. Schapire, 01.2017](https://arxiv.org/abs/1612.06246v2)].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
import numpy.random as rn
from scipy.optimize import minimize_scalar
from .BasePolicy import BasePolicy


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


# --- Log-Barrier-OMD

def log_Barrier_OMB(trusts, losses, rates):
    r""" A step of the *log-barrier Online Mirror Descent*, updating the trusts:

    - Find :math:`\lambda \in [\min_i l_{t,i}, \max_i l_{t,i}]` such that :math:`\sum_i \frac{1}{1/p_{t,i} + \eta_{t,i}(l_{t,i} - \lambda)} = 1`.
    - Return :math:`\mathbf{p}_{t+1,i}` such that :math:`\frac{1}{p_{t+1,i}} = \frac{1}{p_{t,i}} + \eta_{t,i}(l_{t,i} - \lambda)`.

    - Note: uses :func:`scipy.optimize.minimize_scalar` for the optimization.
    - Reference: [Learning in games: Robustness of fast convergence, by D.Foster, Z.Li, T.Lykouris, K.Sridharan, and E.Tardos, NIPS 2016].
    """
    min_loss = max(0, np.min(losses))
    max_loss = np.max(losses)
    def objective(a_loss):
        """Objective function of the loss."""
        lhs = np.sum(1. / ((1. / trusts) + rates * (losses - a_loss)))
        rhs = 1.
        # return np.abs(lhs - rhs)
        return (lhs - rhs) ** 2
    assert min_loss <= max_loss, "Error: the interval [min_loss, max_loss] = [{:.3g}, {:.3g}] is not a valid constraint...".format(min_loss, max_loss)  # DEBUG
    result = minimize_scalar(objective, bounds=(min_loss, max_loss), method='bounded')
    best_loss = result.x
    assert min_loss <= best_loss <= max_loss, "Error: the loss 'lambda={:.3g}' was supposed to be found in [min_loss, max_loss] = [{:.3g}, {:.3g}]...".format(best_loss, min_loss, max_loss)  # DEBUG

    new_trusts = 1. / ((1. / trusts) + rates * (losses - best_loss))

    new_trusts /= np.sum(new_trusts)
    assert np.isclose(np.sum(new_trusts), 1), "Error: the new trusts vector = {} was supposed to sum to 1 but does not...".format(list(new_trusts))  # DEBUG

    if not np.all(new_trusts >= 0):
        print("Warning: the new trusts vector = {} was supposed to be a valid probability >= 0, but it is not... Let's cheat!".format(list(new_trusts)))  # DEBUG)
        x = np.min(new_trusts)
        assert x < 0
        new_trusts /= np.abs(x)
        new_trusts += 1
        assert np.isclose(np.min(new_trusts), 0)
        assert np.all(new_trusts >= 0)
        new_trusts /= np.sum(new_trusts)
    assert np.all(new_trusts >= 0), "Error: the new trusts vector = {} was supposed to be a valid probability >= 0, but it is not...".format(list(new_trusts))  # DEBUG

    assert np.all(new_trusts <= 1), "Error: the new trusts vector = {} was supposed to be a valid probability <= 1, but it is not...".format(list(new_trusts))  # DEBUG
    return new_trusts


# --- Parameters for the CORRAL algorithm

# Default values for the parameters

#: self.unbiased is a flag to know if the rewards are used as biased estimator,
#: i.e., just :math:`r_t`, or unbiased estimators, :math:`r_t / p_t`, if :math:`p_t` is the probability of selecting that arm at time :math:`t`.
#: It seemed to work better with unbiased estimators (of course).
UNBIASED = False
UNBIASED = True  # Better


#: Whether to give back a reward to only one slave algorithm (default, `False`) or to all slaves who voted for the same arm
BROADCAST_ALL = True
BROADCAST_ALL = False


# --- CORRAL algorithm

class CORRAL(BasePolicy):
    """ The CORRAL aggregation bandit algorithm, similar to Exp4 but not exactly equivalent."""

    def __init__(self, nbArms, children=None,
                 horizon=None, rate=None,
                 unbiased=UNBIASED, broadcast_all=BROADCAST_ALL, prior='uniform',
                 lower=0., amplitude=1.
                ):
        # Attributes
        self.nbArms = nbArms  #: Number of arms.
        self.lower = lower  #: Lower values for rewards.
        self.amplitude = amplitude  #: Larger values for rewards.
        self.unbiased = unbiased  #: Flag, see above.
        self.broadcast_all = broadcast_all  #: Flag, see above.

        # FIXED I should make this algorithm subject to be used with DoublingTrickWrapper, by changing these if self.horizon is changed
        self.gamma = 1. / horizon  #: Constant :math:`\gamma = 1 / T`.
        assert self.gamma < 1, "Error: parameter 'gamma' for a CORRAL player was expected to be < 1, but = {:.3g}...".format(self.gamma)  # DEBUG
        self.beta = np.exp(1. / np.log(horizon))  #: Constant :math:`\beta = \exp(1 / \log(T))`.
        assert self.beta > 1, "Error: parameter 'beta' for a CORRAL player was expected to be > 1, but = {:.3g}...".format(self.beta)  # DEBUG

        self._default_parameters = True
        self.nbChildren = nbChildren = len(children)  #: Number N of slave algorithms.
        if rate is None:
            # Use the default horizon-dependent rate value
            # rate = np.sqrt(nbChildren / (nbArms * horizon))
            rate = np.sqrt(nbChildren / horizon)
        else:
            self._default_parameters = False
        assert rate > 0, "Error: parameter 'rate' for a CORRAL player was expected to be > 0, but = {:.3g}...".format(rate)  # DEBUG
        self.rates = np.full(nbChildren, rate)  #: Value of the learning rate (will be **increasing** in time).

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
        # Assume uniform prior if not given or if = 'uniform'
        self.trusts = np.full(nbChildren, 1. / nbChildren)  #: Initial trusts in the slaves. Default to uniform, but a prior can also be given.
        if prior is not None and prior != 'uniform':
            assert len(prior) == nbChildren, "Error: the 'prior' argument given to CORRAL has to be an array of the good size ({}).".format(nbChildren)  # DEBUG
            self.trusts = prior
        self.bar_trusts = np.copy(self.trusts)  #: Initial bar trusts in the slaves. Default to uniform, but a prior can also be given.

        # Internal vectorial memory
        self.choices = np.full(self.nbChildren, -10000, dtype=int)  #: Keep track of the last choices of each slave, to know whom to update if update_all_children is false.

        # Internal memory, additionally to what is found not in Aggregator
        self.last_choice = None  #: Remember the index of the last child trusted for a decision.
        self.losses = np.zeros(nbChildren)  #: For the log-barrier OMD step, a vector of losses has to be given. Faster to keep it as an attribute instead of reallocating it every time.
        self.rhos = self.bar_trusts / 2  #: I use the inverses of the :math:`\rho_{t,i}` from the Algorithm in the reference article. Simpler to understand, less numerical errors.

    def __str__(self):
        """ Nicely print the name of the algorithm with its relevant parameters."""
        is_unbiased = "" if self.unbiased else ", biased"
        is_broadcast_all = "broadcast to all" if self.broadcast_all else "broadcast to one"
        if self._default_parameters:
            return r"CORRAL($N={}${}, {})".format(self.nbChildren, is_unbiased, is_broadcast_all)
        else:
            if len(set(self.rhos)) > 1 or len(set(self.rates)) > 1:
                return r"CORRAL($N={}${}, {}, $\gamma=1/T$, $\beta={:.3g}$, $\rho={}$, $\eta={}$)".format(self.nbChildren, is_unbiased, is_broadcast_all, self.beta, list(self.rhos), list(self.rates))
            else:
                return r"CORRAL($N={}${}, {}, $\gamma=1/T$, $\beta={:.3g}$, $\rho={:.2g}$, $\eta={:.2g}$)".format(self.nbChildren, is_unbiased, is_broadcast_all, self.beta, self.rhos[0], self.rates[0])

    def __setattr__(self, name, value):
        r"""Trick method, to update the :math:`\gamma` and :math:`\beta` parameters of the CORRAL algorithm if the horizon T changes.

        - This is here just to eventually allow :class:`Policies.DoublingTrickWrapper` to be used with a CORRAL player.

        .. warning:: Not tested yet!
        """
        if name in ['horizon', '_horizon']:
            horizon = float(value)
            self.gamma = 1. / horizon  #: Constant :math:`\gamma = 1 / T`.
            self.beta = np.exp(1. / np.log(horizon))  #: Constant :math:`\beta = \exp(1 / \log(T))`.
        else:
            # self.__dict__[name] = value  # <-- old style class
            object.__setattr__(self, name, value)  # <-- new style class

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
        # new_reward = renormalize_reward(reward, lower=self.lower, amplitude=self.amplitude, trust=self.bar_trusts[self.last_choice], unbiased=self.unbiased)
        # , mintrust=(self.gamma / self.nbChildren)  # XXX

        # print("  A CORRAL player {} received a reward = {:.3g} on arm {} and trust = {:.3g} on that choice = {}, giving {:.3g} ...".format(self, reward, arm, self.bar_trusts[self.last_choice], self.last_choice, new_reward))  # DEBUG
        # 1. First, give rewards to all children
        if self.broadcast_all:
            for i, child in enumerate(self.children):
                # # if i == self.last_choice:
                # if self.choices[i] == arm:
                #     # Give reward, biased or not
                #     # child.getReward(arm, unnormalize_reward(new_reward, lower=self.lower, amplitude=self.amplitude))
                child.getReward(arm, reward)
                #     child.getReward(arm, reward)
                # else:  # give 0 reward to all other children
                #     child.getReward(arm, 0)  # <-- this is a bad idea!
        else:
            # XXX this makes WAY more sense!
            self.children[self.last_choice].getReward(arm, reward)

        # 2. Then reinitialize this array of losses
        self.losses[:] = 0
        assert 0 <= new_reward <= 1, "Error: the normalized reward {:.3g} was NOT in [0, 1] ...".format(new_reward)  # DEBUG
        if self.broadcast_all:
            self.losses[self.choices == arm] = (1 - new_reward)
            if self.unbiased:
                self.losses[self.choices == arm] /= self.bar_trusts[self.choices == arm]
        else:
            self.losses[self.last_choice] = (1 - new_reward)
            if self.unbiased:
                self.losses[self.last_choice] /= self.bar_trusts[self.last_choice]

        # 3. Compute the new trust proba, with a log-barrier Online-Mirror-Descent step
        trusts = log_Barrier_OMB(self.trusts, self.losses, self.rates)
        # 4. renormalize self.trusts to make it a proba dist
        # In practice, it also decreases the self.trusts for the children who were wrong
        self.trusts = trusts / np.sum(trusts)  # XXX maybe this isn't necessary...

        # add uniform mixing of proportion gamma
        bar_trusts = (1 - self.gamma) * self.trusts + (self.gamma / self.nbChildren)
        self.bar_trusts = bar_trusts / np.sum(bar_trusts)  # XXX maybe this isn't necessary...

        # 5. Compare trusts with the self.rhos values to compute the new learning rates and rhos
        for i in range(self.nbChildren):
            if self.bar_trusts[i] < self.rhos[i]:
                # print("  For child #i = {}, the sampling trust was = {:.3g}, smaller than the threshold rho = {:.3g} so the learning rate is increased from {:.3g} to {:.3g}, and the threshold is now {:.3g} ...".format(i, self.bar_trusts[i], self.rhos[i], self.rates[i], self.rates[i] * self.beta, self.bar_trusts[i] / 2.))  # DEBUG
                self.rhos[i] = self.bar_trusts[i] / 2.
                self.rates[i] *= self.beta  # increase the rate for this guy
            # else:  # nothing to do
            #     self.rhos[i] = self.rhos[i]
            #     self.rates[i] = self.rates[i]

        # print("  The most trusted child policy is the {}th with confidence {}...".format(1 + np.argmax(self.bar_trusts), np.max(self.bar_trusts)))  # DEBUG
        assert np.isclose(np.sum(self.bar_trusts), 1), "Error: 'bar_trusts' do not sum to 1 but to {:.3g} instead...".format(np.sum(self.bar_trusts))  # DEBUG
        # print("self.bar_trusts =", self.bar_trusts)  # DEBUG
        assert np.isclose(np.sum(self.trusts), 1), "Error: 'trusts' do not sum to 1 but to {:.3g} instead...".format(np.sum(self.trusts))  # DEBUG
        # print("self.trusts =", self.trusts)  # DEBUG

    # --- Choice of arm methods

    def choice(self):
        """ Trust one of the slave and listen to his `choice`."""
        # 1. first decide who to listen to
        self.last_choice = rn.choice(self.nbChildren, p=self.bar_trusts)
        if self.broadcast_all:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choice()
        else:
        # 2. then listen to him
            self.choices[self.last_choice] = self.children[self.last_choice].choice()
        return self.choices[self.last_choice]

    def choiceWithRank(self, rank=1):
        """ Trust one of the slave and listen to his `choiceWithRank`."""
        # 1. first decide who to listen to
        self.last_choice = rn.choice(self.nbChildren, p=self.bar_trusts)
        if self.broadcast_all:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceWithRank(rank=rank)
        else:
        # 2. then listen to him
            self.choices[self.last_choice] = self.children[self.last_choice].choiceWithRank(rank=rank)
        return self.choices[self.last_choice]

    def choiceFromSubSet(self, availableArms='all'):
        """ Trust one of the slave and listen to his `choiceFromSubSet`."""
        # 1. first decide who to listen to
        self.last_choice = rn.choice(self.nbChildren, p=self.bar_trusts)
        if self.broadcast_all:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceFromSubSet(availableArms=availableArms)
        else:
        # 2. then listen to him
            self.choices[self.last_choice] = self.children[self.last_choice].choiceFromSubSet(availableArms=availableArms)
        return self.choices[self.last_choice]

    def choiceMultiple(self, nb=1):
        """ Trust one of the slave and listen to his `choiceMultiple`."""
        # 1. first decide who to listen to
        self.last_choice = rn.choice(self.nbChildren, p=self.bar_trusts)
        if self.broadcast_all:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceMultiple(nb=nb)
        else:
        # 2. then listen to him
            self.choices[self.last_choice] = self.children[self.last_choice].choiceMultiple(nb=nb)
        return self.choices[self.last_choice]

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        """ Trust one of the slave and listen to his `choiceIMP`."""
        # 1. first decide who to listen to
        self.last_choice = rn.choice(self.nbChildren, p=self.bar_trusts)
        if self.broadcast_all:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceIMP(nb=nb)
        else:
        # 2. then listen to him
            self.choices[self.last_choice] = self.children[self.last_choice].choiceIMP(nb=nb)
        return self.choices[self.last_choice]

    def estimatedOrder(self):
        r""" Trust one of the slave and listen to his `estimatedOrder`.

        - Return the estimate order of the arms, as a permutation on :math:`[0,...,K-1]` that would order the arms by increasing means.
        """
        # 1. first decide who to listen to
        self.last_choice = rn.choice(self.nbChildren, p=self.bar_trusts)
        # 2. then listen to him
        return self.children[self.last_choice].estimatedOrder()

    def estimatedBestArms(self, M=1):
        """ Return a (non-necessarily sorted) list of the indexes of the M-best arms. Identify the set M-best."""
        assert 1 <= M <= self.nbArms, "Error: the parameter 'M' has to be between 1 and K = {}, but it was {} ...".format(self.nbArms, M)  # DEBUG
        order = self.estimatedOrder()
        return order[-M:]
