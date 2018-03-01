# -*- coding: utf-8 -*-
""" My Aggregated bandit algorithm, similar to Exp4 but not exactly equivalent.

The algorithm is a master A, managing several "slave" algorithms, :math:`A_1, ..., A_N`.

- At every step, the prediction of every slave is gathered, and a vote is done to decide A's decision.
- The vote is simply a majority vote, weighted by a trust probability. If :math:`A_i` decides arm :math:`I_i`, then the probability of selecting :math:`k` is the sum of trust probabilities, :math:`P_i`, of every :math:`A_i` for which :math:`I_i = k`.
- The trust probabilities are first uniform, :math:`P_i = 1/N`, and then at every step, after receiving the feedback for *one* arm :math:`k` (the reward), the trust in each slave :math:`A_i` is updated: :math:`P_i` increases if :math:`A_i` advised :math:`k` (:math:`I_i = k`), or decreases if :math:`A_i` advised another arm.

- The detail about how to increase or decrease the probabilities are specified below.

- Reference: [[Aggregation of Multi-Armed Bandits Learning Algorithms for Opportunistic Spectrum Access, Lilian Besson and Emilie Kaufmann and Christophe Moy, 2017]](https://hal.inria.fr/hal-01705292)

.. note::

   Why call it *Aggregator* ?
   Because this algorithm is an efficient *aggregation* algorithm,
   and like The Terminator, he beats his opponents with an iron fist!
   (*OK, that's a stupid joke but a cool name, thanks Emilie!*)

   .. image::  https://media.giphy.com/media/YoB1eEFB6FZ1m/giphy.gif
      :target: https://en.wikipedia.org/wiki/Terminator_T-800_Model_101
      :alt:    https://en.wikipedia.org/wiki/Terminator_T-800_Model_101

.. note::

   I wanted to call it *Aggragorn*.
   Because this algorithm is like `Aragorn the ranger <https://en.wikipedia.org/wiki/Aragorn>`_,
   it starts like a simple bandit, but soon it will become king!!

   .. image::  https://media.giphy.com/media/12GSQqUY9CSmJO/giphy.gif
      :target: https://en.wikipedia.org/wiki/Aragorn
      :alt:    https://en.wikipedia.org/wiki/Aragorn
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
import numpy.random as rn
from .BasePolicy import BasePolicy


# Default values for the parameters

#: A flag to know if the rewards are used as biased estimator,
#: i.e., just :math:`r_t`, or unbiased estimators, :math:`r_t / p_t`, if :math:`p_t` is the probability of selecting that arm at time :math:`t`.
#: It seemed to work better with unbiased estimators (of course).
UNBIASED = False
UNBIASED = True    # Better

#: Flag to know if we should update the trusts proba like in Exp4 or like in my initial Aggregator proposal
#:
#: - First choice: like Exp4, trusts are fully recomputed, ``trusts^(t+1) = exp(rate_t * estimated mean rewards upto time t)``,
#: - Second choice: my proposal, trusts are just updated multiplicatively, ``trusts^(t+1) <-- trusts^t * exp(rate_t * estimate instant reward at time t)``.
#:
#: Both choices seem fine, and anyway the trusts are renormalized to be a probability distribution, so it doesn't matter much.
UPDATE_LIKE_EXP4 = True
UPDATE_LIKE_EXP4 = False  # Better

#: Non parametric flag to know if the Exp4-like update uses losses or rewards.
#: Losses are ``1 - reward``, in which case the ``rate_t`` is negative.
USE_LOSSES = True
USE_LOSSES = False

#: Should all trusts be updated, or only the trusts of slaves Ai who advised the decision ``Aggregator[A1..AN]`` followed.
UPDATE_ALL_CHILDREN = False


class Aggregator(BasePolicy):
    """ My Aggregated bandit algorithm, similar to Exp4 but not exactly equivalent."""

    def __init__(self, nbArms, children=None,
                 learningRate=None, decreaseRate=None, horizon=None,
                 update_all_children=UPDATE_ALL_CHILDREN, update_like_exp4=UPDATE_LIKE_EXP4,
                 unbiased=UNBIASED, prior='uniform',
                 lower=0., amplitude=1.,
                 extra_str=''
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
        self.horizon = int(horizon) if horizon is not None else None  #: Horizon T, if given and not None, can be used to compute a "good" constant learning rate, :math:`\sqrt{\frac{2 \log(N)}{T K}}` for N slaves, K arms (heuristic).
        self.extra_str = extra_str  #: A string to add at the end of the ``str(self)``, to specify which algorithms are aggregated for instance.
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
            elif callable(child):
                print("  Using this delayed function to create player 'children[{}]' = {} ...".format(i, child))  # DEBUG
                self.children.append(child())
            else:
                print("  Using this already created player 'children[{}]' = {} ...".format(i, child))  # DEBUG
                self.children.append(child)
        # Initialize the arrays
        # Assume uniform prior if not given or if = 'uniform'
        self.trusts = np.full(self.nbChildren, 1. / self.nbChildren)  #: Initial trusts in the slaves. Default to uniform, but a prior can also be given.
        if prior is not None and prior != 'uniform':
            assert len(prior) == self.nbChildren, "Error: the 'prior' argument given to Aggregator has to be an array of the good size ({}).".format(self.nbChildren)  # DEBUG
            self.trusts = prior
        # Internal vectorial memory
        self.choices = np.full(self.nbChildren, -10000, dtype=int)  #: Keep track of the last choices of each slave, to know whom to update if update_all_children is false.
        if self.update_like_exp4:
            self.children_cumulated_losses = np.zeros(self.nbChildren)  #: Keep track of the cumulated loss (empirical mean)
        self.index = np.zeros(nbArms)  #: Numerical index for each arms

    # Print, different output according to the parameters
    def __str__(self):
        """ Nicely print the name of the algorithm with its relevant parameters."""
        name = "Exp4" if self.update_like_exp4 else "Aggregator"
        all_children = ", update all" if self.update_all_children else ""
        if self.decreaseRate == 'auto':
            if self.horizon:
                s = r"{}($T={}$, $N={}${})".format(name, self.horizon, self.nbChildren, all_children)
            else:
                s = r"{}($N={}${})".format(name, self.nbChildren, all_children)
        elif self.decreaseRate is not None:
            s = r"{}($N={}${}, $\eta={:.3g}$, $dRate={:.3g}$)".format(name, self.nbChildren, all_children, self.learningRate, self.decreaseRate)
        else:
            s = r"{}($N={}${}, $\eta={:.3g}$)".format(name, self.nbChildren, all_children, self.learningRate)
        return s + self.extra_str

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def rate(self):
        """ Learning rate, can be constant if self.decreaseRate is None, or decreasing.

        - if horizon is known, use the formula which uses it,
        - if horizon is not known, use the formula which uses current time :math:`t`,
        - else, if decreaseRate is a number, use an exponentionally decreasing learning rate, ``rate = learningRate * exp(- t / decreaseRate)``. Bad.
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
        self.index.fill(0)

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
                trusts = np.exp(-1.0 * rate * self.children_cumulated_losses)
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
        """ Make each child vote, then sample the decision by `importance sampling <https://en.wikipedia.org/wiki/Importance_sampling>`_ on their votes with the trust probabilities."""
        # 1. make vote every child
        self._makeChildrenChoose()
        # 2. select the vote to trust, randomly
        return rn.choice(self.choices, p=self.trusts)

    def choiceWithRank(self, rank=1):
        """ Make each child vote, with rank, then sample the decision by `importance sampling <https://en.wikipedia.org/wiki/Importance_sampling>`_ on their votes with the trust probabilities."""
        if rank == 1:
            return self.choice()
        else:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceWithRank(rank)
            return rn.choice(self.choices, p=self.trusts)

    def choiceFromSubSet(self, availableArms='all'):
        """ Make each child vote, on subsets of arms, then sample the decision by `importance sampling <https://en.wikipedia.org/wiki/Importance_sampling>`_ on their votes with the trust probabilities."""
        if (availableArms == 'all') or (len(availableArms) == self.nbArms):
            return self.choice()
        else:
            for i, child in enumerate(self.children):
                self.choices[i] = child.choiceFromSubSet(availableArms)
            return rn.choice(self.choices, p=self.trusts)

    def choiceMultiple(self, nb=1):
        """ Make each child vote, multiple times, then sample the decision by `importance sampling <https://en.wikipedia.org/wiki/Importance_sampling>`_ on their votes with the trust probabilities."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            choices = [None] * self.nbChildren
            for i, child in enumerate(self.children):
                choices[i] = child.choiceMultiple(nb)
                self.choices[i] = choices[i][0]
            this_choices = choices[rn.choice(self.nbChildren, replace=False, p=self.trusts)]
            return this_choices

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        """ Make each child vote, multiple times (with IMP scheme), then sample the decision by `importance sampling <https://en.wikipedia.org/wiki/Importance_sampling>`_ on their votes with the trust probabilities."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            choices = [None] * self.nbChildren
            for i, child in enumerate(self.children):
                choices[i] = child.choiceIMP(nb)
                self.choices[i] = choices[i][0]
            this_choices = choices[rn.choice(self.nbChildren, replace=False, p=self.trusts)]
            return this_choices

    def estimatedOrder(self):
        """ Make each child vote for their estimate order of the arms, then randomly select an ordering by `importance sampling <https://en.wikipedia.org/wiki/Importance_sampling>`_ with the trust probabilities.
        Return the estimate order of the arms, as a permutation on ``[0..K-1]`` that would order the arms by increasing means."""
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

    def estimatedBestArms(self, M=1):
        """ Return a (non-necessarily sorted) list of the indexes of the M-best arms. Identify the set M-best."""
        assert 1 <= M <= self.nbArms, "Error: the parameter 'M' has to be between 1 and K = {}, but it was {} ...".format(self.nbArms, M)  # DEBUG
        order = self.estimatedOrder()
        return order[-M:]

    def computeIndex(self, arm):
        """ Compute the current index of arm 'arm', by computing all the indexes of the children policies, and computing a convex combination using the trusts probabilities."""
        indexes = [None] * self.nbChildren
        for i, child in enumerate(self.children):
            indexes[i] = child.computeIndex(arm)
        index = np.dot(indexes, self.trusts)
        return index

    def computeAllIndex(self):
        """ Compute the current indexes for all arms. Possibly vectorized, by default it can *not* be vectorized automatically."""
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)

    def handleCollision(self, arm, reward=None):
        """ Default to give a 0 reward (or ``self.lower``)."""
        # FIXME not clear why it should be like giving a zero reward to the master policy,
        super(Aggregator, self).handleCollision(arm, reward=reward)
        # FIXME maybe giving the collision information to all children players is enough...
        for child in self.children:
            child.handleCollision(arm, reward=reward)
