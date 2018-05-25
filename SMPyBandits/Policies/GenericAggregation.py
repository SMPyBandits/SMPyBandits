# -*- coding: utf-8 -*-
r""" The GenericAggregation aggregation bandit algorithm: use a bandit policy A (master), managing several "slave" algorithms, :math:`A_1, ..., A_N`.

- At every step, one slave algorithm A_i is selected, by the master policy A.
- Then its decision is listen to, played by the master algorithm, and a feedback reward is received.
- All slaves receive the observation (arm, reward).
- The master also receives the same observation.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from random import random
import numpy as np
import numpy.random as rn
try:
    from .BasePolicy import BasePolicy
    from .with_proba import with_proba
except ImportError:
    from BasePolicy import BasePolicy
    from with_proba import with_proba


# --- GenericAggregation algorithm

class GenericAggregation(BasePolicy):
    """ The GenericAggregation aggregation bandit algorithm."""

    def __init__(self, nbArms, master=None, children=None,
            lower=0., amplitude=1.
        ):
        # Attributes
        self.nbArms = nbArms  #: Number of arms.
        self.lower = lower  #: Lower values for rewards.
        self.amplitude = amplitude  #: Larger values for rewards.
        self.last_choice = 0  #: Remember the index of the last child trusted for a decision.
        self.nbChildren = nbChildren = len(children)  #: Number N of slave algorithms.

        # Internal object memory
        self.master = None
        if isinstance(master, dict):
            print("  Creating this master player from a dictionnary 'master' = {} ...".format(master))  # DEBUG
            localparams = {'lower': lower, 'amplitude': amplitude}
            localparams.update(master['params'])
            self.master = master['archtype'](nbChildren, **localparams)
        elif isinstance(master, type):
            print("  Using this not-yet created player 'master' = {} ...".format(master))  # DEBUG
            self.master = master(nbChildren, lower=lower, amplitude=amplitude)  # Create it here
        else:
            print("  Using this already created player 'master' = {} ...".format(master))  # DEBUG
            self.master = master

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

    def __str__(self):
        """ Nicely print the name of the algorithm with its relevant parameters."""
        return r"GenericAggr({}, $N={}$)".format(self.master, self.nbChildren)

    # --- Start the game

    def startGame(self):
        """ Start the game for each child, and for the master."""
        self.master.startGame()
        for i in range(self.nbChildren):
            self.children[i].startGame()

    # --- Get a reward

    def getReward(self, arm, reward):
        """ Give reward for each child, and for the master."""
        self.master.getReward(self.last_choice, reward)
        for i in range(self.nbChildren):
            self.children[i].getReward(arm, reward)

    # --- Choice of arm methods

    def choice(self):
        """ Trust one of the slave and listen to his `choice`."""
        # 1. first decide who to listen to
        self.last_choice = self.master.choice()
        # 2. then listen to him
        return self.children[self.last_choice].choice()

    def choiceWithRank(self, rank=1):
        """ Trust one of the slave and listen to his `choiceWithRank`."""
        # 1. first decide who to listen to
        self.last_choice = self.master.choice()
        # 2. then listen to him
        return self.children[self.last_choice].choiceWithRank(rank=rank)

    def choiceFromSubSet(self, availableArms='all'):
        """ Trust one of the slave and listen to his `choiceFromSubSet`."""
        # 1. first decide who to listen to
        self.last_choice = self.master.choice()
        # 2. then listen to him
        return self.children[self.last_choice].choiceFromSubSet(availableArms=availableArms)

    def choiceMultiple(self, nb=1):
        """ Trust one of the slave and listen to his `choiceMultiple`."""
        # 1. first decide who to listen to
        self.last_choice = self.master.choice()
        # 2. then listen to him
        return self.children[self.last_choice].choiceMultiple(nb=nb)

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        """ Trust one of the slave and listen to his `choiceIMP`."""
        # 1. first decide who to listen to
        self.last_choice = self.master.choice()
        # 2. then listen to him
        return self.children[self.last_choice].choiceIMP(nb=nb)

    def estimatedOrder(self):
        r""" Trust one of the slave and listen to his `estimatedOrder`.

        - Return the estimate order of the arms, as a permutation on :math:`[0,...,K-1]` that would order the arms by increasing means.
        """
        # 1. first decide who to listen to
        self.last_choice = self.master.choice()
        # 2. then listen to him
        return self.children[self.last_choice].estimatedOrder()

    def estimatedBestArms(self, M=1):
        """ Return a (non-necessarily sorted) list of the indexes of the M-best arms. Identify the set M-best."""
        assert 1 <= M <= self.nbArms, "Error: the parameter 'M' has to be between 1 and K = {}, but it was {} ...".format(self.nbArms, M)  # DEBUG
        order = self.estimatedOrder()
        return order[-M:]
