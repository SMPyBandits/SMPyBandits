# -*- coding: utf-8 -*-
""" Base class for any wrapper policy.

- It encapsulates another policy, and defer all methods calls to the underlying policy.
- For instance, see :class:`Policies.SparseWrapper`, :class:`Policies.DoublingTrickWrapper` or :class:`Policies.SlidingWindowRestart`.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np

try:
    from .BasePolicy import BasePolicy
    from .UCB import UCB as DefaultPolicy
except ImportError:
    from BasePolicy import BasePolicy
    from UCB import UCB as DefaultPolicy


class BaseWrapperPolicy(BasePolicy):
    """ Base class for any wrapper policy."""

    def __init__(self, nbArms, policy=DefaultPolicy, *args, **kwargs):
        super(BaseWrapperPolicy, self).__init__(nbArms, *args, **kwargs)
        # --- Policy
        self._policy = policy  # Class to create the underlying policy
        self._args = args  # To keep them
        if 'params' in kwargs:
            kwargs.update(kwargs['params'])
            del kwargs['params']
        self._kwargs = kwargs  # To keep them
        self.policy = None

    # --- Start game by creating new underlying policy

    def startGame(self, createNewPolicy=True):
        """ Initialize the policy for a new game.

        .. warning:: ``createNewPolicy=True`` creates a new object for the underlying policy, while ``createNewPolicy=False`` only call :meth:`BasePolicy.startGame`.
        """
        super(BaseWrapperPolicy, self).startGame()
        # now for the underlying policy
        if createNewPolicy or (self.policy is None):
            # if self.policy is not None: print("INFO: BaseWrapperPolicy: creating a new underlying policy with startGame(createNewPolicy=True)...")  # DEBUG
            # del self.policy  # XXX be sure that we delete the attribute and the object?
            self.policy = self._policy(self.nbArms, *self._args, **self._kwargs)
        # now also start game for the underlying policy
        self.policy.startGame()

    # --- Pass the call to the subpolicy

    def getReward(self, arm, reward):
        """ Pass the reward, as usual, update t and sometimes restart the underlying policy."""
        # print(" - At time t = {}, got a reward = {} from arm {} ...".format(self.t, arm, reward))  # DEBUG
        super(BaseWrapperPolicy, self).getReward(arm, reward)
        self.policy.getReward(arm, reward)

    # --- Sub methods

    def choice(self):
        r""" Pass the call to ``choice`` of the underlying policy."""
        return self.policy.choice()

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def index(self):
        r""" Get attribute ``index`` from the underlying policy."""
        return self.policy.index

    def choiceWithRank(self, rank=1):
        r""" Pass the call to ``choiceWithRank`` of the underlying policy."""
        return self.policy.choiceWithRank(rank=rank)

    def choiceFromSubSet(self, availableArms='all'):
        r""" Pass the call to ``choiceFromSubSet`` of the underlying policy."""
        return self.policy.choiceFromSubSet(availableArms=availableArms)

    def choiceMultiple(self, nb=1):
        r""" Pass the call to ``choiceMultiple`` of the underlying policy."""
        return self.policy.choiceMultiple(nb=nb)

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        r""" Pass the call to ``choiceIMP`` of the underlying policy."""
        return self.policy.choiceIMP(nb=nb, startWithChoiceMultiple=startWithChoiceMultiple)

    def estimatedOrder(self):
        r""" Pass the call to ``estimatedOrder`` of the underlying policy."""
        return self.policy.estimatedOrder()

    def estimatedBestArms(self, M=1):
        r""" Pass the call to ``estimatedBestArms`` of the underlying policy."""
        return self.policy.estimatedBestArms(M=M)

    def computeIndex(self, arm):
        r""" Pass the call to ``computeIndex`` of the underlying policy."""
        return self.policy.computeIndex(arm)

    def computeAllIndex(self):
        r""" Pass the call to ``computeAllIndex`` of the underlying policy."""
        return self.policy.computeAllIndex()

