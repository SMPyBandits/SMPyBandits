# -*- coding: utf-8 -*-
r""" A policy that acts as a wrapper on another policy `P`, which requires to know the range :math:`[a, b]` of the rewards, by implementing a "doubling trick" to adapt to an unknown range of rewards.

It's an interesting variant of the "doubling trick", used to tackle another unknown aspect of sequential experiments: some algorithms need to use rewards in :math:`[0,1]`, and are easy to use if the rewards known to be in some interval :math:`[a, b]` (I did this from the very beginning here, with ``[lower, lower+amplitude]``).
But if the interval :math:`[a,b]` is unknown, what can we do?
The "Doubling Trick", in this setting, refers to this algorithm:

1. Start with :math:`[a_0, b_0] = [0, 1]`,
2. If a reward :math:`r_t` is seen below :math:`a_i`, use :math:`a_{i+1} = r_t`,
3. If a reward :math:`r_t` is seen above :math:`b_i`, use :math:`b_{i+1} = r_t - a_i`.

Instead of just doubling the length of the interval ("doubling trick"), we use :math:`[r_t, b_i]` or :math:`[a_i, r_t]` as it is the smallest interval compatible with the past and the new observation :math:`r_t`

- Reference.  I'm not sure which work is the first to have proposed this idea, but [[Normalized online learning, Stéphane Ross & Paul Mineiro & John Langford, 2013](https://arxiv.org/pdf/1305.6646.pdf)] proposes a similar idea.

.. seealso:: See for instance `Obandit.WrapRange <https://freuk.github.io/obandit/api.docdir/Obandit.WrapRange.html>`_ by `@freuk <https://github.com/freuk/>`_.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


import numpy as np
from .BasePolicy import BasePolicy

from .UCB import UCB
default_rangeDependent_policy = UCB


# --- The interesting class

class WrapRange(BasePolicy):
    r""" A policy that acts as a wrapper on another policy `P`, which requires to know the range :math:`[a, b]` of the rewards, by implementing a "doubling trick" to adapt to an unknown range of rewards.
    """

    def __init__(self, nbArms,
                 policy=default_rangeDependent_policy,
                 lower=0., amplitude=1.,
                 *args, **kwargs):
        super(WrapRange, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # --- Policy
        self._policy = policy  # Class to create the underlying policy
        self._args = args  # To keep them
        if 'params' in kwargs:
            kwargs.update(kwargs['params'])
            del kwargs['params']
        self._kwargs = kwargs  # To keep them
        self.policy = None  #: Underlying policy
        # --- Horizon
        self._i = 0
        # XXX Force it, just for pretty printing...
        self.startGame()

    # --- pretty printing

    def __str__(self):
        return r"WrapRange[{}]".format(self.policy)

    # --- Start game by creating new underlying policy

    def startGame(self):
        """ Initialize the policy for a new game."""
        super(WrapRange, self).startGame()
        self._i = 0  # reinitialize this
        self.policy = self._policy(self.nbArms, lower=self.lower, amplitude=self.amplitude, *self._args, **self._kwargs)
        # now also start game for the underlying policy
        self.policy.startGame()
        self.rewards = self.policy.rewards  # just pointers to the underlying arrays!
        self.pulls = self.policy.pulls      # just pointers to the underlying arrays!

    # --- Pass the call to the subpolicy

    def getReward(self, arm, reward):
        r""" Maybe change the current range and rescale all the past history, and then pass the reward, and update t.

        Let call :math:`r_s` the reward at time :math:`s`, :math:`l_{t-1}` and :math:`a_{t-1}` the lower-bound and amplitude of rewards at previous time :math:`t-1`, and :math:`l_t` and :math:`a_t` the new lower-bound and amplitude for current time :math:`t`.
        The previous history is :math:`R_t := \sum_{s=1}^{t-1} r_s`.

        The generic formula for rescaling the previous history is the following:

        .. math:: R_t := \frac{(a_{t-1} \times R_t + l_{t-1}) - l_t}{a_t}.

        So we have the following efficient algorithm:

        1. If :math:`r < l_{t-1}`, let :math:`l_t = r` and :math:`R_t := R_t + \frac{l_{t-1} - l_t}{a_t}`,
        2. Else if :math:`r > l_{t-1} + a_{t-1}`, let :math:`a_t = r - l_{t-1}` and :math:`R_t := R_t \times \frac{a_{t-1}}{a-t}`,
        3. Otherwise, nothing to do, the current reward is still correctly in :math:`[l_{t-1}, l_{t-1} + a_{t-1}]`, so simply keep :math:`l_t = l_{t-1}` and :math:`a_t = a_{t-1}`.
        """
        # print(" - At time t = {}, got a reward = {:.3g} from arm {} ...".format(self.t, reward, arm))  # DEBUG
        self.t += 1

        # check if the reward is still in [lower, lower + amplitude]
        if reward < self.lower or reward > self.lower + self.amplitude:
            self._i += 1  # just count how many change in range we seen
            old_l, old_a = self.lower, self.amplitude
            if reward < self.lower:
                print("Info: {} saw a reward {:.3g} below its lower value {:.3g}, now lower = {:.3g}... ({}th change)".format(self, reward, self.lower, reward, self._i))  # DEBUG
                self.lower = self.policy.lower = reward
                # Now we maybe have to rescale all the previous history!
                # 1st case: At+1=At, so just a small linear shift is enough
                # FIXME bring this back! but count it correctly for the regret
                self.rewards += (old_l - self.lower) / self.amplitude
            else:
                print("Info: {} saw a reward {:.3g} above its higher value {:.3g}, now amplitude = {:.3g}... ({}th change)".format(self, reward, self.lower + self.amplitude, reward - self.lower, self._i))  # DEBUG
                self.amplitude = self.policy.amplitude = reward - self.lower
                # Now we maybe have to rescale all the previous history!
                # 2nd case: Lt+1=Lt, so just a small multiplicative shift is enough
                # the shift is < 1, OK
                # FIXME bring this back! but count it correctly for the regret
                self.rewards *= old_a / self.amplitude

        self.policy.getReward(arm, reward)

    # --- Sub methods

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def index(self):
        r""" Get attribute ``index`` from the underlying policy."""
        return self.policy.index

    def choice(self):
        r""" Pass the call to ``choice`` of the underlying policy."""
        return self.policy.choice()

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
