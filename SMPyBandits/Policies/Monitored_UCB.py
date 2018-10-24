# -*- coding: utf-8 -*-
r""" The Monitored-UCB generic policy for non-stationary bandits.

- Reference: [["Nearly Optimal Adaptive Procedure for Piecewise-Stationary Bandit: a Change-Point Detection Approach". Yang Cao, Zheng Wen, Branislav Kveton, Yao Xie. arXiv preprint arXiv:1802.03692, 2018]](https://arxiv.org/pdf/1802.03692)
- It runs on top of a simple policy, e.g., :class:`Policy.UCB.UCB`, and :func:`Monitored_IndexPolicy` is a wrapper:

    >>> policy = Monitored_IndexPolicy(nbArms, UCB)
    >>> # use policy as usual, with policy.startGame(), r = policy.choice(), policy.getReward(arm, r)

- It uses an additional :math:`\mathcal{O}(\tau_\max)` memory for a game of maximum stationary length :math:`\tau_\max`.

.. warning:: This implementation is still experimental!
.. warning:: It can only work on basic index policy based on empirical averages (and an exploration bias), like :class:`Policy.UCB.UCB`, and cannot work on any Bayesian policy (for which we would have to remember all previous observations in order to reset the history with a small history)!
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


import numpy as np

try:
    from .with_proba import with_proba
    from .BaseWrapperPolicy import BaseWrapperPolicy
    from .UCB import UCB as DefaultPolicy, UCB
except ImportError:
    from with_proba import with_proba
    from BaseWrapperPolicy import BaseWrapperPolicy
    from UCB import UCB as DefaultPolicy, UCB


#: Default value for the parameter :math:`\delta`, the lower-bound for :math:`\delta_k^{(i)}` the amplitude of change of arm k at break-point 1.
#: The default abruptly-changing non-stationary problem draws news means in :math:`[0,1]` so :math:`\delta=0` is the only possible (worst-case) lower-bound on amplitude of changes.
#: I force ``0.1`` because I can force the minimum gap when calling :func:`Arms.randomMeans` to be ``0.1``.
DELTA = 0.01

#: Should we fully restart the algorithm or simply reset one arm empirical average ?
FULL_RESTART_WHEN_REFRESH = False


# --- The very generic class

class Monitored_IndexPolicy(BaseWrapperPolicy):
    r""" The Monitored-UCB generic policy for non-stationary bandits, from [["Nearly Optimal Adaptive Procedure for Piecewise-Stationary Bandit: a Change-Point Detection Approach". Yang Cao, Zheng Wen, Branislav Kveton, Yao Xie. arXiv preprint arXiv:1802.03692, 2018]](https://arxiv.org/pdf/1802.03692)
    """
    def __init__(self, nbArms,
            full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
            horizon=None, delta=DELTA, max_nb_random_events=None,
            w=None, b=None, gamma=None,
            policy=DefaultPolicy,
            lower=0., amplitude=1., *args, **kwargs
        ):
        super(Monitored_IndexPolicy, self).__init__(nbArms, policy=policy, lower=lower, amplitude=amplitude, *args, **kwargs)
        if max_nb_random_events is None or max_nb_random_events <= 1:
            max_nb_random_events = 1

        # New parameters
        if w is None or w == 'auto':
            # XXX Estimate w from Remark 1
            w = (4/delta**2) * (np.sqrt(np.log(2 * nbArms * horizon**2)) + np.sqrt(2 * horizon))**2
            w = int(w)
            if w % 2 != 0:
                w = 2*(1 + w//2)
        assert w > 0, "Error: for Monitored_UCB policy the parameter w should be > 0 but it was given as {}.".format(w)  # DEBUG
        self.w = w  #: Parameter :math:`w` for the M-UCB algorithm.

        if b is None or b == 'auto':
            # XXX compute b from the formula from Theorem 6.1
            b = np.sqrt((w/2) * np.log(2 * nbArms * horizon**2))
        assert b > 0, "Error: for Monitored_UCB policy the parameter b should be > 0 but it was given as {}.".format(b)  # DEBUG
        self.b = b  #: Parameter :math:`b` for the M-UCB algorithm.

        if gamma is None or gamma == 'auto':
            M = max_nb_random_events
            assert M >= 1, "Error: for Monitored_UCB policy the parameter M should be >= 1 but it was given as {}.".format(M)  # DEBUG
            # XXX compute gamma from the formula from Theorem 6.1
            gamma = np.sqrt((M-1) * nbArms * min(w/2, np.ceil(b / delta) +  3 * np.sqrt(w)) / (2 * horizon))
        if gamma > 1:
            gamma = 0.05 * nbArms
        assert 0 <= gamma <= 1, "Error: for Monitored_UCB policy the parameter gamma should be 0 <= gamma <= 1, but it was given as {}.".format(gamma)  # DEBUG
        gamma = max(0, min(1, gamma))  # clip gamma to (0, 1) it's a probability!
        self.proba_random_exploration = gamma  #: What they call :math:`\gamma` in their paper: the probability of uniform exploration at each time.

        self._full_restart_when_refresh = full_restart_when_refresh  # Should we fully restart the algorithm or simply reset one arm empirical average ?

        # Internal memory
        self.all_rewards = [[] for _ in range(self.nbArms)]  #: Keep in memory all the rewards obtained since the last restart on that arm.
        self.last_pulls = np.full(nbArms, -1)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)

    def __str__(self):
        return r"Monitored({}, $w={:.3g}$, $b={:.3g}$, $\gamma={:.3g}$)".format(self._policy.__name__, self.w, self.b, self.proba_random_exploration)

    def choice(self):
        r""" With a probability :math:`\alpha`, play uniformly at random, otherwise, pass the call to ``choice`` of the underlying policy."""
        if with_proba(self.proba_random_exploration):
            return np.random.randint(0, self.nbArms - 1)
        return self.policy.choice()

    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the whole empirical average if the change detection algorithm says so.
        """
        super(Monitored_IndexPolicy, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_pulls[arm] += 1
        # we update the total number of samples available to the underlying policy
        self.policy.t = sum(self.last_pulls)
        # Store it in place for the empirical average of that arm
        self.all_rewards[arm].append(reward)
        if self.detect_change(arm):
            # Fully restart the algorithm ?!
            if self._full_restart_when_refresh:
                self.startGame(createNewPolicy=False)
            # Or simply reset one of the empirical averages?
            else:
                self.rewards[arm] = np.sum(self.all_rewards[arm])
                self.pulls[arm] = len(self.all_rewards[arm])
            # reset current memory for THIS arm
            self.last_pulls[arm] = 0
            self.all_rewards[arm] = []

    def detect_change(self, arm):
        r""" A change i detected in the current arm if the following test is true:

        .. math:: |\sum_{i=w/2+1}^{w} Y_i - \sum_{i=1}^{w/2} Y_i | > b ?

        - where :math:`Y_i` is the i-th data in the latest w data from this arm.
        - where b is the threshold of the test, and w is the window-size.
        """
        data_y = self.all_rewards[arm]
        sum_first_half = np.sum(data_y[:self.w])
        sum_second_half = np.sum(data_y[self.w:])
        return abs(sum_first_half - sum_second_half) > self.b


