# -*- coding: utf-8 -*-
r""" The Monitored-UCB generic policy for non-stationary bandits.

- Reference: [["Nearly Optimal Adaptive Procedure for Piecewise-Stationary Bandit: a Change-Point Detection Approach". Yang Cao, Zheng Wen, Branislav Kveton, Yao Xie. arXiv preprint arXiv:1802.03692, 2018]](https://arxiv.org/pdf/1802.03692)
- It runs on top of a simple policy, e.g., :class:`UCB`, and :class:`Monitored_IndexPolicy` is a wrapper:

    >>> policy = Monitored_IndexPolicy(nbArms, UCB)
    >>> # use policy as usual, with policy.startGame(), r = policy.choice(), policy.getReward(arm, r)

- It uses an additional :math:`\mathcal{O}(K w)` memory for a window of size :math:`w`.

.. warning:: It can only work on basic index policy based on empirical averages (and an exploration bias), like :class:`UCB`, and cannot work on any Bayesian policy (for which we would have to remember all previous observations in order to reset the history with a small history)!
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np

try:
    from .with_proba import with_proba
    from .BaseWrapperPolicy import BaseWrapperPolicy
except ImportError:
    from with_proba import with_proba
    from BaseWrapperPolicy import BaseWrapperPolicy


#: Default value for the parameter :math:`\delta`, the lower-bound for :math:`\delta_k^{(i)}` the amplitude of change of arm k at break-point.
#: Default is ``0.05``.
DELTA = 0.1

#: Should we reset one arm empirical average or all? For M-UCB it is ``False`` by default.
PER_ARM_RESTART = False

#: Should we fully restart the algorithm or simply reset one arm empirical average? For M-UCB it is ``True`` by default.
FULL_RESTART_WHEN_REFRESH = True


#: Default value of the window-size. Give ``None`` to use the default value computed from a knowledge of the horizon and number of break-points.
WINDOW_SIZE = 50
WINDOW_SIZE = None

#: For any algorithm with uniform exploration and a formula to tune it, :math:`\alpha` is usually too large and leads to larger regret. Multiplying it by a 0.1 or 0.2 helps, a lot!
GAMMA_SCALE_FACTOR = 1
# GAMMA_SCALE_FACTOR = 0.1


# --- The very generic class

class Monitored_IndexPolicy(BaseWrapperPolicy):
    r""" The Monitored-UCB generic policy for non-stationary bandits, from [["Nearly Optimal Adaptive Procedure for Piecewise-Stationary Bandit: a Change-Point Detection Approach". Yang Cao, Zheng Wen, Branislav Kveton, Yao Xie. arXiv preprint arXiv:1802.03692, 2018]](https://arxiv.org/pdf/1802.03692)

    - For a window size ``w``, it uses only :math:`\mathcal{O}(K w)` memory.
    """
    def __init__(self, nbArms,
            full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
            per_arm_restart=PER_ARM_RESTART,
            horizon=None, delta=DELTA, max_nb_random_events=None,
            w=WINDOW_SIZE, b=None, gamma=None,
            *args, **kwargs
        ):
        super(Monitored_IndexPolicy, self).__init__(nbArms, *args, **kwargs)
        if max_nb_random_events is None or max_nb_random_events <= 1:
            max_nb_random_events = 1

        # New parameters
        if w is None or w == 'auto':
            # XXX Estimate w from Remark 1
            w = (4/delta**2) * (np.sqrt(np.log(2 * nbArms * horizon**2)) + np.sqrt(np.log(2 * horizon)))**2
            w = int(np.ceil(w))
            if w % 2 != 0:
                w = 2*(1 + w//2)
            if w >= horizon / (1 + max_nb_random_events):
                print("Warning: the formula for w in the paper gave w = {}, that's crazy large, we use instead {}".format(w, 50 * nbArms * max_nb_random_events))
                w = 50 * nbArms * max_nb_random_events
        assert w > 0, "Error: for Monitored_UCB policy the parameter w should be > 0 but it was given as {}.".format(w)  # DEBUG
        self.window_size = w  #: Parameter :math:`w` for the M-UCB algorithm.

        if b is None or b == 'auto':
            # XXX compute b from the formula from Theorem 6.1
            b = np.sqrt((w/2) * np.log(2 * nbArms * horizon**2))
        assert b > 0, "Error: for Monitored_UCB policy the parameter b should be > 0 but it was given as {}.".format(b)  # DEBUG
        self.threshold_b = b  #: Parameter :math:`b` for the M-UCB algorithm.

        if gamma is None or gamma == 'auto':
            M = max_nb_random_events
            assert M >= 1, "Error: for Monitored_UCB policy the parameter M should be >= 1 but it was given as {}.".format(M)  # DEBUG
            # XXX compute gamma from the formula from Theorem 6.1
            # gamma = np.sqrt(M * nbArms * min(w/2, np.ceil(b / delta) +  3 * np.sqrt(w)) / (2 * horizon))
            # XXX compute gamma from the tuning used for their experiment (§6.1)
            gamma = np.sqrt(M * nbArms * (2 * b +  3 * np.sqrt(w)) / (2 * horizon))
        if gamma <= 0:
            print("\n\nWarning: the formula for gamma in the paper gave gamma = {}, that's absurd, we use instead {}".format(gamma, 1.0 / (1 + 50 * nbArms * max_nb_random_events)))
            gamma = 1.0 / (1 + 50 * nbArms * max_nb_random_events)
        elif gamma > 1:
            print("\n\nWarning: the formula for gamma in the paper gave gamma = {}, that's absurd, we use instead {}".format(gamma, 0.01 * nbArms))
            gamma = 0.01 * nbArms
        # FIXME just to do as the two other approaches GLR-UCB and CUSUM-UCB
        gamma *= GAMMA_SCALE_FACTOR
        assert 0 <= gamma <= 1, "Error: for Monitored_UCB policy the parameter gamma should be 0 <= gamma <= 1, but it was given as {}.".format(gamma)  # DEBUG
        gamma = max(0, min(1, gamma))  # clip gamma to (0, 1) it's a probability!
        self.gamma = gamma  #: What they call :math:`\gamma` in their paper: the share of uniform exploration.

        self._full_restart_when_refresh = full_restart_when_refresh  # Should we fully restart the algorithm or simply reset one arm empirical average ?
        self._per_arm_restart = per_arm_restart  # Should we reset one arm empirical average or all?
        self.last_update_time_tau = 0  #: Keep in memory the last time a change was detected, ie, the variable :math:`\tau` in the algorithm.

        # Internal memory
        self.last_w_rewards = [[] for _ in range(self.nbArms)]  #: Keep in memory all the rewards obtained since the last restart on that arm.
        self.last_pulls = np.zeros(nbArms, dtype=int)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
        self.last_restart_times = np.zeros(nbArms, dtype=int)  #: Keep in memory the times of last restarts (for each arm).

    def __str__(self):
        args = "{}{}".format("$w={:g}$".format(self.window_size) if self.window_size != WINDOW_SIZE else "", "" if self._per_arm_restart else ", Global")
        args = "({})".format(args) if args else ""
        return r"M-{}{}".format(self._policy.__name__, args)

    def choice(self):
        r""" Essentially play uniformly at random with probability :math:`\gamma`, otherwise, pass the call to ``choice`` of the underlying policy (eg. UCB).

        .. warning:: Actually, it's more complicated:

        - If :math:`t` is the current time and :math:`\tau` is the latest restarting time, then uniform exploration is done if:

        .. math::

            A &:= (t - \tau) \mod \lceil \frac{K}{\gamma} \rceil,\\
            A &\leq K \implies A_t = A.
        """
        if self.gamma > 0:
            A = (self.t - self.last_update_time_tau) % int(np.ceil(self.nbArms / self.gamma))
            if A < self.nbArms:
                return int(A)
        # FIXED no in this algorithm they do not use a uniform chance of random exploration!
        # if with_proba(self.gamma):
        #     return np.random.randint(0, self.nbArms - 1)
        return self.policy.choice()

    def choiceWithRank(self, rank=1):
        r""" Essentially play uniformly at random with probability :math:`\gamma`, otherwise, pass the call to ``choiceWithRank`` of the underlying policy (eg. UCB). """
        if self.gamma > 0:
            A = (self.t - self.last_update_time_tau) % int(np.ceil(self.nbArms / self.gamma))
            if A < self.nbArms:
                return int(A)
        return self.policy.choiceWithRank(rank=1)

    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the whole empirical average if the change detection algorithm says so.
        """
        super(Monitored_IndexPolicy, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_pulls[arm] += 1
        # DONE use only :math:`\mathcal{O}(K w)` memory.
        if len(self.last_w_rewards[arm]) >= self.window_size:
            self.last_w_rewards[arm].pop(0)
        # Store it in place for the empirical average of that arm
        self.last_w_rewards[arm].append(reward)

        if self.detect_change(arm):
            # print("For a player {} a change was detected at time {} for arm {}, after {} pulls of that arm (giving mean reward = {:.3g}). Last restart on that arm was at tau = {}".format(self, self.t, arm, self.last_pulls[arm], np.sum(self.last_w_rewards[arm]) / self.last_pulls[arm], self.last_restart_times[arm]))  # DEBUG
            self.last_update_time_tau = self.t

            if not self._per_arm_restart:
                # or reset current memory for ALL THE arms
                for other_arm in range(self.nbArms):
                    self.last_restart_times[other_arm] = self.t
                    self.last_pulls[other_arm] = 0
                    self.last_w_rewards[other_arm] = []
            # reset current memory for THIS arm
            self.last_restart_times[arm] = self.t
            self.last_pulls[arm] = 1
            self.last_w_rewards[arm] = [reward]

            # Fully restart the algorithm ?!
            if self._full_restart_when_refresh:
                self.startGame(createNewPolicy=True)
            # Or simply reset one of the empirical averages?
            else:
                if not self._per_arm_restart:
                # or reset current memory for ALL THE arms
                    for other_arm in range(self.nbArms):
                        self.policy.rewards[other_arm] = 0
                        self.policy.pulls[other_arm] = 0
                # reset current memory for THIS arm
                self.policy.rewards[arm] = np.sum(self.last_w_rewards[arm])
                self.policy.pulls[arm] = len(self.last_w_rewards[arm])

        # we update the total number of samples available to the underlying policy
        # self.policy.t = np.sum(self.last_pulls)  # XXX SO NOT SURE HERE

    def detect_change(self, arm):
        r""" A change is detected for the current arm if the following test is true:

        .. math:: |\sum_{i=w/2+1}^{w} Y_i - \sum_{i=1}^{w/2} Y_i | > b ?

        - where :math:`Y_i` is the i-th data in the latest w data from this arm (ie, :math:`X_k(t)` for :math:`t = n_k - w + 1` to :math:`t = n_k` current number of samples from arm k).
        - where :attr:`threshold_b` is the threshold b of the test, and :attr:`window_size` is the window-size w.

        .. warning:: FIXED only the last :math:`w` data are stored, using lists that got their first element ``pop()``ed out (deleted). See https://github.com/SMPyBandits/SMPyBandits/issues/174
        """
        data_y = self.last_w_rewards[arm]
        # don't try to detect change if there is not enough data!
        if len(data_y) < self.window_size:
            return False
        # last_w_data_y = data_y[-self.window_size:]  # WARNING revert to this if data_y can be larger
        last_w_data_y = data_y
        sum_first_half = np.sum(last_w_data_y[:self.window_size//2])
        sum_second_half = np.sum(last_w_data_y[self.window_size//2:])
        has_detected = abs(sum_first_half - sum_second_half) > self.threshold_b
        return has_detected, None
