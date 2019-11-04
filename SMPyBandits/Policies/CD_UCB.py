# -*- coding: utf-8 -*-
r""" The CD-UCB generic policy policies for non-stationary bandits.

- Reference: [["A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem". F. Liu, J. Lee and N. Shroff. arXiv preprint arXiv:1711.03539, 2017]](https://arxiv.org/pdf/1711.03539)
- It runs on top of a simple policy, e.g., :class:`UCB`, and :class:`UCBLCB_IndexPolicy` is a wrapper:

    >>> policy = UCBLCB_IndexPolicy(nbArms, UCB)
    >>> # use policy as usual, with policy.startGame(), r = policy.choice(), policy.getReward(arm, r)

- It uses an additional :math:`\mathcal{O}(\tau_\max)` memory for a game of maximum stationary length :math:`\tau_\max`.

.. warning:: It can only work on basic index policy based on empirical averages (and an exploration bias), like :class:`UCB`, and cannot work on any Bayesian policy (for which we would have to remember all previous observations in order to reset the history with a small history)!
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
from math import log, sqrt, isinf

try:
    from .with_proba import with_proba
    from .BaseWrapperPolicy import BaseWrapperPolicy
    from .UCB import UCB as DefaultPolicy
except ImportError:
    from with_proba import with_proba
    from BaseWrapperPolicy import BaseWrapperPolicy
    from UCB import UCB as DefaultPolicy


#: Whether to be verbose when doing the change detection algorithm.
VERBOSE = False

#: Default probability of random exploration :math:`\alpha`.
PROBA_RANDOM_EXPLORATION = 0.1

#: Should we reset one arm empirical average or all? Default is ``True``, it's usually more efficient!
PER_ARM_RESTART = False
PER_ARM_RESTART = True

#: Should we fully restart the algorithm or simply reset one arm empirical average? Default is ``False``, it's usually more efficient!
FULL_RESTART_WHEN_REFRESH = True
FULL_RESTART_WHEN_REFRESH = False

#: Precision of the test. For CUSUM/PHT, :math:`\varepsilon` is the drift correction threshold (see algorithm).
EPSILON = 0.05

#: Default value of :math:`\lambda`.
LAMBDA = 1

#: Hypothesis on the speed of changes: between two change points, there is at least :math:`M * K` time steps, where K is the number of arms, and M is this constant.
MIN_NUMBER_OF_OBSERVATION_BETWEEN_CHANGE_POINT = 50

#: XXX Be lazy and try to detect changes only X steps, where X is small like 10 for instance.
#: It is a simple but efficient way to speed up CD tests, see https://github.com/SMPyBandits/SMPyBandits/issues/173
#: Default value is 0, to not use this feature, and 10 should speed up the test by x10.
LAZY_DETECT_CHANGE_ONLY_X_STEPS = 1
LAZY_DETECT_CHANGE_ONLY_X_STEPS = 10


# --- The very generic class

class CD_IndexPolicy(BaseWrapperPolicy):
    r""" The CD-UCB generic policy for non-stationary bandits, from [["A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem". F. Liu, J. Lee and N. Shroff. arXiv preprint arXiv:1711.03539, 2017]](https://arxiv.org/pdf/1711.03539).
    """
    def __init__(self, nbArms,
            full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
            per_arm_restart=PER_ARM_RESTART,
            epsilon=EPSILON,
            proba_random_exploration=None,
            lazy_detect_change_only_x_steps=LAZY_DETECT_CHANGE_ONLY_X_STEPS,
            *args, **kwargs
        ):
        super(CD_IndexPolicy, self).__init__(nbArms, *args, **kwargs)
        # New parameters
        self.epsilon = epsilon  #: Parameter :math:`\varepsilon` for the test.
        self.lazy_detect_change_only_x_steps = lazy_detect_change_only_x_steps  #: Be lazy and try to detect changes only X steps, where X is small like 10 for instance.
        if proba_random_exploration is not None:
            alpha = max(0, min(1, proba_random_exploration))  # crop to [0, 1]
            self.proba_random_exploration = alpha  #: What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time.
        self._full_restart_when_refresh = full_restart_when_refresh  # Should we fully restart the algorithm or simply reset one arm empirical average ?
        self._per_arm_restart = per_arm_restart  # Should we reset one arm empirical average or all?
        # Internal memory
        self.all_rewards = [[] for _ in range(self.nbArms)]  #: Keep in memory all the rewards obtained since the last restart on that arm.
        self.last_pulls = np.zeros(nbArms, dtype=int)  #: Keep in memory the number times since last restart. Start with -1 (never seen)
        self.last_restart_times = np.zeros(nbArms, dtype=int)  #: Keep in memory the times of last restarts (for each arm).
        self.number_of_restart = 0  #: Keep in memory the number of restarts.

    def __str__(self):
        return r"CD-{}($\varepsilon={:.3g}$, $\gamma={:.3g}$, {}{})".format(self._policy.__name__, self.epsilon, self.proba_random_exploration, "" if self._per_arm_restart else "Global", ", lazy detect {}".format(self.lazy_detect_change_only_x_steps) if self.lazy_detect_change_only_x_steps != LAZY_DETECT_CHANGE_ONLY_X_STEPS else "")

    def choice(self):
        r""" With a probability :math:`\alpha`, play uniformly at random, otherwise, pass the call to :meth:`choice` of the underlying policy."""
        if with_proba(self.proba_random_exploration):
            return np.random.randint(0, self.nbArms - 1)
        return self.policy.choice()

    def choiceWithRank(self, rank=1):
        r""" With a probability :math:`\alpha`, play uniformly at random, otherwise, pass the call to :meth:`choiceWithRank` of the underlying policy."""
        if with_proba(self.proba_random_exploration):
            return np.random.randint(0, self.nbArms - 1)
        return self.policy.choiceWithRank(rank=1)

    def getReward(self, arm, reward):
        r""" Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the whole empirical average if the change detection algorithm says so, with method :meth:`detect_change`, for this arm at this current time step.

        .. warning:: This is computationally costly, so an easy way to speed up this step is to use :attr:`lazy_detect_change_only_x_steps` :math:`= \mathrm{Step_t}` for a small value (e.g., 10), so not test for all :math:`t\in\mathbb{N}^*` but only :math:`s\in\mathbb{N}^*, s % \mathrm{Step_t} = 0` (e.g., one out of every 10 steps).

        .. warning:: If the :math:`detect_change` method also returns an estimate of the position of the change-point, :math:`\hat{tau}`, then it is used to reset the memory of the changing arm and keep the observations from :math:`\hat{tau}+1`.
        """
        super(CD_IndexPolicy, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_pulls[arm] += 1
        # Store it in place for the empirical average of that arm
        self.all_rewards[arm].append(reward)

        should_you_try_to_detect = (self.last_pulls[arm] % self.lazy_detect_change_only_x_steps) == 0
        if not should_you_try_to_detect: return
        has_detected_and_maybe_position = self.detect_change(arm)
        if isinstance(has_detected_and_maybe_position, bool):
            has_detected, position = has_detected_and_maybe_position, None
        else:
            has_detected, position = has_detected_and_maybe_position
        if not has_detected: return
        self.number_of_restart += 1
        if position is None:
            # print("For a player {} a change was detected at time {} for arm {}, after {} pulls of that arm (giving mean reward = {:.3g}). Last restart on that arm was at tau = {}".format(self, self.t, arm, self.last_pulls[arm], np.sum(self.all_rewards[arm]) / self.last_pulls[arm], self.last_restart_times[arm]))  # DEBUG

            if not self._per_arm_restart:
                # or reset current memory for ALL THE arms
                for other_arm in range(self.nbArms):
                    self.last_restart_times[other_arm] = self.t
                    self.last_pulls[other_arm] = 0
                    self.all_rewards[other_arm] = []
            # reset current memory for THIS arm
            self.last_restart_times[arm] = self.t
            self.last_pulls[arm] = 1
            self.all_rewards[arm] = [reward]

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
                self.policy.rewards[arm] = np.sum(self.all_rewards[arm])
                self.policy.pulls[arm] = len(self.all_rewards[arm])
        # XXX second case, for policies implementing the localization trick
        else:
            # print("For a player {} a change was detected at time {} for arm {}, after {} pulls of that arm (giving mean reward = {:.3g}). Last restart on that arm was at tau = {}".format(self, self.t, arm, self.last_pulls[arm], np.sum(self.all_rewards[arm]) / self.last_pulls[arm], self.last_restart_times[arm]))  # DEBUG

            if not self._per_arm_restart:
                # or reset current memory for ALL THE arms
                for other_arm in range(self.nbArms):
                    self.last_restart_times[other_arm] = self.t
                    self.last_pulls[other_arm] = 0
                    self.all_rewards[other_arm] = []
            # reset current memory for THIS arm
            self.all_rewards[arm] = self.all_rewards[arm][position:] + [reward]
            self.last_pulls[arm] = len(self.all_rewards[arm])
            self.last_restart_times[arm] = self.t - len(self.all_rewards[arm]) + 1

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
                self.policy.rewards[arm] = np.sum(self.all_rewards[arm])
                self.policy.pulls[arm] = len(self.all_rewards[arm])

        # we update the total number of samples available to the underlying policy
        # self.policy.t = sum(self.last_pulls)  # DONE for the generic CD UCB policy, nothing to do here to change self.policy.t
        # XXX See CUSUM_UCB which forces self.policy.t = sum(last_pulls)
        # XXX See GLR_UCB which forces self.policy.t_for_each_arm = t - self.last_restart_times (different t for each arm)!

    def detect_change(self, arm, verbose=VERBOSE):
        """ Try to detect a change in the current arm.

        .. warning:: This is not implemented for the generic CD algorithm, it has to be implement by a child of the class :class:`CD_IndexPolicy`.
        """
        raise NotImplementedError


# --- Different change detection algorithms

class SlidingWindowRestart_IndexPolicy(CD_IndexPolicy):
    r""" A more generic implementation is the :class:`Policies.SlidingWindowRestart` class.

    .. warning:: I have no idea if what I wrote is correct or not!
    """

    def detect_change(self, arm, verbose=VERBOSE):
        """ Try to detect a change in the current arm.

        .. warning:: This one is simply using a sliding-window of fixed size = 100. A more generic implementation is the :class:`Policies.SlidingWindowRestart` class.
        """
        tau = self.M * self.nbArms
        if self.last_pulls[arm] >= tau and self.pulls[arm] >= tau:
            # Compute the empirical average for that arm
            empirical_average = self.rewards[arm] / self.pulls[arm]
            # And the small empirical average for that arm
            small_empirical_average = np.mean(self.last_rewards[arm])
            if np.abs(empirical_average - small_empirical_average) >= self.epsilon:
                return True, None
        return False, None


# --- UCB-CDP based on LCB/UCB Mukherjee & Maillard's paper

#: XXX Be lazy and try to detect changes for :math:`s` taking steps of size ``steps_s``. Default is to have ``steps_s=1``, but only using ``steps_s=2`` should already speed up by 2.
#: It is a simple but efficient way to speed up GLR tests, see https://github.com/SMPyBandits/SMPyBandits/issues/173
#: Default value is 1, to not use this feature, and 10 should speed up the test by x10.
LAZY_TRY_VALUE_S_ONLY_X_STEPS = 1
LAZY_TRY_VALUE_S_ONLY_X_STEPS = 10


#: Default value of ``use_localization`` for policies. All the experiments I tried showed that the localization always helps improving learning, so the default value is set to True.
USE_LOCALIZATION = False
USE_LOCALIZATION = True


class UCBLCB_IndexPolicy(CD_IndexPolicy):
    r""" The UCBLCB-UCB generic policy for non-stationary bandits, from [[Improved Changepoint Detection for Piecewise i.i.d Bandits, by S. Mukherjee  & O.-A. Maillard, preprint 2018](https://subhojyoti.github.io/pdf/aistats_2019.pdf)].

    .. warning:: This is still experimental! See https://github.com/SMPyBandits/SMPyBandits/issues/177
    """
    def __init__(self, nbArms,
            delta=None, delta0=1.0,
            lazy_try_value_s_only_x_steps=LAZY_TRY_VALUE_S_ONLY_X_STEPS,
            use_localization=USE_LOCALIZATION,
            *args, **kwargs
        ):
        super(UCBLCB_IndexPolicy, self).__init__(nbArms, per_arm_restart=False, *args, **kwargs)
        # New parameters
        self.proba_random_exploration = 0  #: What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time.
        self.lazy_try_value_s_only_x_steps = lazy_try_value_s_only_x_steps  #: Be lazy and try to detect changes for :math:`s` taking steps of size ``steps_s``.
        self.use_localization = use_localization  #: experiment to use localization of the break-point, ie, restart memory of arm by keeping observations s+1...n instead of just the last one
        self._delta = delta
        self._delta0 = delta0

    def __str__(self):
        args = "{}{}{}".format(
            "lazy detect {}, ".format(self.lazy_detect_change_only_x_steps) if self.lazy_detect_change_only_x_steps != LAZY_DETECT_CHANGE_ONLY_X_STEPS else "",
            "lazy s {}, ".format(self.lazy_try_value_s_only_x_steps) if self.lazy_try_value_s_only_x_steps != LAZY_TRY_VALUE_S_ONLY_X_STEPS else "",
            r"$\delta_0={:.3g}$, ".format(self._delta0) if self._delta0 != 1 else "",
        )
        if args.endswith(', '): args = args[:-2]
        args = "({})".format(args) if args else ""
        return r"{}-CDP{}".format(self._policy.__name__, args)

    def delta(self, t):
        r""" Use :math:`\delta = \delta_0` if it was given as an argument to the policy, or :math:`\frac{\delta_0}{t}` as the confidence level of UCB/LCB test (default is :math:`\delta_0=1`).

        .. warning:: It is unclear (in the article) whether :math:`t` is the time since the last restart or the total time?
        """
        if self._delta is not None:
            return self._delta
        else:
            return self._delta0 / t

    def detect_change(self, arm, verbose=VERBOSE):
        r""" Detect a change in the current arm, using the two-sided UCB-LCB algorithm [Mukherjee & Maillard, 2018].

        - Let :math:`\hat{\mu}_{i,t:t'}` the empirical mean of rewards obtained for arm i from time :math:`t` to :math:`t'`, and :math:`N_{i,t:t'}` the number of samples.
        - Let :math:`S_{i,t:t'} = \sqrt{\frac{\log(4 t^2 / \delta)}{2 N_{i,t:t'}}}` the length of the confidence interval.

        - When we have data starting at :math:`t_0=0` (since last restart) and up-to current time :math:`t`, for each *arm* i,
            - For each intermediate time steps :math:`t' \in [t_0, t)`,
                - Compute :math:`LCB_{\text{before}} = \hat{\mu}_{i,t_0:t'} - S_{i,t_0:t'}`,
                - Compute :math:`UCB_{\text{before}} = \hat{\mu}_{i,t_0:t'} + S_{i,t_0:t'}`,
                - Compute :math:`LCB_{\text{after}} = \hat{\mu}_{i,t'+1:t} - S_{i,t'+1:t}`,
                - Compute :math:`UCB_{\text{after}} = \hat{\mu}_{i,t'+1:t} + S_{i,t'+1:t}`,
                - If :math:`UCB_{\text{before}} < LCB_{\text{after}}` or :math:`UCB_{\text{after}} < LCB_{\text{before}}`, then restart.
        """
        for armId in range(self.nbArms):
            data_y = self.all_rewards[armId]
            t0 = 0
            t = len(data_y)-1
            if t <= 2:
                continue
            mean_all = np.mean(data_y[t0 : t+1])
            mean_before = 0.0
            mean_after = mean_all
            for s in range(t0, t):
                y = data_y[s]
                mean_before = (s * mean_before + y) / (s + 1)
                mean_after = ((t + 1 - s + t0) * mean_after - y) / (t - s + t0)
                if s % self.lazy_try_value_s_only_x_steps != 0:
                    continue

                ucb_lcb_cst = sqrt(log(4 * t / self.delta(t)) / 2.0)
                S_before = ucb_lcb_cst / sqrt(s + 1)
                S_after  = ucb_lcb_cst / sqrt(t - s + t0)
                ucb_after  = mean_after  + S_after
                lcb_before = mean_before - S_before

                if ucb_after < lcb_before:
                    if verbose: print("  - For arm = {}, t0 = {}, s = {}, t = {}, the mean before mu(t0,s) = {:.3g} and the mean after mu(s+1,t) = {:.3g} and the S_before = {:.3g} and S_after = {:.3g}, so UCB_after = {:.3g} < LCB_before = {:.3g}...".format(armId, t0, s, t, mean_before, mean_after, S_before, S_after, ucb_after, lcb_before))
                    return True, t0 + s if self.use_localization else None

                ucb_before = mean_before + S_before
                lcb_after  = mean_after  - S_after

                if ucb_before < lcb_after:
                    if verbose: print("  - For arm = {}, t0 = {}, s = {}, t = {}, the mean before mu(t0,s) = {:.3g} and the mean after mu(s+1,t) = {:.3g} and the S_before = {:.3g} and S_after = {:.3g}, so UCB_before = {:.3g} < LCB_after = {:.3g}...".format(armId, t0, s, t, mean_before, mean_after, S_before, S_after, ucb_before, lcb_after))
                    return True, t0 + s if self.use_localization else None
            return False, None
