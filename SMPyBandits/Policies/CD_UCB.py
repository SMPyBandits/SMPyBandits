# -*- coding: utf-8 -*-
r""" The CD-UCB generic policy and CUSUM-UCB, PHT-UCB, GLR-UCB policies for non-stationary bandits.

- Reference: [["A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem". F. Liu, J. Lee and N. Shroff. arXiv preprint arXiv:1711.03539, 2017]](https://arxiv.org/pdf/1711.03539)
- It runs on top of a simple policy, e.g., :class:`UCB`, and :class:`CUSUM_IndexPolicy` is a wrapper:

    >>> policy = CUSUM_IndexPolicy(nbArms, UCB)
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
LAZY_DETECT_CHANGE_ONLY_X_STEPS = 4


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

    def __str__(self):
        return r"CD-{}($\varepsilon={:.3g}$, $\gamma={:.3g}$, {}{})".format(self._policy.__name__, self.epsilon, self.proba_random_exploration, "" if self._per_arm_restart else "Global", ", lazy detect {}".format(self.lazy_detect_change_only_x_steps) if self.lazy_detect_change_only_x_steps != LAZY_DETECT_CHANGE_ONLY_X_STEPS else "")

    def choice(self):
        r""" With a probability :math:`\alpha`, play uniformly at random, otherwise, pass the call to :meth:`choice` of the underlying policy."""
        if with_proba(self.proba_random_exploration):
            return np.random.randint(0, self.nbArms - 1)
        return self.policy.choice()

    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the whole empirical average if the change detection algorithm says so, with method :meth:`detect_change`, for this arm at this current time step.

        .. warning:: This is computationally costly, so an easy way to speed up this step is to use :attr:`lazy_detect_change_only_x_steps` :math:`= \mathrm{Step_t}` for a small value (e.g., 10), so not test for all :math:`t\in\mathbb{N}^*` but only :math:`s\in\mathbb{N}^*, s % \mathrm{Step_t} = 0` (e.g., one out of every 10 steps).
        """
        super(CD_IndexPolicy, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_pulls[arm] += 1
        # Store it in place for the empirical average of that arm
        self.all_rewards[arm].append(reward)

        should_you_try_to_detect = (self.last_pulls[arm] % self.lazy_detect_change_only_x_steps) == 0
        if should_you_try_to_detect and self.detect_change(arm):
            print("For a player {} a change was detected at time {} for arm {} after seeing reward = {}!".format(self, self.t, arm, reward))  # DEBUG

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

        # we update the total number of samples available to the underlying policy
        # self.policy.t = sum(self.last_pulls)  # XXX SO NOT SURE HERE

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
                return True
        return False


# --- Different change detection algorithms


ALPHA0_SCALE_FACTOR = 0.1  #: For any algorithm with uniform exploration and a formula to tune it, :math:`\alpha` is usually too large and leads to larger regret. Multiplying it by a 0.1 or 0.2 helps,a  lot!


from scipy.special import comb

def compute_h_alpha_from_input_parameters__CUSUM_complicated(horizon, max_nb_random_events, nbArms=None, epsilon=None, lmbda=None, M=None, scaleFactor=ALPHA0_SCALE_FACTOR):
    r""" Compute the values :math:`C_1^+, C_1^-, C_1, C_2, h` from the formulas in Theorem 2 and Corollary 2 in the paper."""
    T = int(max(1, horizon))
    UpsilonT = int(max(1, max_nb_random_events))
    K = int(max(1, nbArms))
    print("compute_h_alpha_from_input_parameters__CUSUM() with:\nT = {}, UpsilonT = {}, K = {}, epsilon = {}, lmbda = {}, M = {}".format(T, UpsilonT, K, epsilon, lmbda, M))  # DEBUG
    C2 = np.log(3) + 2 * np.exp(- 2 * epsilon**2 * M) / lmbda
    C1_minus = np.log(((4 * epsilon) / (1-epsilon)**2) * comb(M, int(np.floor(2 * epsilon * M))) * (2 * epsilon)**M + 1)
    C1_plus = np.log(((4 * epsilon) / (1+epsilon)**2) * comb(M, int(np.ceil(2 * epsilon * M))) * (2 * epsilon)**M + 1)
    C1 = min(C1_minus, C1_plus)
    if C1 == 0: C1 = 1  # FIXME This case of having C1=0 for CUSUM parameters should not happen...
    h = 1/C1 * np.log(T / UpsilonT)
    alpha = K * np.sqrt((C2 * UpsilonT)/(C1 * T) * np.log(T / UpsilonT))
    alpha *= scaleFactor  # XXX Just divide alpha to not have too large, for CUSUM-UCB.
    alpha = max(0, min(1, alpha))  # crop to [0, 1]
    print("Gave C2 = {}, C1- = {} and C1+ = {} so C1 = {}, and h = {} and alpha = {}".format(C2, C1_minus, C1_plus, C1, h, alpha))  # DEBUG
    return h, alpha

def compute_h_alpha_from_input_parameters__CUSUM(horizon, max_nb_random_events, nbArms=None, epsilon=None, lmbda=None, M=None, scaleFactor=ALPHA0_SCALE_FACTOR):
    r""" Compute the values :math:`h, \alpha` from the simplified formulas in Theorem 2 and Corollary 2 in the paper.

    .. math::

        h &= \log(\frac{T}{\Upsilon_T}),\\
        \alpha &= \mathrm{scaleFactor} \times \sqrt{\frac{\Upsilon_T}{T} \log(\frac{T}{\Upsilon_T})}.
    """
    T = int(max(1, horizon))
    UpsilonT = int(max(1, max_nb_random_events))
    ratio = T / UpsilonT
    h = np.log(ratio)
    alpha = np.sqrt(np.log(ratio) / ratio)
    alpha = max(0, min(1, alpha))  # crop to [0, 1]
    alpha *= scaleFactor  # XXX Just divide alpha to not have too large, for CUSUM-UCB.
    return h, alpha


class CUSUM_IndexPolicy(CD_IndexPolicy):
    r""" The CUSUM-UCB generic policy for non-stationary bandits, from [["A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem". F. Liu, J. Lee and N. Shroff. arXiv preprint arXiv:1711.03539, 2017]](https://arxiv.org/pdf/1711.03539).
    """
    def __init__(self, nbArms,
            horizon=None, max_nb_random_events=None,
            lmbda=LAMBDA,
            min_number_of_observation_between_change_point=MIN_NUMBER_OF_OBSERVATION_BETWEEN_CHANGE_POINT,
            *args, **kwargs
        ):
        super(CUSUM_IndexPolicy, self).__init__(nbArms, *args, **kwargs)
        # New parameters
        self.max_nb_random_events = max_nb_random_events
        self.M = min_number_of_observation_between_change_point  #: Parameter :math:`M` for the test.
        h, alpha = compute_h_alpha_from_input_parameters__CUSUM(horizon, max_nb_random_events, nbArms=nbArms, epsilon=self.epsilon, lmbda=lmbda, M=min_number_of_observation_between_change_point)
        self.threshold_h = h  #: Parameter :math:`h` for the test (threshold).
        self.proba_random_exploration = alpha  #: What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time.

    def __str__(self):
        return r"CUSUM-{}($\alpha={:.3g}$, $M={}${}{})".format(self._policy.__name__, self.proba_random_exploration, self.M, "" if self._per_arm_restart else ", Global", ", lazy detect {}".format(self.lazy_detect_change_only_x_steps) if self.lazy_detect_change_only_x_steps != LAZY_DETECT_CHANGE_ONLY_X_STEPS else "")

    def detect_change(self, arm, verbose=VERBOSE):
        r""" Detect a change in the current arm, using the two-sided CUSUM algorithm [Page, 1954].

        - For each *data* k, compute:

        .. math::

            s_k^- &= (y_k - \hat{u}_0 - \varepsilon) 1(k > M),\\
            s_k^+ &= (\hat{u}_0 - y_k - \varepsilon) 1(k > M),\\
            g_k^+ &= \max(0, g_{k-1}^+ + s_k^+),\\
            g_k^- &= \max(0, g_{k-1}^- + s_k^-).

        - The change is detected if :math:`\max(g_k^+, g_k^-) > h`, where :attr:`threshold_h` is the threshold of the test,
        - And :math:`\hat{u}_0 = \frac{1}{M} \sum_{k=1}^{M} y_k` is the mean of the first M samples, where M is :attr:`M` the min number of observation between change points.
        """
        gp, gm = 0, 0
        data_y = self.all_rewards[arm]
        if len(data_y) <= self.M:
            return False
        # First we use the first M samples to calculate the average :math:`\hat{u_0}`.
        u0hat = np.mean(data_y[:self.M])  # DONE okay this is efficient we don't compute the same means too many times!
        for k, y_k in enumerate(data_y, self.M + 1): # no need to multiply by (k > self.M)
            gp = max(0, gp + (u0hat - y_k - self.epsilon))
            gm = max(0, gm + (y_k - u0hat - self.epsilon))
            if verbose: print("  - For u0hat = {}, k = {}, y_k = {}, gp = {}, gm = {}, and max(gp, gm) = {} compared to threshold h = {}".format(u0hat, k, y_k, gp, gm, max(gp, gm), self.threshold_h))  # DEBUG
            if gp >= self.threshold_h or gm >= self.threshold_h:
                return True
        return False


class PHT_IndexPolicy(CUSUM_IndexPolicy):
    r""" The PHT-UCB generic policy for non-stationary bandits, from [["A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem". F. Liu, J. Lee and N. Shroff. arXiv preprint arXiv:1711.03539, 2017]](https://arxiv.org/pdf/1711.03539).
    """

    def __str__(self):
        return r"PHT-{}($\alpha={:.3g}$, $M={}${}{})".format(self._policy.__name__, self.proba_random_exploration, self.M, "" if self._per_arm_restart else ", Global", ", lazy detect {}".format(self.lazy_detect_change_only_x_steps) if self.lazy_detect_change_only_x_steps != LAZY_DETECT_CHANGE_ONLY_X_STEPS else "")

    def detect_change(self, arm, verbose=VERBOSE):
        r""" Detect a change in the current arm, using the two-sided PHT algorithm [Hinkley, 1971].

        - For each *data* k, compute:

        .. math::

            s_k^- &= y_k - \hat{y}_k - \varepsilon,\\
            s_k^+ &= \hat{y}_k - y_k - \varepsilon,\\
            g_k^+ &= \max(0, g_{k-1}^+ + s_k^+),\\
            g_k^- &= \max(0, g_{k-1}^- + s_k^-).

        - The change is detected if :math:`\max(g_k^+, g_k^-) > h`, where :attr:`threshold_h` is the threshold of the test,
        - And :math:`\hat{y}_k = \frac{1}{k} \sum_{s=1}^{k} y_s` is the mean of the first k samples.
        """
        gp, gm = 0, 0
        data_y = self.all_rewards[arm]
        # First we use the first M samples to calculate the average :math:`\hat{u_0}`.
        y_k_hat = 0
        for k, y_k in enumerate(data_y):
            # y_k_hat = np.mean(data_y[:k+1])  # XXX this is not efficient we compute the same means too many times!
            y_k_hat = (k * y_k_hat + y_k) / (k + 1)  # DONE okay this is efficient we don't compute the same means too many times!
            # Note doing this optimization step improves about 12 times faster!
            gp = max(0, gp + (y_k_hat - y_k - self.epsilon))
            gm = max(0, gm + (y_k - y_k_hat - self.epsilon))
            if verbose: print("  - For y_k_hat = {}, k = {}, y_k = {}, gp = {}, gm = {}, and max(gp, gm) = {} compared to threshold h = {}".format(y_k_hat, k, y_k, gp, gm, max(gp, gm), self.threshold_h))  # DEBUG
            if gp >= self.threshold_h or gm >= self.threshold_h:
                return True
        return False


# --- UCB-CDP based on LCB/UCB Mukherjee & Maillard's paper

#: XXX Be lazy and try to detect changes for :math:`s` taking steps of size ``steps_s``. Default is to have ``steps_s=1``, but only using ``steps_s=2`` should already speed up by 2.
#: It is a simple but efficient way to speed up GLR tests, see https://github.com/SMPyBandits/SMPyBandits/issues/173
#: Default value is 1, to not use this feature, and 10 should speed up the test by x10.
LAZY_TRY_VALUE_S_ONLY_X_STEPS = 1
LAZY_TRY_VALUE_S_ONLY_X_STEPS = 4


class UCBLCB_IndexPolicy(CD_IndexPolicy):
    r""" The UCBLCB-UCB generic policy for non-stationary bandits, from [[Improved Changepoint Detection for Piecewise i.i.d Bandits, by S. Mukherjee  & O.-A. Maillard, preprint 2018](https://subhojyoti.github.io/pdf/aistats_2019.pdf)].

    .. warning:: This is still experimental! See https://github.com/SMPyBandits/SMPyBandits/issues/177
    """
    def __init__(self, nbArms,
            delta=None, delta0=1.0,
            lazy_try_value_s_only_x_steps=LAZY_TRY_VALUE_S_ONLY_X_STEPS,
            *args, **kwargs
        ):
        super(UCBLCB_IndexPolicy, self).__init__(nbArms, per_arm_restart=False, *args, **kwargs)
        # New parameters
        self.proba_random_exploration = 0  #: What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time.
        self.lazy_try_value_s_only_x_steps = lazy_try_value_s_only_x_steps  #: Be lazy and try to detect changes for :math:`s` taking steps of size ``steps_s``.
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
        - Let :math:`S_{i,t:t'} = \srqt{\frac{\log(4 t^2 / \delta)}{2 N_{i,t:t'}}}` the length of the confidence interval.

        - When we have data starting at :math:`t_0=0` (since last restart) and up-to current time :math:`t`, for each *arm* i,
            - For each intermediate time steps :math:`t' \in [t_0, t)`,
                - Compute :math:`LCB_{before} = \hat{\mu}_{i,t_0:t'} - S_{i,t_0:t'}`,
                - Compute :math:`UCB_{before} = \hat{\mu}_{i,t_0:t'} + S_{i,t_0:t'}`,
                - Compute :math:`LCB_{after} = \hat{\mu}_{i,t'+1:t} - S_{i,t'+1:t}`,
                - Compute :math:`UCB_{after} = \hat{\mu}_{i,t'+1:t} + S_{i,t'+1:t}`,
                - If :math:`UCB_{before} < LCB_{after}` or :math:`UCB_{after} < LCB_{before}`, then restart.
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
            for s in range(t0, t, self.lazy_try_value_s_only_x_steps):
                y = data_y[s]
                mean_before = (s * mean_before + y) / (s + 1)
                mean_after = ((t + 1 - s + t0) * mean_after - y) / (t - s + t0)

                ucb_lcb_cst = sqrt(log(4 * t / self.delta(t)) / 2.0)
                S_before = ucb_lcb_cst / sqrt(s + 1)
                S_after  = ucb_lcb_cst / sqrt(t - s + t0)
                ucb_after  = mean_after  + S_after
                lcb_before = mean_before - S_before

                if ucb_after < lcb_before:
                    if verbose: print("  - For arm = {}, t0 = {}, s = {}, t = {}, the mean before mu(t0,s) = {:.3g} and the mean after mu(s+1,t) = {:.3g} and the S_before = {:.3g} and S_after = {:.3g}, so UCB_after = {:.3g} < LCB_before = {:.3g}...".format(armId, t0, s, t, mean_before, mean_after, S_before, S_after, ucb_after, lcb_before))
                    return True

                ucb_before = mean_before + S_before
                lcb_after  = mean_after  - S_after

                if ucb_before < lcb_after:
                    if verbose: print("  - For arm = {}, t0 = {}, s = {}, t = {}, the mean before mu(t0,s) = {:.3g} and the mean after mu(s+1,t) = {:.3g} and the S_before = {:.3g} and S_after = {:.3g}, so UCB_before = {:.3g} < LCB_after = {:.3g}...".format(armId, t0, s, t, mean_before, mean_after, S_before, S_after, ucb_before, lcb_after))
                    return True
            return False



# --- Generic GLR for 1-dimensional exponential families

eps = 1e-10  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]

# --- Simple Kullback-Leibler divergence for known distributions


def klBern(x, y):
    r""" Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution#Kullback.E2.80.93Leibler_divergence

    .. math:: \mathrm{KL}(\mathcal{B}(x), \mathcal{B}(y)) = x \log(\frac{x}{y}) + (1-x) \log(\frac{1-x}{1-y}).
    """
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))


def klGauss(x, y, sig2x=1):
    r""" Kullback-Leibler divergence for Gaussian distributions of means ``x`` and ``y`` and variances ``sig2x`` and ``sig2y``, :math:`\nu_1 = \mathcal{N}(x, \sigma_x^2)` and :math:`\nu_2 = \mathcal{N}(y, \sigma_x^2)`:

    .. math:: \mathrm{KL}(\nu_1, \nu_2) = \frac{(x - y)^2}{2 \sigma_y^2} + \frac{1}{2}\left( \frac{\sigma_x^2}{\sigma_y^2} - 1 \log\left(\frac{\sigma_x^2}{\sigma_y^2}\right) \right).

    See https://en.wikipedia.org/wiki/Normal_distribution#Other_properties
    """
    return (x - y) ** 2 / (2. * sig2x)


def threshold_GaussianGLR(s, t, horizon=None, delta=None):
    r""" Compute the value :math:`c from the corollary of of Theorem 2 from ["Sequential change-point detection: Laplace concentration of scan statistics and non-asymptotic delay bounds", O.-A. Maillard, 2018].

    - The threshold is computed as (with :math:`t_0 = 0`):

    .. math:: \beta(t_0, t, \delta) := \left(1 + \frac{1}{t - t_0 + 1}\right) 2 \log\left(\frac{2 (t - t_0) \sqrt{(t - t_0) + 2}}{\delta}\right).
    """
    if delta is None:
        delta = 1.0 / int(max(1, horizon))
    c = (1 + (1.0 / (t + 1.0))) * log((2 * t * sqrt(t + 2)) / delta)
    if c < 0 or isinf(c):
        c = float('+inf')
    return c


def threshold_BernoulliGLR(s, t, horizon=None, delta=None):
    r""" Compute the value :math:`c from the corollary of of Theorem 2 from ["Sequential change-point detection: Laplace concentration of scan statistics and non-asymptotic delay bounds", O.-A. Maillard, 2018].

    - The threshold is computed as:

    .. math:: \beta(t, \delta) := \log(\frac{1}{\delta}) + \log(1 + \log(s)) + \log(1 + \log(t - s)).

    .. warning:: FIXME This is still experimental! We need to finish the maths before deciding what threshold to use!
    """
    if delta is None:
        delta = 1.0 / horizon
    # c = -log(delta) + log(1 + log(s)) + log(1 + log(t-s))
    c = -log(delta) + log(s) + log(t-s)
    if c < 0 or isinf(c):
        c = float('+inf')
    return c


EXPONENT_BETA = 1.01  #: The default value of parameter :math:`\beta` for the function :func:`decreasing_alpha__GLR`.
ALPHA_T1 = 0.05  #: The default value of parameter :math:`\alpha_{t=1}` for the function :func:`decreasing_alpha__GLR`.


def decreasing_alpha__GLR(alpha0=None, t=1, exponentBeta=EXPONENT_BETA, alpha_t1=ALPHA_T1):
    r""" Either use a fixed alpha, or compute it with an exponential decay (if ``alpha0=None``).

    .. note:: I am currently exploring the following variant (November 2018):

        - The probability of uniform exploration, :math:`\alpha`, is computed as a function of the current time:

        .. math:: \forall t>0, \alpha = \alpha_t := \alpha_{t=1} \frac{1}{\max(1, t^{\beta})}.

        - with :math:`\beta > 1, \beta` = ``exponentBeta`` (=1.05) and :math:`\alpha_{t=1} < 1, \alpha_{t=1}` = ``alpha_t1`` (=0.01).
        - the only requirement on :math:`\alpha_t` seems to be that `\sum_{t=1}^T \alpha_t < +\infty` (ie. be finite), which is the case for :math:`\alpha_t = \alpha = \frac{1}{T}`, but also any :math:`\alpha_t = \frac{\alpha_1}{t^{\beta}}` for any :math:`\beta>1` (cf. Riemann series).
    """
    assert exponentBeta > 1.0, "Error: decreasing_alpha__GLR should have a exponentBeta > 1 but it was given = {}...".format(exponentBeta)  # DEBUG
    if alpha0 is not None:
        return alpha0
    return alpha_t1 / max(1, t)**exponentBeta


def smart_alpha_from_T_UpsilonT(horizon=1, max_nb_random_events=1, scaleFactor=ALPHA0_SCALE_FACTOR):
    r""" Compute a smart estimate of the optimal value for the *fixed* forced exploration probability :math:`\alpha`.

    .. math:: \alpha = \mathrm{scaleFactor} \times \sqrt{\frac{\Upsilon_T}{T} \log(\frac{T}{\Upsilon_T})}
    """
    ratio = max_nb_random_events / float(horizon)
    assert 0 < ratio <= 1, "Error: Upsilon_T = {} should be smaller than horizon T = {}...".format(max_nb_random_events, horizon)  # DEBUG
    alpha = scaleFactor * sqrt(- ratio * log(ratio))
    print("DEBUG: smart_alpha_from_T_UpsilonT: horizon = {}, max_nb_random_events = {}, gives alpha = {}...".format(horizon, max_nb_random_events, alpha))  # DEBUG
    return alpha


class GLR_IndexPolicy(CD_IndexPolicy):
    r""" The GLR-UCB generic policy for non-stationary bandits, using the Generalized Likelihood Ratio test (GLR), for 1-dimensional exponential families.

    - It works for any 1-dimensional exponential family, you just have to give a ``kl`` function.
    - For instance :func:`kullback.klBern`, for Bernoulli distributions, gives :class:`GaussianGLR_IndexPolicy`,
    - And :func:`kullback.klGauss` for univariate Gaussian distributions, gives :class:`BernoulliGLR_IndexPolicy`.

    - ``threshold_function`` computes the threshold :math:`\beta(s, t, \delta)`, it can be for instance :func:`threshold_GaussianGLR` or :func:`threshold_BernoulliGLR`.

    - From ["Sequential change-point detection: Laplace concentration of scan statistics and non-asymptotic delay bounds", O.-A. Maillard, 2018].
    """
    def __init__(self, nbArms,
            horizon=None, delta=None, max_nb_random_events=None,
            kl=klGauss,
            alpha0=None, exponentBeta=EXPONENT_BETA, alpha_t1=ALPHA_T1,
            threshold_function=threshold_BernoulliGLR,
            lazy_try_value_s_only_x_steps=LAZY_TRY_VALUE_S_ONLY_X_STEPS,
            *args, **kwargs
        ):
        super(GLR_IndexPolicy, self).__init__(nbArms, epsilon=1, *args, **kwargs)
        # New parameters
        self.horizon = horizon  #: The horizon :math:`T`.
        self.max_nb_random_events = max_nb_random_events  #: The number of breakpoints :math:`\Upsilon_T`.
        # if delta is None and horizon is not None: delta = 1.0 / horizon
        self.delta = delta  #: The confidence level :math:`\delta`. Defaults to :math:`\delta=\frac{1}{T}` if ``horizon`` is given and ``delta=None``.
        self._exponentBeta = exponentBeta
        self._alpha_t1 = alpha_t1
        if alpha0 is None and horizon is not None and max_nb_random_events is not None:
            alpha0 = smart_alpha_from_T_UpsilonT(horizon=self.horizon, max_nb_random_events=self.max_nb_random_events)
        self._alpha0 = alpha0
        self._threshold_function = threshold_function
        self._args_to_kl = tuple()  # Tuple of extra arguments to give to the :attr:`kl` function.
        self.kl = kl  #: The parametrized Kullback-Leibler divergence (:math:`\mathrm{kl}(x,y) = KL(D(x),D(y))`) for the 1-dimensional exponential family :math:`x\mapsto D(x)`. Example: :func:`kullback.klBern` or :func:`kullback.klGauss`.
        self.lazy_try_value_s_only_x_steps = lazy_try_value_s_only_x_steps  #: Be lazy and try to detect changes for :math:`s` taking steps of size ``steps_s``.

    def compute_threshold_h(self, s, t):
        """Compute the threshold :math:`h` with :attr:`_threshold_function`."""
        return self._threshold_function(s, t, horizon=self.horizon, delta=self.delta)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def proba_random_exploration(self):
        r"""What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time."""
        if self._alpha0 is not None:
            return self._alpha0
        else:
            smallest_time_since_last_restart = np.min(self.last_pulls)
            t = min(self.t, 2 * smallest_time_since_last_restart)
            return decreasing_alpha__GLR(alpha0=self._alpha0, t=t, exponentBeta=self._exponentBeta, alpha_t1=self._alpha_t1)

    def __str__(self):
        class_name = self.__class__.__name__
        name = "Gaussian"
        if "Bernoulli" in class_name:
            name = "Bernoulli"
        if "Sub" in class_name:
            name = "Sub{}".format(name)
        with_tracking = ", tracking" if "WithTracking" in class_name else ""
        with_deterministicexploration = ", determ.explo." if "DeterministicExploration" in class_name else ""
        return r"{}-GLR-{}({}{}, {}{}{}{}{})".format(
            name,
            self._policy.__name__,
            "" if self._per_arm_restart else "Global, ",
            r"$\delta={:.3g}$".format(self.delta) if self.delta is not None else r"$\delta=\frac{1}{T}$",
            r"$\alpha={:.3g}$".format(self._alpha0) if self._alpha0 is not None else r"decreasing $\alpha_t$",
            ", lazy detect {}".format(self.lazy_detect_change_only_x_steps) if self.lazy_detect_change_only_x_steps != LAZY_DETECT_CHANGE_ONLY_X_STEPS else "",
            ", lazy s {}".format(self.lazy_try_value_s_only_x_steps) if self.lazy_try_value_s_only_x_steps != LAZY_TRY_VALUE_S_ONLY_X_STEPS else "",
            with_tracking,
            with_deterministicexploration,
        )

    def detect_change(self, arm, verbose=VERBOSE):
        r""" Detect a change in the current arm, using the Generalized Likelihood Ratio test (GLR) and the :attr:`kl` function.

        - For each *time step* :math:`s` between :math:`t_0=0` and :math:`t`, compute:

        .. math::

            G^{\mathrm{kl}}_{t_0:s:t} = (s-t_0+1) \mathrm{kl}(\mu_{t_0,s}, \mu_{t_0,t}) + (t-s) \mathrm{kl}(\mu_{s+1,t}, \mu_{t_0,t}).

        - The change is detected if there is a time :math:`s` such that :math:`G^{\mathrm{kl}}_{t_0:s:t} > h`, where :attr:`threshold_h` is the threshold of the test,
        - And :math:`\mu_{a,b} = \frac{1}{b-a+1} \sum_{s=a}^{b} y_s` is the mean of the samples between :math:`a` and :math:`b`.

        .. warning:: This is computationally costly, so an easy way to speed up this test is to use :attr:`lazy_try_value_s_only_x_steps` :math:`= \mathrm{Step_s}` for a small value (e.g., 10), so not test for all :math:`s\in[t_0, t-1]` but only :math:`s\in[t_0, t-1], s % \mathrm{Step_s} = 0` (e.g., one out of every 10 steps).
        """
        data_y = self.all_rewards[arm]
        t0 = 0
        t = len(data_y)-1
        mean_all = np.mean(data_y[t0 : t+1])
        mean_before = 0.0
        mean_after = mean_all
        for s in range(t0, t, self.lazy_try_value_s_only_x_steps):
            # XXX nope, that was a mistake: it is only true for the Gaussian kl !
            # this_kl = self.kl(mu(s+1, t), mu(s), *self._args_to_kl)
            # glr = ((s - t0 + 1) * (t - s) / (t - t0 + 1)) * this_kl
            # FIXED this is the correct formula!
            # XXX this is not efficient we compute the same means too many times!
            # mean_before = np.mean(data_y[t0 : s+1])
            # mean_after = np.mean(data_y[s+1 : t+1])
            # DONE okay this is efficient we don't compute the same means too many times!
            y = data_y[s]
            mean_before = (s * mean_before + y) / (s + 1)
            mean_after = ((t + 1 - s + t0) * mean_after - y) / (t - s + t0)
            kl_before = self.kl(mean_before, mean_all, *self._args_to_kl)
            kl_after  = self.kl(mean_after, mean_all, *self._args_to_kl)
            glr = (s - t0 + 1) * kl_before + (t - s) * kl_after
            threshold_h = self.compute_threshold_h(s + 1, t + 1)
            if verbose: print("  - For t0 = {}, s = {}, t = {}, the mean before mu(t0,s) = {} and the mean after mu(s+1,t) = {} and the total mean mu(t0,t) = {}, so the kl before = {} and kl after = {} and GLR = {}, compared to c = {}...".format(t0, s, t, mean_before, mean_after, mean_all, kl_before, kl_after, glr, threshold_h))
            if glr >= threshold_h:
                return True
        return False


class GLR_IndexPolicy_WithTracking(GLR_IndexPolicy):
    """ A variant of the GLR policy where the exploration is not forced to be uniformly random but based on a tracking of arms that haven't been explored enough (with a tracking).

    .. warning:: FIXME this is still experimental!
    """
    def choice(self):
        r""" If any arm is not explored enough (:math:`n_k < \leq \frac{\alpha}{K} \times (t - n_k)`, play uniformly at random one of these arms, otherwise, pass the call to :meth:`choice` of the underlying policy.
        """
        number_of_explorations = self.last_pulls
        min_number_of_explorations = self.proba_random_exploration * (self.t - self.last_restart_times) / self.nbArms
        not_explored_enough = np.where(number_of_explorations <= min_number_of_explorations)[0]
        # FIXME check numerically what I want to prove mathematically
        # for arm in range(self.nbArms):
        #     if number_of_explorations[arm] > 0:
        #         assert number_of_explorations[arm] >= self.proba_random_exploration * (self.t - self.last_restart_times[arm]) / self.nbArms**2, "Error: for arm k={}, the number of exploration n_k(t) = {} was not >= alpha={} / K={}**2 * (t={} - tau_k(t)={}) and RHS was = {}...".format(arm, number_of_explorations[arm], self.proba_random_exploration, self.nbArms, self.t, self.last_restart_times[arm], self.proba_random_exploration * (self.t - self.last_restart_times[arm]) / self.nbArms**2)  # DEBUG
        if len(not_explored_enough) > 0:
            return np.random.choice(not_explored_enough)
        return self.policy.choice()


class GLR_IndexPolicy_WithDeterministicExploration(GLR_IndexPolicy):
    """ A variant of the GLR policy where the exploration is not forced to be uniformly random but deterministic, inspired by what M-UCB proposed.

    - If :math:`t` is the current time and :math:`\tau` is the latest restarting time, then uniform exploration is done if:

    .. math::

        A &:= (t - \tau) \mod \lceil \frac{K}{\gamma} \rceil,\\
        A &\leq K \implies A_t = A.
    """
    def choice(self):
        r""" For some time steps, play uniformly at random one of these arms, otherwise, pass the call to :meth:`choice` of the underlying policy.
        """
        latest_restart_times = np.max(self.last_restart_times)
        A = (self.t - latest_restart_times) % int(np.ceil(self.nbArms / self.proba_random_exploration))
        if A < self.nbArms:
            return int(A)
        return self.policy.choice()


# --- GLR for sigma=1 Gaussian
class GaussianGLR_IndexPolicy(GLR_IndexPolicy):
    r""" The GaussianGLR-UCB policy for non-stationary bandits, for fixed-variance Gaussian distributions (ie, :math:`\sigma^2`=``sig2`` known and fixed).
    """
    def __init__(self, nbArms, sig2=0.25, kl=klGauss, threshold_function=threshold_GaussianGLR, *args, **kwargs):
        super(GaussianGLR_IndexPolicy, self).__init__(nbArms, kl=kl, threshold_function=threshold_function, *args, **kwargs)
        self._sig2 = sig2  #: Fixed variance :math:`\sigma^2` of the Gaussian distributions. Extra parameter given to :func:`kullback.klGauss`. Default to :math:`\sigma^2 = \frac{1}{4}`.
        self._args_to_kl = (sig2, )


class GaussianGLR_IndexPolicy_WithTracking(GLR_IndexPolicy_WithTracking, GaussianGLR_IndexPolicy):
    """ A variant of the GaussianGLR-UCB policy where the exploration is not forced to be uniformly random but based on a tracking of arms that haven't been explored enough.
    """
    pass

class GaussianGLR_IndexPolicy_WithDeterministicExploration(GLR_IndexPolicy_WithDeterministicExploration, GaussianGLR_IndexPolicy):
    """ A variant of the GaussianGLR-UCB policy where the exploration is not forced to be uniformly random but deterministic, inspired by what M-UCB proposed.
    """
    pass


# --- GLR for Bernoulli
class BernoulliGLR_IndexPolicy(GLR_IndexPolicy):
    r""" The BernoulliGLR-UCB policy for non-stationary bandits, for Bernoulli distributions.
    """
    def __init__(self, nbArms, kl=klBern, threshold_function=threshold_BernoulliGLR, *args, **kwargs):
        super(BernoulliGLR_IndexPolicy, self).__init__(nbArms, kl=kl, threshold_function=threshold_function, *args, **kwargs)


class BernoulliGLR_IndexPolicy_WithTracking(GLR_IndexPolicy_WithTracking, BernoulliGLR_IndexPolicy):
    """ A variant of the BernoulliGLR-UCB policy where the exploration is not forced to be uniformly random but based on a tracking of arms that haven't been explored enough."""
    pass

class BernoulliGLR_IndexPolicy_WithDeterministicExploration(GLR_IndexPolicy_WithDeterministicExploration, BernoulliGLR_IndexPolicy):
    """ A variant of the BernoulliGLR-UCB policy where the exploration is not forced to be uniformly random but deterministic, inspired by what M-UCB proposed.
    """
    pass


# --- Non-Parametric Sub-Gaussian GLR for Sub-Gaussian data

#: Default confidence level for :class:`SubGaussianGLR_IndexPolicy`.
SubGaussianGLR_DELTA = 0.01

#: By default, :class:`SubGaussianGLR_IndexPolicy` assumes distributions are 0.25-sub Gaussian, like Bernoulli or any distributions with support on :math:`[0,1]`.
SubGaussianGLR_SIGMA = 0.25

#: Whether to use the joint or disjoint threshold function (:func:`threshold_SubGaussianGLR_joint` or :func:`threshold_SubGaussianGLR_disjoint`) for :class:`SubGaussianGLR_IndexPolicy`.
SubGaussianGLR_JOINT = True

def threshold_SubGaussianGLR_joint(s, t, delta=SubGaussianGLR_DELTA, sigma=SubGaussianGLR_SIGMA):
    r""" Compute the threshold :math:`b^{\text{joint}}_{t_0}(s,t,\delta) according to this formula:

    .. math:: b^{\text{joint}}_{t_0}(s,t,\delta) := \sigma \sqrt{ \left(\frac{1}{s-t_0+1} + \frac{1}{t-s}\right) \left(1 + \frac{1}{t-t_0+1}\right) 2 \log\left( \frac{2(t-t_0)\sqrt{t-t_0+2}}{\delta} \right)}.
    """
    return sigma * sqrt(
        (1.0 / (s + 1) + 1.0 / (t - s)) * (1.0 + 1.0/(t + 1))
        * 2 * max(0, log(( 2 * t * sqrt(t + 2)) / delta ))
    )

def threshold_SubGaussianGLR_disjoint(s, t, delta=SubGaussianGLR_DELTA, sigma=SubGaussianGLR_SIGMA):
    r""" Compute the threshold :math:`b^{\text{disjoint}}_{t_0}(s,t,\delta)` according to this formula:

    .. math:: b^{\text{disjoint}}_{t_0}(s,t,\delta) := \sqrt{2} \sigma \sqrt{\frac{1 + \frac{1}{s - t_0 + 1}}{s - t_0 + 1} \log\left( \frac{4 \sqrt{s - t_0 + 2}}{\delta}\right)} + \sqrt{\frac{1 + \frac{1}{t - s + 1}}{t - s + 1} \log\left( \frac{4 (t - t_0) \sqrt{t - s + 1}}{\delta}\right)}.
    """
    return sqrt(2) * sigma * (sqrt(
        ((1.0 + (1.0 / (s + 1))) / (s + 1)) * max(0, log( (4 * sqrt(s + 2)) / delta ))
    ) + sqrt(
        ((1.0 + (1.0 / (t - s + 1))) / (t - s + 1)) * max(0, log( (4 * t * sqrt(t - s + 1)) / delta ))
    ))

def threshold_SubGaussianGLR(s, t, delta=SubGaussianGLR_DELTA, sigma=SubGaussianGLR_SIGMA, joint=SubGaussianGLR_JOINT):
    r""" Compute the threshold :math:`b^{\text{joint}}_{t_0}(s,t,\delta)` or :math:`b^{\text{disjoint}}_{t_0}(s,t,\delta)`."""
    if joint:
        return threshold_SubGaussianGLR_joint(s, t, delta=delta, sigma=sigma)
    else:
        return threshold_SubGaussianGLR_disjoint(s, t, delta=delta, sigma=sigma)


class SubGaussianGLR_IndexPolicy(CD_IndexPolicy):
    r""" The SubGaussianGLR-UCB policy for non-stationary bandits, using the Generalized Likelihood Ratio test (GLR), for sub-Gaussian distributions.

    - It works for any sub-Gaussian family of distributions, being :math:`\sigma^2`-sub Gaussian *with known* :math:`\sigma`.

    - From ["Sequential change-point detection: Laplace concentration of scan statistics and non-asymptotic delay bounds", O.-A. Maillard, 2018].
    """
    def __init__(self, nbArms,
            horizon=None, max_nb_random_events=None,
            full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
            policy=DefaultPolicy,
            delta=SubGaussianGLR_DELTA, sigma=SubGaussianGLR_SIGMA, joint=SubGaussianGLR_JOINT,
            exponentBeta=1.05, alpha_t1=0.1, alpha0=None,
            lazy_detect_change_only_x_steps=LAZY_DETECT_CHANGE_ONLY_X_STEPS,
            lazy_try_value_s_only_x_steps=LAZY_TRY_VALUE_S_ONLY_X_STEPS,
            *args, **kwargs
        ):
        super(SubGaussianGLR_IndexPolicy, self).__init__(nbArms, epsilon=1, full_restart_when_refresh=full_restart_when_refresh, policy=policy, lazy_detect_change_only_x_steps=lazy_detect_change_only_x_steps, *args, **kwargs)
        # New parameters
        self.horizon = horizon  #: The horizon :math:`T`.
        self.max_nb_random_events = max_nb_random_events  #: The number of breakpoints :math:`\Upsilon_T`.
        if delta is None and horizon is not None: delta = 1.0 / horizon
        self.delta = delta  #: The confidence level :math:`\delta`. Defaults to :math:`\delta=\frac{1}{T}` if ``horizon`` is given and ``delta=None``.
        self.sigma = sigma  #: Parameter :math:`\sigma` for the Sub-Gaussian-GLR test.
        self.joint = joint  #: Parameter ``joint`` for the Sub-Gaussian-GLR test.
        self._exponentBeta = exponentBeta
        self._alpha_t1 = alpha_t1
        if alpha0 is None and horizon is not None and max_nb_random_events is not None:
            alpha0 = smart_alpha_from_T_UpsilonT(horizon=self.horizon, max_nb_random_events=self.max_nb_random_events)
        self._alpha0 = alpha0
        self.lazy_try_value_s_only_x_steps = lazy_try_value_s_only_x_steps  #: Be lazy and try to detect changes for :math:`s` taking steps of size ``steps_s``.

    def compute_threshold_h(self, s, t):
        """Compute the threshold :math:`h` with :func:`threshold_SubGaussianGLR`."""
        return threshold_SubGaussianGLR(s, t, delta=self.delta, sigma=self.sigma, joint=self.joint)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def proba_random_exploration(self):
        r"""What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time."""
        if self._alpha0 is not None:
            return self._alpha0
        smallest_time_since_last_restart = np.min(self.last_pulls)
        t = min(self.t, 2 * smallest_time_since_last_restart)
        return decreasing_alpha__GLR(alpha0=self._alpha0, t=t, exponentBeta=self._exponentBeta, alpha_t1=self._alpha_t1)

    def __str__(self):
        args0 = r"$\delta={}$, $\sigma={:.3g}$".format("{:.3g}".format(self.delta) if self.delta else "1/T", self.sigma)
        args1 = "{}{}".format(
            "joint" if self.joint else "disjoint",
            "" if self._per_arm_restart else ", Global"
        )
        args2 = "{}{}{}".format(
            r"$\alpha={:.3g}$".format(self._alpha0) if self._alpha0 is not None else r"decreasing $\alpha_t$",
            ", lazy detect {}".format(self.lazy_detect_change_only_x_steps) if self.lazy_detect_change_only_x_steps != LAZY_DETECT_CHANGE_ONLY_X_STEPS else "",
            ", lazy s {}".format(self.lazy_try_value_s_only_x_steps) if self.lazy_try_value_s_only_x_steps != LAZY_TRY_VALUE_S_ONLY_X_STEPS else "")
        return r"SubGaussian-GLR-{}({}, {}, {})".format(self._policy.__name__, args0, args1, args2)

    def detect_change(self, arm, verbose=VERBOSE):
        r""" Detect a change in the current arm, using the non-parametric sub-Gaussian Generalized Likelihood Ratio test (GLR) works like this:

        - For each *time step* :math:`s` between :math:`t_0=0` and :math:`t`, compute:

        .. math:: G^{\text{sub-}\sigma}_{t_0:s:t} = |\mu_{t_0,s} - \mu_{s+1,t}|.

        - The change is detected if there is a time :math:`s` such that :math:`G^{\text{sub-}\sigma}_{t_0:s:t} > b_{t_0}(s,t,\delta)`, where :math:`b_{t_0}(s,t,\delta)` is the threshold of the test,

        - The threshold is computed as:

        .. math:: b_{t_0}(s,t,\delta) := \sigma \sqrt{ \left(\frac{1}{s-t_0+1} + \frac{1}{t-s}\right) \left(1 + \frac{1}{t-t_0+1}\right) 2 \log\left( \frac{2(t-t_0)\sqrt{t-t_0+2}}{\delta} \right)}.

        - And :math:`\mu_{a,b} = \frac{1}{b-a+1} \sum_{s=a}^{b} y_s` is the mean of the samples between :math:`a` and :math:`b`.
        """
        data_y = self.all_rewards[arm]
        t0 = 0
        t = len(data_y)-1
        mean_before = 0.0
        mean_after = np.mean(data_y)
        for s in range(t0, t, self.lazy_try_value_s_only_x_steps):
            # XXX this is not efficient we compute the same means too many times!
            # mean_before = np.mean(data_y[t0 : s+1])
            # mean_after = np.mean(data_y[s+1 : t+1])
            # DONE okay this is efficient we don't compute the same means too many times!
            y = data_y[s]
            mean_before = (s * mean_before + y) / (s + 1)
            mean_after = ((t + 1 - s + t0) * mean_after - y) / (t - s + t0)
            glr = abs(mean_after - mean_before)
            # compute threshold
            threshold_h = self.compute_threshold_h(s, t)
            if verbose: print("  - For t0 = {}, s = {}, t = {}, the mean mu(t0,s) = {} and mu(s+1,t) = {} so glr = {}, compared to c = {}...".format(t0, s, t, mean_before, mean_after, glr, threshold_h))
            if glr >= threshold_h:
                return True
        return False
