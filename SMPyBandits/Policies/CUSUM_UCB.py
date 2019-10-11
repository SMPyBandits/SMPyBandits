# -*- coding: utf-8 -*-
r""" The CUSUM-UCB and PHT-UCB policies for non-stationary bandits.

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
    from .UCB import UCB as DefaultPolicy
    from .CD_UCB import CD_IndexPolicy
except ImportError:
    from with_proba import with_proba
    from UCB import UCB as DefaultPolicy
    from CD_UCB import CD_IndexPolicy


#: Whether to be verbose when doing the change detection algorithm.
VERBOSE = False

#: Default probability of random exploration :math:`\alpha`.
PROBA_RANDOM_EXPLORATION = 0.1

#: Should we reset one arm empirical average or all? For CUSUM-UCB it is ``True`` by default.
PER_ARM_RESTART = True

#: Should we fully restart the algorithm or simply reset one arm empirical average? For CUSUM-UCB it is ``False`` by default.
FULL_RESTART_WHEN_REFRESH = False

#: Precision of the test. For CUSUM/PHT, :math:`\varepsilon` is the drift correction threshold (see algorithm).
EPSILON = 0.01

#: Default value of :math:`\lambda`. Used only if :math:`h` and :math:`\alpha` are computed using :func:`compute_h_alpha_from_input_parameters__CUSUM_complicated`.
LAMBDA = 1

#: Hypothesis on the speed of changes: between two change points, there is at least :math:`M * K` time steps, where K is the number of arms, and M is this constant.
MIN_NUMBER_OF_OBSERVATION_BETWEEN_CHANGE_POINT = 100

#: XXX Be lazy and try to detect changes only X steps, where X is small like 20 for instance.
#: It is a simple but efficient way to speed up CD tests, see https://github.com/SMPyBandits/SMPyBandits/issues/173
#: Default value is 0, to not use this feature, and 20 should speed up the test by x20.
LAZY_DETECT_CHANGE_ONLY_X_STEPS = 1
LAZY_DETECT_CHANGE_ONLY_X_STEPS = 10


#: Default value of ``use_localization`` for policies. All the experiments I tried showed that the localization always helps improving learning, so the default value is set to True.
USE_LOCALIZATION = False
USE_LOCALIZATION = True


# --- Different change detection algorithms

#: For any algorithm with uniform exploration and a formula to tune it, :math:`\alpha` is usually too large and leads to larger regret. Multiplying it by a 0.1 or 0.2 helps, a lot!
ALPHA0_SCALE_FACTOR = 1
# ALPHA0_SCALE_FACTOR = 0.1


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
    if C1 == 0: C1 = 1  # XXX This case of having C1=0 for CUSUM parameters should not happen...
    h = 1/C1 * np.log(T / UpsilonT)
    alpha = K * np.sqrt((C2 * UpsilonT)/(C1 * T) * np.log(T / UpsilonT))
    alpha *= scaleFactor  # XXX Just divide alpha to not have too large, for CUSUM-UCB.
    alpha = max(0, min(1, alpha))  # crop to [0, 1]
    print("Gave C2 = {}, C1- = {} and C1+ = {} so C1 = {}, and h = {} and alpha = {}".format(C2, C1_minus, C1_plus, C1, h, alpha))  # DEBUG
    return h, alpha

def compute_h_alpha_from_input_parameters__CUSUM(horizon, max_nb_random_events, scaleFactor=ALPHA0_SCALE_FACTOR, **kwargs):
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
            full_restart_when_refresh=False,
            per_arm_restart=True,
            use_localization=USE_LOCALIZATION,
            *args, **kwargs
        ):
        super(CUSUM_IndexPolicy, self).__init__(nbArms, full_restart_when_refresh=full_restart_when_refresh, per_arm_restart=per_arm_restart, *args, **kwargs)
        # New parameters
        self.max_nb_random_events = max_nb_random_events
        self.M = min_number_of_observation_between_change_point  #: Parameter :math:`M` for the test.
        h, alpha = compute_h_alpha_from_input_parameters__CUSUM(horizon, max_nb_random_events, nbArms=nbArms, epsilon=self.epsilon, lmbda=lmbda, M=min_number_of_observation_between_change_point)
        self.threshold_h = h  #: Parameter :math:`h` for the test (threshold).
        self.proba_random_exploration = alpha  #: What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time.
        self.use_localization = use_localization  #: Experiment to use localization of the break-point, ie, restart memory of arm by keeping observations s+1...n instead of just the last one

    def __str__(self):
        # return r"CUSUM-{}($\alpha={:.3g}$, $M={}${}{})".format(self._policy.__name__, self.proba_random_exploration, self.M, "" if self._per_arm_restart else ", Global", ", lazy detect {}".format(self.lazy_detect_change_only_x_steps) if self.lazy_detect_change_only_x_steps != LAZY_DETECT_CHANGE_ONLY_X_STEPS else "")
        args = "{}{}{}".format("" if self._per_arm_restart else "Global, ", "Localization, " if self.use_localization else "", "lazy detect {}".format(self.lazy_detect_change_only_x_steps) if self.lazy_detect_change_only_x_steps != LAZY_DETECT_CHANGE_ONLY_X_STEPS else "")
        args = "({})".format(args) if args else ""
        return r"CUSUM-{}{}".format(self._policy.__name__, args)

    def getReward(self, arm, reward):
        r""" Be sure that the underlying UCB or klUCB indexes are used with :math:`\log(n_t)` for the exploration term, where :math:`n_t = \sum_{i=1}^K N_i(t)` the number of pulls of each arm since its last restart times (different restart time for each arm, CUSUM use local restart only)."""
        super(CUSUM_IndexPolicy, self).getReward(arm, reward)
        # FIXED DONE Be sure that CUSUM UCB use log(n_t) in their UCB and not log(t - tau_i)
        # we update the total number of samples available to the underlying policy
        old_policy_t, new_policy_t = self.policy.t, np.sum(self.last_pulls)
        if old_policy_t != new_policy_t:
            # print("==> WARNING: the policy {}, at global time {}, had a sub_policy.t = {} but a total number of pulls of each arm since its last restart times = {}...\n    WARNING: Forcing UCB or klUCB to use this weird t for their log(t) term...".format(self, self.t, old_policy_t, new_policy_t))  # DEBUG
            self.policy.t = new_policy_t  # XXX SO NOT SURE HERE

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
            return False, None
        # First we use the first M samples to calculate the average :math:`\hat{u_0}`.
        u0hat = np.mean(data_y[:self.M])  # DONE okay this is efficient we don't compute the same means too many times!
        for k, y_k in enumerate(data_y, self.M + 1): # no need to multiply by (k > self.M)
            gp = max(0, gp + (u0hat - y_k - self.epsilon))
            gm = max(0, gm + (y_k - u0hat - self.epsilon))
            if verbose: print("  - For u0hat = {}, k = {}, y_k = {}, gp = {}, gm = {}, and max(gp, gm) = {} compared to threshold h = {}".format(u0hat, k, y_k, gp, gm, max(gp, gm), self.threshold_h))  # DEBUG
            if gp >= self.threshold_h or gm >= self.threshold_h:
                return True, k + self.M + 1 if self.use_localization else None
        return False, None


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
                return True, k if self.use_localization else None
        return False, None

