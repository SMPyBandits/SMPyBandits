# -*- coding: utf-8 -*-
r""" The CD-UCB generic policy and CUSUM-UCB and PHT-UCB policies for non-stationary bandits.

- Reference: [["A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem". F. Liu, J. Lee and N. Shroff. arXiv preprint arXiv:1711.03539, 2017]](https://arxiv.org/pdf/1711.03539)
- It runs on top of a simple policy, e.g., :class:`Policy.UCB.UCB`, and :func:`CUSUM_IndexPolicy` is a wrapper:

    >>> policy = CUSUM_IndexPolicy(nbArms, UCB)
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


#: Default probability of random exploration :math:`\alpha`.
PROBA_RANDOM_EXPLORATION = 0.1

#: Should we fully restart the algorithm or simply reset one arm empirical average ?
FULL_RESTART_WHEN_REFRESH = False

#: Precision of the test.
EPSILON = 0.1

#: Default value of :math:`\lambda`.
LAMBDA = 1

#: Hypothesis on the speed of changes: between two change points, there is at least :math:`M * K` time steps, where K is the number of arms, and M is this constant.
MIN_NUMBER_OF_OBSERVATION_BETWEEN_CHANGE_POINT = 10


from scipy.special import comb

def compute_h_alpha_from_input_parameters(horizon, max_nb_random_events, nbArms, epsilon, lmbda, M):
    r""" Compute the values :math:`C_1^+, C_1^-, C_1, C_2, h` from the formulas in Theorem 2 and Corollary 2 in the paper."""
    # print("horizon = {}, max_nb_random_events = {}, nbArms = {}, epsilon = {}, lmbda = {}, M = {}".format(horizon, max_nb_random_events, nbArms, epsilon, lmbda, M))  # DEBUG
    T = horizon
    UpsilonT = max(1, max_nb_random_events)
    K = nbArms
    C2 = np.log(3) + 2 * np.exp(- 2 * epsilon**2 * M) / lmbda
    C1_minus = np.log((4 * epsilon) / (1-epsilon)**2 * comb(M, int(np.floor(2 * epsilon * M))) * (2 * epsilon)**M + 1)
    C1_plus = np.log((4 * epsilon) / (1+epsilon)**2 * comb(M, int(np.ceil(2 * epsilon * M))) * (2 * epsilon)**M + 1)
    C1 = min(C1_minus, C1_plus)
    if C1 == 0: C1 = 1
    h = 1/C1 * np.log(T / UpsilonT)
    alpha = K * np.sqrt((C2 * UpsilonT)/(C1 * T) * np.log(T / UpsilonT))
    return h, alpha



# --- The very generic class

class CD_IndexPolicy(BaseWrapperPolicy):
    r""" The CD-UCB generic policy for non-stationary bandits, from [["A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem". F. Liu, J. Lee and N. Shroff. arXiv preprint arXiv:1711.03539, 2017]](https://arxiv.org/pdf/1711.03539).
    """
    def __init__(self, nbArms,
            full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
            epsilon=EPSILON,
            proba_random_exploration=PROBA_RANDOM_EXPLORATION,
            policy=DefaultPolicy,
            lower=0., amplitude=1., *args, **kwargs
        ):
        super(CD_IndexPolicy, self).__init__(nbArms, policy=policy, lower=lower, amplitude=amplitude, *args, **kwargs)
        # New parameters
        self.epsilon = epsilon  #: Parameter :math:`\varepsilon` for the test.
        alpha = max(0, min(1, proba_random_exploration))  # crop to [0, 1]
        self.proba_random_exploration = alpha  #: What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time.
        self._full_restart_when_refresh = full_restart_when_refresh  # Should we fully restart the algorithm or simply reset one arm empirical average ?
        # Internal memory
        self.all_rewards = [[] for _ in range(self.nbArms)]  #: Keep in memory all the rewards obtained since the last restart on that arm.
        self.last_pulls = np.full(nbArms, -1)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)

    def __str__(self):
        return r"CD({}, $\varepsilon={:.3g}$, $\alpha={:.3g}$)".format(self._policy.__name__, self.epsilon, self.proba_random_exploration)

    def choice(self):
        r""" With a probability :math:`\alpha`, play uniformly at random, otherwise, pass the call to ``choice`` of the underlying policy."""
        if with_proba(self.proba_random_exploration):
            return np.random.randint(0, self.nbArms - 1)
        return self.policy.choice()

    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the whole empirical average if the change detection algorithm says so.
        """
        super(CD_IndexPolicy, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_pulls[arm] += 1
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
            self.last_pulls[arm] = 1
            self.all_rewards[arm] = [reward]
        # we update the total number of samples available to the underlying policy
        self.policy.t = sum(self.last_pulls)

    def detect_change(self, arm):
        """ Try to detect a change in the current arm.

        .. warning:: This is not implemented for the generic CD algorithm, it has to be implement by a child of the class :class:`CD_IndexPolicy`.
        """
        raise NotImplementedError


# --- Different change detection algorithms

class SlidingWindowRestart_IndexPolicy(CD_IndexPolicy):
    r""" A more generic implementation is the :class:`Policies.SlidingWindowRestart` class.
    """

    def detect_change(self, arm):
        """ Try to detect a change in the current arm.

        .. warning:: This one is simply using a sliding-window of fixed size = 100. A more generic implementation is the :class:`Policies.SlidingWindowRestart` class.
        """
        tau = self.M * self.nbArms
        if self.last_pulls[arm] >= 100 and self.pulls[arm] >= 100:
            # Compute the empirical average for that arm
            empirical_average = self.rewards[arm] / self.pulls[arm]
            # And the small empirical average for that arm
            small_empirical_average = np.mean(self.last_rewards[arm])
            if np.abs(empirical_average - small_empirical_average) >= self.epsilon:
                return True
        return False


# --- Different change detection algorithms

class CUSUM_IndexPolicy(CD_IndexPolicy):
    r""" The CUSUM-UCB generic policy for non-stationary bandits, from [["A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem". F. Liu, J. Lee and N. Shroff. arXiv preprint arXiv:1711.03539, 2017]](https://arxiv.org/pdf/1711.03539).
    """
    def __init__(self, nbArms,
            horizon=None, max_nb_random_events=None,
            full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
            epsilon=EPSILON,
            lmbda=LAMBDA,
            min_number_of_observation_between_change_point=MIN_NUMBER_OF_OBSERVATION_BETWEEN_CHANGE_POINT,
            policy=DefaultPolicy,
            lower=0., amplitude=1., *args, **kwargs
        ):
        super(CUSUM_IndexPolicy, self).__init__(nbArms, epsilon=epsilon, full_restart_when_refresh=full_restart_when_refresh, policy=policy, lower=lower, amplitude=amplitude, *args, **kwargs)
        # New parameters
        self.max_nb_random_events = max_nb_random_events
        self.M = min_number_of_observation_between_change_point  #: Parameter :math:`M` for the test.
        h, alpha = compute_h_alpha_from_input_parameters(horizon, max_nb_random_events, nbArms, epsilon, lmbda, min_number_of_observation_between_change_point)
        self.threshold_h = h  #: Parameter :math:`h` for the test (threshold).
        alpha = max(0, min(1, alpha))  # crop to [0, 1]
        self.proba_random_exploration = alpha  #: What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time.

    def __str__(self):
        return r"CUSUM({}, $\varepsilon={:.3g}$, $\Upsilon_T={:.3g}$, $M={:.3g}$, $h={:.3g}$, $\alpha={:.3g}$)".format(self._policy.__name__, self.epsilon, self.max_nb_random_events, self.M, self.threshold_h, self.proba_random_exploration)

    def detect_change(self, arm):
        """ Detect a change in the current arm, using the two-sided CUSUM algorithm [Page, 1954]."""
        gp, gm = 0, 0
        data_y = self.all_rewards[arm]
        # First we use the first M samples to calculate the average :math:`\hat{u_0}`.
        u0hat = np.mean(data_y[:self.M])
        for k, y_k in enumerate(data_y):
            sp = (u0hat - y_k - self.epsilon) * (k > self.M)
            sm = (y_k - u0hat - self.epsilon) * (k > self.M)
            gp, gm = max(0, gp + sp), max(0, gm + sm)
            if max(gp, gm) >= self.threshold_h:
                return True
        return False


class PHT_IndexPolicy(CUSUM_IndexPolicy):
    r""" The PHT-UCB generic policy for non-stationary bandits, from [["A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem". F. Liu, J. Lee and N. Shroff. arXiv preprint arXiv:1711.03539, 2017]](https://arxiv.org/pdf/1711.03539).
    """

    def __str__(self):
        return r"PHT({}, $\varepsilon={:.3g}$, $\Upsilon_T={:.3g}$, $M={:.3g}$, $h={:.3g}$, $\alpha={:.3g}$)".format(self._policy.__name__, self.epsilon, self.max_nb_random_events, self.M, self.threshold_h, self.proba_random_exploration)

    def detect_change(self, arm):
        """ Detect a change in the current arm, using the two-side PHT algorithm [Hinkley, 1971]."""
        gp, gm = 0, 0
        data_y = self.all_rewards[arm]
        # First we use the first M samples to calculate the average :math:`\hat{u_0}`.
        for k, y_k in enumerate(data_y):
            y_k_hat = np.mean(data_y[:k])
            sp = (y_k_hat - y_k - self.epsilon) * (k > self.M)
            sm = (y_k - y_k_hat - self.epsilon) * (k > self.M)
            gp, gm = max(0, gp + sp), max(0, gm + sm)
            if max(gp, gm) >= self.threshold_h:
                return True
        return False
