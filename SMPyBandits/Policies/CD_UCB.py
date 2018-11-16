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

#: Should we reset one arm empirical average or all?
PER_ARM_RESTART = True

#: Should we fully restart the algorithm or simply reset one arm empirical average ?
FULL_RESTART_WHEN_REFRESH = False

#: Precision of the test.
EPSILON = 0.5

#: Default value of :math:`\lambda`.
LAMBDA = 1

#: Hypothesis on the speed of changes: between two change points, there is at least :math:`M * K` time steps, where K is the number of arms, and M is this constant.
MIN_NUMBER_OF_OBSERVATION_BETWEEN_CHANGE_POINT = 100


from scipy.special import comb

def compute_h_alpha_from_input_parameters__CUSUM(horizon, max_nb_random_events, nbArms, epsilon, lmbda, M):
    r""" Compute the values :math:`C_1^+, C_1^-, C_1, C_2, h` from the formulas in Theorem 2 and Corollary 2 in the paper."""
    T = int(max(1, horizon))
    UpsilonT = int(max(1, max_nb_random_events))
    K = int(max(1, nbArms))
    print("compute_h_alpha_from_input_parameters__CUSUM() with:\nT = {}, UpsilonT = {}, K = {}, epsilon = {}, lmbda = {}, M = {}".format(T, UpsilonT, K, epsilon, lmbda, M))  # DEBUG
    C2 = np.log(3) + 2 * np.exp(- 2 * epsilon**2 * M) / lmbda
    C1_minus = np.log(((4 * epsilon) / (1-epsilon)**2) * comb(M, int(np.floor(2 * epsilon * M))) * (2 * epsilon)**M + 1)
    C1_plus = np.log(((4 * epsilon) / (1+epsilon)**2) * comb(M, int(np.ceil(2 * epsilon * M))) * (2 * epsilon)**M + 1)
    C1 = min(C1_minus, C1_plus)
    if C1 == 0: C1 = 1  # FIXME
    h = 1/C1 * np.log(T / UpsilonT)
    alpha = K * np.sqrt((C2 * UpsilonT)/(C1 * T) * np.log(T / UpsilonT))
    alpha *= 0.01  # FIXME Just divide alpha to not have too large
    alpha = max(0, min(1, alpha))  # crop to [0, 1]
    print("Gave C2 = {}, C1- = {} and C1+ = {} so C1 = {}, and h = {} and alpha = {}".format(C2, C1_minus, C1_plus, C1, h, alpha))  # DEBUG
    return h, alpha


# --- The very generic class

class CD_IndexPolicy(BaseWrapperPolicy):
    r""" The CD-UCB generic policy for non-stationary bandits, from [["A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem". F. Liu, J. Lee and N. Shroff. arXiv preprint arXiv:1711.03539, 2017]](https://arxiv.org/pdf/1711.03539).
    """
    def __init__(self, nbArms,
            full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
            per_arm_restart=PER_ARM_RESTART,
            epsilon=EPSILON,
            # proba_random_exploration=PROBA_RANDOM_EXPLORATION,
            proba_random_exploration=None,
            policy=DefaultPolicy,
            lower=0., amplitude=1., *args, **kwargs
        ):
        super(CD_IndexPolicy, self).__init__(nbArms, policy=policy, lower=lower, amplitude=amplitude, *args, **kwargs)
        # New parameters
        self.epsilon = epsilon  #: Parameter :math:`\varepsilon` for the test.
        if proba_random_exploration is not None:
            alpha = max(0, min(1, proba_random_exploration))  # crop to [0, 1]
            self.proba_random_exploration = alpha  #: What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time.
        self._full_restart_when_refresh = full_restart_when_refresh  # Should we fully restart the algorithm or simply reset one arm empirical average ?
        self._per_arm_restart = per_arm_restart  # Should we reset one arm empirical average or all?
        # Internal memory
        self.all_rewards = [[] for _ in range(self.nbArms)]  #: Keep in memory all the rewards obtained since the last restart on that arm.
        self.last_pulls = np.full(nbArms, -1)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)

    def __str__(self):
        return r"CD-{}($\varepsilon={:.3g}$, $\gamma={:.3g}${})".format(self._policy.__name__, self.epsilon, self.proba_random_exploration, ", Per-Arm" if self._per_arm_restart else ", Global")

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
            print("For a player {} a change was detected at time {} for arm {} after seeing reward = {}!".format(self, self.t, arm, reward))  # DEBUG
            # print("The current pulls vector is =", self.pulls)  # DEBUG
            # print("The current pulls vector of underlying policy is =", self.policy.pulls)  # DEBUG
            # print("The current last pulls vector is =", self.last_pulls)  # DEBUG
            # print("The current rewards vector is =", self.rewards)  # DEBUG
            # print("The current rewards vector of underlying policy is =", self.policy.rewards)  # DEBUG
            # print("The current sum/mean of all rewards for this arm is =", np.sum(self.all_rewards[arm]), np.mean(self.all_rewards[arm]))  # DEBUG

            # Fully restart the algorithm ?!
            if self._full_restart_when_refresh:
                self.startGame(createNewPolicy=False)
            # Or simply reset one of the empirical averages?
            else:
                self.policy.rewards[arm] = np.sum(self.all_rewards[arm])
                self.policy.pulls[arm] = len(self.all_rewards[arm])

            # reset current memory for THIS arm
            if self._per_arm_restart:
                self.last_pulls[arm] = 1
                self.all_rewards[arm] = [reward]
            # or reset current memory for ALL THE arms
            else:
                for other_arm in range(self.nbArms):
                    self.last_pulls[other_arm] = 0
                    self.all_rewards[other_arm] = []
                self.last_pulls[arm] = 1
                self.all_rewards[arm] = [reward]
        # we update the total number of samples available to the underlying policy
        # self.policy.t = sum(self.last_pulls)  # XXX SO NOT SURE HERE

    def detect_change(self, arm):
        """ Try to detect a change in the current arm.

        .. warning:: This is not implemented for the generic CD algorithm, it has to be implement by a child of the class :class:`CD_IndexPolicy`.
        """
        raise NotImplementedError


# --- Different change detection algorithms

class SlidingWindowRestart_IndexPolicy(CD_IndexPolicy):
    r""" A more generic implementation is the :class:`Policies.SlidingWindowRestart` class.

    .. warning:: I have no idea if what I wrote is correct or not!
    """

    def detect_change(self, arm):
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
        h, alpha = compute_h_alpha_from_input_parameters__CUSUM(horizon, max_nb_random_events, nbArms, epsilon, lmbda, min_number_of_observation_between_change_point)
        self.threshold_h = h  #: Parameter :math:`h` for the test (threshold).
        self.proba_random_exploration = alpha  #: What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time.

    def __str__(self):
        return r"CUSUM-{}($\varepsilon={:.3g}$, $\Upsilon_T={:.3g}$, $M={:.3g}$, $h={:.3g}$, $\gamma={:.3g}${})".format(self._policy.__name__, self.epsilon, self.max_nb_random_events, self.M, self.threshold_h, self.proba_random_exploration, ", Per-Arm" if self._per_arm_restart else ", Global")

    def detect_change(self, arm):
        r""" Detect a change in the current arm, using the two-sided CUSUM algorithm [Page, 1954].

        - For each *data* k, compute:

        .. math::

            s_k^- &= (y_k - \hat{u}_0 - \varepsilon) 1(k > M),\\
            s_k^+ &= (\hat{u}_0 - y_k - \varepsilon) 1(k > M),\\
            g_k^+ &= max(0, g_{k-1}^+ + s_k^+),\\
            g_k^- &= max(0, g_{k-1}^- + s_k^-),\\

        - The change is detected if :math:`\max(g_k^+, g_k^-) > h`, where :attr:`threshold_h` is the threshold of the test,
        - And :math:`\hat{u}_0 = \frac{1}{M} \sum_{k=1}^{M} y_k` is the mean of the first M samples, where M is :attr:`M` the min number of observation between change points.
        """
        gp, gm = 0, 0
        data_y = self.all_rewards[arm]
        # First we use the first M samples to calculate the average :math:`\hat{u_0}`.
        u0hat = np.mean(data_y[:self.M])
        for k, y_k in enumerate(data_y):
            if k <= self.M:
                continue
            sp = u0hat - y_k - self.epsilon  # no need to multiply by (k > self.M)
            sm = y_k - u0hat - self.epsilon  # no need to multiply by (k > self.M)
            gp, gm = max(0, gp + sp), max(0, gm + sm)
            if max(gp, gm) >= self.threshold_h:
                return True
        return False


class PHT_IndexPolicy(CUSUM_IndexPolicy):
    r""" The PHT-UCB generic policy for non-stationary bandits, from [["A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem". F. Liu, J. Lee and N. Shroff. arXiv preprint arXiv:1711.03539, 2017]](https://arxiv.org/pdf/1711.03539).
    """

    def __str__(self):
        return r"PHT-{}($\varepsilon={:.3g}$, $\Upsilon_T={:.3g}$, $M={:.3g}$, $h={:.3g}$, $\gamma={:.3g}${})".format(self._policy.__name__, self.epsilon, self.max_nb_random_events, self.M, self.threshold_h, self.proba_random_exploration, ", Per-Arm" if self._per_arm_restart else ", Global")

    def detect_change(self, arm):
        r""" Detect a change in the current arm, using the two-side PHT algorithm [Hinkley, 1971].

        - For each *data* k, compute:

        .. math::

            s_k^- &= y_k - \hat{y}_k - \varepsilon,\\
            s_k^+ &= \hat{y}_k - y_k - \varepsilon,\\
            g_k^+ &= max(0, g_{k-1}^+ + s_k^+),\\
            g_k^- &= max(0, g_{k-1}^- + s_k^-),\\

        - The change is detected if :math:`\max(g_k^+, g_k^-) > h`, where :attr:`threshold_h` is the threshold of the test,
        - And :math:`\hat{y}_k = \frac{1}{k} \sum_{s=1}^{k} y_s` is the mean of the first k samples.
        """
        gp, gm = 0, 0
        data_y = self.all_rewards[arm]
        # First we use the first M samples to calculate the average :math:`\hat{u_0}`.
        for k, y_k in enumerate(data_y):
            y_k_hat = np.mean(data_y[:k])
            sp = y_k_hat - y_k - self.epsilon
            sm = y_k - y_k_hat - self.epsilon
            gp, gm = max(0, gp + sp), max(0, gm + sm)
            if max(gp, gm) >= self.threshold_h:
                return True
        return False


# --- Generic GLR for 1-dimensional exponential families

try:
    from .kullback import klBern, klGauss
except (ImportError, SystemError):
    from kullback import klBern, klGauss

VERBOSE = True
#: Whether to be verbose when doing the search for valid parameter :math:`\ell`.
VERBOSE = False


def compute_c_alpha__GLR(t0, t, horizon, verbose=False):
    r""" Compute the values :math:`c, \alpha` from the corollary of of Theorem 2 from ["Sequential change-point detection: Laplace concentration of scan statistics and non-asymptotic delay bounds", O.-A. Maillard, 2018]."""
    T = int(max(1, horizon))
    delta = 1.0 / T
    if verbose: print("compute_c_alpha__GLR() with t = {}, t0 = {}, T = {}, delta = 1/T = {}".format(t, t0, T, delta))  # DEBUG
    t_m_t0 = abs(t - t0)
    c = (1 + (1 / (t_m_t0 + 1.0))) * 2 * np.log((2 * t_m_t0 * np.sqrt(t_m_t0 + 2)) / delta)
    if c < 0 and np.isinf(c): c = float('+inf')
    alpha = 1.0 / (100 * max(1, t**2))
    if verbose: print("Gave c = {} and alpha = {}".format(c, alpha))  # DEBUG
    # FIXME I just want to try, with a small threshold do we detect changes?
    # c = 20.0
    return c, alpha


class GLR_IndexPolicy(CD_IndexPolicy):
    r""" The GLR-UCB generic policy for non-stationary bandits, using the Generalized Likelihood Ratio test (GLR),  for 1-dimensional exponential families.

    - It works for any 1-dimensional exponential family, you just have to give a ``kl`` function.
    - For instance :func:`kullback.klBern`, for Bernoulli distributions, gives :class:`GaussianGLR_IndexPolicy`,
    - And :func:`kullback.klGauss` for univariate Gaussian distributions, gives :class:`BernoulliGLR_IndexPolicy`.
    """
    def __init__(self, nbArms,
            horizon=None,
            full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
            policy=DefaultPolicy,
            kl=klBern,
            lower=0., amplitude=1., *args, **kwargs
        ):
        super(GLR_IndexPolicy, self).__init__(nbArms, epsilon=1, full_restart_when_refresh=full_restart_when_refresh, policy=policy, lower=lower, amplitude=amplitude, *args, **kwargs)
        # New parameters
        self.horizon = horizon
        c, alpha = compute_c_alpha__GLR(0, 1, self.horizon)
        self._threshold_h, self._alpha = c, alpha
        self._args_to_kl = tuple()  # Tuple of extra arguments to give to the :attr:`kl` function.
        self.kl = kl  #: The parametrized Kullback-Leibler divergence (:math:`\mathrm{kl}(x,y) = KL(D(x),D(y))`) for the 1-dimensional exponential family :math:`x\mapsto D(x)`. Example: :func:`kullback.klBern` or :func:`kullback.klGauss`.

    @property
    def threshold_h(self):
        r"""Parameter :math:`c` for the test (threshold)."""
        c, alpha = compute_c_alpha__GLR(0, self.t, self.horizon)
        self._threshold_h, self._alpha = c, alpha
        return self._threshold_h

    @property
    def proba_random_exploration(self):
        r"""What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time."""
        c, alpha = compute_c_alpha__GLR(0, self.t, self.horizon)
        self._threshold_h, self._alpha = c, alpha
        return self._alpha

    def __str__(self):
        name = self.kl.__name__[2:]
        name = "" if name == "Bern" else name + ", "
        return r"GLR-{}({}$T={}$, $c={:.3g}$, $\gamma={:.3g}${})".format(self._policy.__name__, name, self.horizon, self.threshold_h, self.proba_random_exploration, ", Per-Arm" if self._per_arm_restart else ", Global")

    def detect_change(self, arm, verbose=VERBOSE):
        r""" Detect a change in the current arm, using the Generalized Likelihood Ratio test (GLR) and the :attr:`kl` function.

        - For each *time step* :math:`s` between :math:`t_0=0` and :math:`t`, compute:

        .. math::

            G^{\mathcal{N}_1}_{t_0:s:t} = (s-t_0+1)(t-s) \mathrm{kl}(\mu_{s+1,t}, \mu_{t_0,s}) / (t-t_0+1).

        - The change is detected if there is a time :math:`s` such that :math:`G^{\mathcal{N}_1}_{t_0:s:t} > h`, where :attr:`threshold_h` is the threshold of the test,
        - And :math:`\mu_{a,b} = \frac{1}{b-a+1} \sum_{s=a}^{b} y_s` is the mean of the samples between :math:`a` and :math:`b`.
        """
        data_y = self.all_rewards[arm]
        t0 = 0
        t = len(data_y)
        mu = lambda a, b: np.mean(data_y[a : b+1])
        for s in range(t0, t - 1):
            this_kl = self.kl(mu(s+1, t), mu(t0, s), *self._args_to_kl)
            glr = ((s - t0 + 1) * (t - s) / (t - t0 + 1)) * this_kl
            if verbose: print("  - For t0 = {}, s = {}, t = {}, the mean mu(t0,s) = {} and mu(s+1,t) = {} and so the kl = {} and GLR = {}, compared to c = {}...".format(t0, s, t, mu(t0, s), mu(s+1, t), this_kl, glr, self.threshold_h))
            if glr >= self.threshold_h:
                return True
        return False


# --- GLR for sigma=1 Gaussian
class GaussianGLR_IndexPolicy(GLR_IndexPolicy):
    r""" The GaussianGLR-UCB generic policy for non-stationary bandits, for fixed-variance Gaussian distributions (ie, :math:`\sigma^2` known and fixed).

    .. warning:: FIXME This is HIGHLY experimental!
    """

    def __init__(self, nbArms, horizon=None, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH, sig2=0.25, policy=DefaultPolicy, lower=0., amplitude=1., *args, **kwargs
        ):
        super(GaussianGLR_IndexPolicy, self).__init__(nbArms, horizon=horizon, full_restart_when_refresh=full_restart_when_refresh, policy=policy, kl=klGauss, lower=lower, amplitude=amplitude, *args, **kwargs)
        self.sig2 = sig2  #: Fixed variance :math:`\sigma^2` of the Gaussian distributions. Extra parameter given to :func:`kullback.klGauss`.
        self._args_to_kl = (sig2, )

    def __str__(self):
        return r"GaussianGLR-{}($T={}$, $c={:.3g}$, $\gamma={:.3g}${})".format(self._policy.__name__,  self.horizon, self.threshold_h, self.proba_random_exploration, ", Per-Arm" if self._per_arm_restart else ", Global")

# --- GLR for Bernoulli
class BernoulliGLR_IndexPolicy(GLR_IndexPolicy):
    r""" The BernoulliGLR-UCB generic policy for non-stationary bandits, for Bernoulli distributions.

    .. warning:: FIXME This is HIGHLY experimental!
    """

    def __init__(self, nbArms, horizon=None, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH, policy=DefaultPolicy, lower=0., amplitude=1., *args, **kwargs
        ):
        super(BernoulliGLR_IndexPolicy, self).__init__(nbArms, horizon=horizon, full_restart_when_refresh=full_restart_when_refresh, policy=policy, kl=klBern, lower=lower, amplitude=amplitude, *args, **kwargs)

    def __str__(self):
        return r"BernoulliGLR-{}($T={}$, $c={:.3g}$, $\gamma={:.3g}${})".format(self._policy.__name__,  self.horizon, self.threshold_h, self.proba_random_exploration, ", Per-Arm" if self._per_arm_restart else ", Global")
