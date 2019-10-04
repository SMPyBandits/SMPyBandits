# -*- coding: utf-8 -*-
r""" The AdSwitch policy for non-stationary bandits, from [["Adaptively Tracking the Best Arm with an Unknown Number of Distribution Changes". Peter Auer, Pratik Gajane and Ronald Ortner]](https://ewrl.files.wordpress.com/2018/09/ewrl_14_2018_paper_28.pdf)

- It uses an additional :math:`\mathcal{O}(\tau_\max)` memory for a game of maximum stationary length :math:`\tau_\max`.

.. warning:: This implementation is still experimental!
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


from enum import Enum  # For the different phases
import numpy as np

try:
    from .BasePolicy import BasePolicy
    from .with_proba import with_proba
except ImportError:
    from BasePolicy import BasePolicy
    from with_proba import with_proba


#: Different phases during the AdSwitch algorithm
Phase = Enum('Phase', ['Estimation', 'Checking', 'Exploitation'])


def mymean(x):
    r""" Simply :func:`numpy.mean` on x if x is non empty, otherwise ``0.0``.

    .. info:: Avoid to see the following warning:

    >>> np.mean([])
    /usr/local/lib/python3.6/dist-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
    """
    return np.mean(x) if len(x) else 0.0


# --- Class

Constant_C1 = 128.0  #: Default value for the constant :math:`C_1`. Should be :math:`>0` and as large as possible, but not too large.
Constant_C1 = 1.0  #: Default value for the constant :math:`C_1`. Should be :math:`>0` and as large as possible, but not too large.

Constant_C2 = 128.0  #: Default value for the constant :math:`C_2`. Should be :math:`>0` and as large as possible, but not too large.
Constant_C2 = 1.0  #: Default value for the constant :math:`C_2`. Should be :math:`>0` and as large as possible, but not too large.


class AdSwitch(BasePolicy):
    r""" The AdSwitch policy for non-stationary bandits, from [["Adaptively Tracking the Best Arm with an Unknown Number of Distribution Changes". Peter Auer, Pratik Gajane and Ronald Ortner]](https://ewrl.files.wordpress.com/2018/09/ewrl_14_2018_paper_28.pdf)
    """

    def __init__(self, nbArms,
            horizon=None, C1=Constant_C1, C2=Constant_C2,
            *args, **kwargs
        ):
        if nbArms > 2:
            print("WARNING: so far, for the AdSwitch algorithm, only the special case of K=2 arms was explained in the paper, but I generalized it. Maybe it does not work!")  # DEBUG
            print("FIXME for the generic case of K > 2 arms, use the AdSwitchNew algorithm (paper from 2019)!")
        super(AdSwitch, self).__init__(nbArms, *args, **kwargs)

        # Parameters
        assert horizon is not None, "Error: for a AdSwitch policy, the parameter 'horizon' should be > 0 and not None but was = {}".format(horizon)  # DEBUG
        self.horizon = horizon  #: Parameter :math:`T` for the AdSwitch algorithm, the horizon of the experiment. TODO try to use :class:`DoublingTrickWrapper` to remove the dependency in :math:`T` ?

        assert C1 > 0, "Error: for a AdSwitch policy, the parameter 'C1' should be > 0 but was = {}".format(C1)  # DEBUG
        self.C1 = C1  #: Parameter :math:`C_1` for the AdSwitch algorithm.

        assert C2 > 0, "Error: for a AdSwitch policy, the parameter 'C2' should be > 0 but was = {}".format(C2)  # DEBUG
        self.C2 = C2  #: Parameter :math:`C_2` for the AdSwitch algorithm.

        # Internal memory
        self.phase = Phase.Estimation  #: Current phase, exploration or exploitation.
        self.current_exploration_arm = None  #: Currently explored arm. It cycles uniformly, in step 2.
        self.current_exploitation_arm = None  #: Currently exploited arm. It is :math:`\overline{a_k}` in the algorithm.
        self.batch_number = 1  #: Number of batch
        self.last_restart_time = 0  #: Time step of the last restart (beginning of phase of Estimation)
        self.length_of_current_phase = None  #: Length of the current tests phase, computed as :math:`s_i`, with :func:`compute_di_pi_si`.
        self.step_of_current_phase = 0  #: Timer inside the current phase.
        # self.all_pulls = np.zeros(self.nbArms) #: Number of pulls of arms since the last restart?
        self.current_best_arm = None  #: Current best arm, when finishing step 3. Denote :math:`\overline{a_k}` in the algorithm.
        self.current_worst_arm = None  #: Current worst arm, when finishing step 3. Denote :math:`\underline{a_k}` in the algorithm.
        self.current_estimated_gap = None  #: Gap between the current best and worst arms, ie largest gap, when finishing step 3. Denote :math:`\widehat{\Delta_k}` in the algorithm.
        self.last_used_di_pi_si = None  #: Memory of the currently used :math:`(d_i, p_i, s_i)`.

        self.all_rewards = [{} for _ in range(self.nbArms)]  #: Memory of all the rewards. A *dictionary* per arm, mapping time to rewards. Growing size until restart of that arm!

    def __str__(self):
        return r"AdSwitch($T={}$, $C_1={:.3g}$, $C_2={:.3g}$)".format(self.horizon, self.C1, self.C2)

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        super(AdSwitch, self).startGame()
        self.phase = Phase.Estimation
        self.current_exploration_arm = None
        self.current_exploitation_arm = None
        self.batch_number = 1
        self.last_restart_time = 0
        self.length_of_current_phase = None
        self.step_of_current_phase = 0
        self.current_best_arm = None
        self.current_worst_arm = None
        self.current_estimated_gap = None
        self.last_used_di_pi_si = None
        self.all_rewards = [{} for _ in range(self.nbArms)]

    def getReward(self, arm, reward):
        """ Get a reward from an arm."""
        super(AdSwitch, self).getReward(arm, reward)
        reward = (reward - self.lower) / self.amplitude
        self.all_rewards[arm][self.t] = reward

    def read_range_of_rewards(self, arm, start, end):
        r""" Read the :attr:`all_rewards` attribute to extract all the rewards for that ``arm``, obtained between time ``start`` (included) and ``end`` (not included)."""
        rewards = self.all_rewards[arm]
        return [
            rewards[t] for t in range(start, end)
            if t in rewards
            and not np.isnan(rewards[t])
        ]

    def statistical_test(self, t, t0):
        r""" Test if at time :math:`t` there is a :math:`\sigma`, :math:`t_0 \leq \sigma < t`, and a pair of arms :math:`a,b`, satisfying this test:

        .. math:: | \hat{\mu_a}[\sigma,t] - \hat{\mu_b}[\sigma,t] | > \sqrt{\frac{C_1 \log T}{t - \sigma}}.

        where :math:`\hat{\mu_a}[t_1,t_2]` is the empirical mean for arm :math:`a` for samples obtained from times :math:`t \in [t_1,t_2)`.

        - Return ``True, sigma`` if the test was satisfied, and the smallest :math:`\sigma` that was satisfying the test, or ``False, None`` otherwise.
        """
        for a in range(self.nbArms):
            for b in range(a + 1, self.nbArms):
                for sigma in range(t0, t):
                    # DONE be sure that I can compute the means like this!
                    mu_a = mymean(self.read_range_of_rewards(a, sigma, t + 1))
                    mu_b = mymean(self.read_range_of_rewards(b, sigma, t + 1))
                    ucb = np.sqrt(self.C1 * np.log(self.horizon) / (t - sigma))
                    # print("Info: test |mu_a[sigma,t] = {} - mu_b[sigma,t] = {}| > UCB = {} is {} for a = {}, b = {}, sigma = {} and t0 = {} and t = {}...".format(mu_a, mu_b, ucb, abs(mu_a - mu_b) > ucb, a, b, sigma, t0, t))  # DEBUG
                    if abs(mu_a - mu_b) > ucb:
                        # print("DEBUG: test |mu_a[sigma,t] = {} - mu_b[sigma,t] = {}| > UCB = {} is {} for a = {}, b = {}, sigma = {} and t0 = {} and t = {}...".format(mu_a, mu_b, ucb, abs(mu_a - mu_b) > ucb, a, b, sigma, t0, t))  # DEBUG
                        return True, sigma
        return False, None

    def find_Ik(self):
        r""" Follow the algorithm and, with a gap estimate :math:`\widehat{\Delta_k}`, find :math:`I_k = \max\{ i : d_i \geq \widehat{\Delta_k} \}`, where :math:`d_i := 2^{-i}`. There is no need to do an exhaustive search:

        .. math:: I_k := \lfloor - \log_2(\widehat{\Delta_k}) \rfloor.
        """
        assert self.current_estimated_gap >= 0, "Error: cannot find Ik if self.current_estimated_gap = {} is not > 0.".format(self.current_estimated_gap)  # DEBUG
        if np.isclose(self.current_estimated_gap, 0):
            Ik = 1
        else:
            # XXX Direct formula!
            Ik = max(1, int(np.floor(- np.log2(self.current_estimated_gap))))
        # print("DEBUG: at time t = {}, with gap Delta_k = {}, I_k = {}".format(self.t, self.current_estimated_gap, Ik))  # DEBUG
        return Ik
        # # XXX Manual search
        # i = 0
        # while (2**(-i)) >= self.current_estimated_gap:
        #     i += 1
        # Ik = i - 1
        # return Ik

    def compute_di_pi_si(self):
        r""" Compute the values of :math:`d_i`, :math:`p_{k,i}`, :math:`s_i` according to the AdSwitch algorithm."""
        Ik = self.find_Ik()
        i_values = np.arange(1, Ik+1, dtype=float)
        di_values = 2 ** (-i_values)
        pi_values = di_values * np.sqrt((self.batch_number + 1) / self.horizon)
        si_values = self.nbArms * np.ceil((self.C2 * np.log(self.horizon)) / di_values**2)
        # print("DEBUG: At time t = {} and batch number k = {}, restarted last time at t0k = {}... Ik = {}, and i = {} and di = {} and pi = {} and si = {} ".format(self.t, self.batch_number, self.last_restart_time, Ik, i_values, di_values, pi_values, si_values))  # DEBUG
        return di_values, pi_values, si_values

    def choice(self):
        """ Choose an arm following the different phase of growing lengths according to the AdSwitch algorithm."""
        # print("For a {} policy: t = {}, current_exploration_arm = {}, current_exploitation_arm = {}, batch_number = {}, length_of_current_phase = {}, step_of_current_phase = {}".format(self, self.t, self.current_exploration_arm, self.current_exploitation_arm, self.batch_number, self.length_of_current_phase, self.step_of_current_phase))  # DEBUG
        # 1) exploration
        # --------------
        if self.phase == Phase.Estimation:
            # beginning of exploration phase
            if self.current_exploration_arm is None:
                self.current_exploration_arm = -1
            # Round-Robin phase
            self.current_exploration_arm = (self.current_exploration_arm + 1) % self.nbArms  # go for next arm
            # Test!
            saw_a_change, sigma = self.statistical_test(self.t, self.last_restart_time)
            if saw_a_change:
                mus = [ mymean(self.read_range_of_rewards(a, sigma, self.t)) for a in range(self.nbArms) ]
                self.current_best_arm = np.argmax(mus)
                self.current_worst_arm = np.argmin(mus)
                self.current_estimated_gap = abs(mus[self.current_best_arm] - mus[self.current_worst_arm])
                self.last_restart_time = self.t
                # change of phase
                self.length_of_current_phase = None  # flag to start the next one
                self.phase = Phase.Exploitation
                self.step_of_current_phase = 0
                self.current_exploration_arm = 0
                # note that this last update might force to sample the arm 0 instead of arm K-1, once in a while...
            return self.current_exploration_arm
        # 2) exploitation
        # ---------------
        elif self.phase == Phase.Exploitation:
            # if in a phase, do it
            if self.length_of_current_phase is not None and self.step_of_current_phase < self.length_of_current_phase:
                # beginning of exploration phase
                if self.current_exploitation_arm is None:
                    self.current_exploitation_arm = -1
                # Round-Robin phase
                if self.current_exploitation_arm >= self.nbArms:
                    self.step_of_current_phase += 1  # explore each arm, ONE more time!
                self.current_exploitation_arm = (self.current_exploitation_arm + 1) % self.nbArms  # go for next arm
            else:
                if self.current_exploitation_arm is None:
                    self.current_exploitation_arm = 0
                # test for a change of size d_i
                compute_new_di_pi_si = self.last_used_di_pi_si is None
                if not compute_new_di_pi_si:
                    di, pi, si = self.last_used_di_pi_si
                    t1 = self.last_restart_time
                    t2 = self.t + 1
                    mus = [ np.mean(self.read_range_of_rewards(a, t1, t2)) for a in range(self.nbArms) ]
                    current_best_mean = np.max(mus)
                    current_worst_mean = np.min(mus)
                    print("Info: the test |mu_a[t1,t2] - mu_b[t1,t2] - Delta| > di/4 is {} for a = best = {}, b = worst = {}, t1 = {} and t2 = {}, Delta = {} and di = {}...".format(abs(current_best_mean - current_worst_mean - self.current_estimated_gap) > di / 4, np.argmax(mus), np.argmin(mus), t1, t2, self.current_estimated_gap, di))  # DEBUG
                    if abs(current_best_mean - current_worst_mean - self.current_estimated_gap) > di / 4:
                        print("Info: the test |mu_a[t1,t2] - mu_b[t1,t2] - Delta| > di/4 was true for a = best = {}, b = worst = {}, t1 = {} and t2 = {}, Delta = {} and di = {}...".format(np.argmax(mus), np.argmin(mus), t1, t2, self.current_estimated_gap, di))  # DEBUG
                        # go back to Estimation phase
                        self.phase = Phase.Estimation
                        self.length_of_current_phase = None  # flag to start the next one
                        self.step_of_current_phase = 0
                        self.current_exploration_arm = 0
                        self.batch_number += 1
                    else:
                        compute_new_di_pi_si = True
                if compute_new_di_pi_si:
                    di_values, pi_values, si_values = self.compute_di_pi_si()
                    proba_of_checking = np.sum(pi_values)
                    assert 0 <= proba_of_checking < 1, "Error: the sum of pi should be < 1 but it is = {}, impossible to do a Step 5 of Exploitation!".format(proba_of_checking)
                    if proba_of_checking > 0:
                        for di, pi, si in zip(di_values, pi_values, si_values):
                            if with_proba(pi):
                                # Start a checking phase!
                                self.last_used_di_pi_si = (di, pi, si)
                                self.length_of_current_phase = si
                                break
                                # ---
                                # DONE OK I understood correctly this sentence, my implementation is correct!
                                # Then for any i from {1, 2,..., Ik} with probability p_k,i, sample both arms alternatingly for si steps
                                # ---
                                # this will make the test sample each arm alternatingly for s_i steps to check for changes of size d_i
                # if no checking is performed at current time step t, then select best arm, and repeat checking phase.
            return self.current_exploitation_arm
        else:
            raise ValueError("Error: AdSwitch should only be in phase Exploration or Checking or Exploitation.")
