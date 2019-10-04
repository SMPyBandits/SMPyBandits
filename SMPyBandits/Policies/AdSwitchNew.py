# -*- coding: utf-8 -*-
r""" The AdSwitchNew policy for non-stationary bandits, from [["Adaptively Tracking the Best Arm with an Unknown Number of Distribution Changes". Peter Auer, Pratik Gajane and Ronald Ortner, 2019]](http://proceedings.mlr.press/v99/auer19a/auer19a.pdf)

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


class AdSwitchNew(BasePolicy):
    r""" The AdSwitchNew policy for non-stationary bandits, from [["Adaptively Tracking the Best Arm with an Unknown Number of Distribution Changes". Peter Auer, Pratik Gajane and Ronald Ortner, 2019]](http://proceedings.mlr.press/v99/auer19a/auer19a.pdf)
    """

    def __init__(self, nbArms,
            horizon=None, C1=Constant_C1,
            *args, **kwargs
        ):
        if nbArms > 2:
            print("WARNING: so far, for the AdSwitchNew algorithm, only the special case of K=2 arms was explained in the paper, but I generalized it. Maybe it does not work!")  # DEBUG
        super(AdSwitchNew, self).__init__(nbArms, *args, **kwargs)

        # Parameters
        assert horizon is not None, "Error: for a AdSwitchNew policy, the parameter 'horizon' should be > 0 and not None but was = {}".format(horizon)  # DEBUG
        self.horizon = horizon  #: Parameter :math:`T` for the AdSwitchNew algorithm, the horizon of the experiment. TODO try to use :class:`DoublingTrickWrapper` to remove the dependency in :math:`T` ?

        assert C1 > 0, "Error: for a AdSwitchNew policy, the parameter 'C1' should be > 0 but was = {}".format(C1)  # DEBUG
        self.C1 = C1  #: Parameter :math:`C_1` for the AdSwitchNew algorithm.

        # Internal memory
        self.ell = 0  #: Variable :math:`\ell` in the algorithm. Count the number of new episode.
        self.start_of_episode = 0  #: Variable :math:`t_l` in the algorithm. Count the starting time of the current episode.
        self.set_GOOD = set(range(nbArms))  #: Variable :math:`\mathrm{GOOD}_t` in the algorithm. Set of "good" arms at current time.
        self.set_BAD = set()  #: Variable :math:`\mathrm{BAD}_t` in the algorithm. Set of "bad" arms at current time. It always satisfies :math:`\mathrm{BAD}_t = \{1,\dots,K\} \setminus \mathrm{GOOD}_t`.
        self.set_S = set()  #: Variable :math:`S_t` in the algorithm. Set of sampling obligations of arm :math:`a` at current time.
        self.mu_tilde_of_l = np.zeros(nbArms, dtype=float)  #: Vector of variables :math:`\tilde{\mu}_{\ell}(a)` in the algorithm. Count the empirical average of arm :math:`a`.
        self.gap_Delta_tilde_of_l = np.zeros(nbArms, dtype=float)  #: Vector of variables :math:`\tilde{\Delta}_{\ell}(a)` in the algorithm. Count the estimate of the gap of arm :math:`a` against the best of the "good" arms.

        self.all_rewards = [{} for _ in range(self.nbArms)]  #: Memory of all the rewards. A *dictionary* per arm, mapping time to rewards. Growing size until restart of that arm!
        self.history_of_plays = []  #: Memory of all the past actions played!

    def __str__(self):
        return r"AdSwitchNew($T={}$, $C_1={:.3g}$)".format(self.horizon, self.C1)

    def new_episode(self):
        """ Start a new episode, line 3-6 of the algorithm."""
        self.ell += 1
        self.start_of_episode = self.t
        self.set_GOOD = set(range(self.nbArms))
        self.set_BAD = set()

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        super(AdSwitchNew, self).startGame()

        self.ell = 0
        self.start_of_episode = 0
        self.set_GOOD = set(range(self.nbArms))
        self.set_BAD = set()
        self.set_S = {i: set() for i in range(self.nbArms)}
        self.mu_tilde_of_l.fill(0)
        self.gap_Delta_tilde_of_l.fill(0)

        self.all_rewards = [{} for _ in range(self.nbArms)]
        self.history_of_plays = []

    def getReward(self, arm, reward):
        """ Get a reward from an arm."""
        super(AdSwitchNew, self).getReward(arm, reward)
        reward = (reward - self.lower) / self.amplitude
        self.all_rewards[arm][self.t] = reward

        should_start_new_episode = False
        # 3. Check for changes of good arms:
        # TODO this takes a crazy O(K t^3) time, it HAS to be done faster!
        for good_arm in self.set_GOOD:
            for s_1 in range(self.start_of_episode, self.t + 1):  # WARNING we could speed up this loop with their trick
                for s_2 in range(s_1, self.t + 1):  # WARNING we could speed up this loop with their trick
                    for s in range(self.start_of_episode, self.t + 1):  # WARNING we could speed up this loop with their trick
                        # check condition (3)
                        n_s1_s2_a = self.n_s_t(good_arm, s_1, s_2)  # sub interval [s1, s2] <= [s, t] (s <= s1 <= s2 <= t).
                        mu_hat_s1_s2_a = self.mu_hat_s_t(good_arm, s_1, s_2)  # sub interval [s1, s2] <= [s, t] (s <= s1 <= s2 <= t).
                        n_s_t_a = self.n_s_t(good_arm, s, self.t)
                        mu_hat_s_t_a = self.mu_hat_s_t(good_arm, s, self.t)
                        abs_difference_in_s1s2_st = abs(mu_hat_s1_s2_a - mu_hat_s_t_a)
                        confidence_radius_s1s2 = np.sqrt(2 * max(1, np.log(self.horizon)) / max(n_s1_s2_a, 1))
                        confidence_radius_st = np.sqrt(2 * max(1, np.log(self.horizon)) / max(n_s_t_a, 1))
                        right_side = confidence_radius_s1s2 + confidence_radius_st
                        if abs_difference_in_s1s2_st > right_side:  # check condition 3:
                            should_start_new_episode = True

        # 4. Check for changes of bad arms, in O(K t):
        for bad_arm in self.set_BAD:
            for s in range(self.start_of_episode, self.t + 1):  # WARNING we could speed up this loop with their trick
                # check condition (4)
                n_s_t_a = self.n_s_t(bad_arm, s, self.t)
                mu_hat_s_t_a = self.mu_hat_s_t(bad_arm, s, self.t)
                abs_difference_in_st_l = abs(mu_hat_s_t_a - self.mu_tilde_of_l[bad_arm])
                confidence_radius_st = np.sqrt(2 * max(1, np.log(self.horizon)) / max(n_s_t_a, 1))
                gap = self.gap_Delta_tilde_of_l[bad_arm] / 4
                right_side = gap + confidence_radius_st
                if abs_difference_in_st_l > right_side:  # check condition 3:
                    should_start_new_episode = True
            # 5'. Recompute S_t+1
            new_set_Stp1 = set()
            for triplet in self.set_S[bad_arm]:
                _, n, s = triplet
                n_s_t_a = self.n_s_t(bad_arm, s, self.t)
                if n_s_t_a < n:
                    new_set_Stp1.add(triplet)
            self.set_S[bad_arm] = new_set_Stp1

        # 5. Evict arms from GOOD_t
        for good_arm in self.set_GOOD.copy():
            # check condition (1)
            for s in range(self.start_of_episode, self.t + 1):  # WARNING we could speed up this loop with their trick
                mu_hat_s_t_a = self.mu_hat_s_t(good_arm, s, self.t)
                mu_hat_s_t_good = [self.mu_hat_s_t(other_arm, s, self.t) for other_arm in self.set_GOOD]
                mu_hat_s_t_best = max(mu_hat_s_t_good)
                gap_Delta = mu_hat_s_t_best - mu_hat_s_t_a
                gap_to_check = np.sqrt(self.C1 * max(1, np.log(self.horizon)) / max(self.n_s_t(good_arm, s, self.t) - 1, 1))
                if gap_Delta > gap_to_check:  # check condition 1:
                    # ==> evict the arm, it shouldn't be in GOOD any longer!
                    evicted_arm = good_arm
                    self.set_BAD.add(evicted_arm)  # added to the bad arms
                    self.set_GOOD.remove(evicted_arm)  # this arm is now evicted
                    # compute mu_tilde_l(evicted_arm) and delta_tilde_l(evicted_arm) according to (2)
                    self.mu_tilde_of_l[evicted_arm] = mu_hat_s_t_a
                    self.gap_Delta_tilde_of_l[evicted_arm] = gap_Delta
                    self.set_S[evicted_arm] = set()
                    break  # break the inner for loop on s

        # set of new good arms = { all arms } \ { bad arms }
        assert self.set_GOOD == set(range(self.nbArms)) - self.set_BAD  # XXX done iteratively, see above

        if should_start_new_episode:
            self.new_episode()

    def n_s_t(self, arm, s, t):
        r""" Compute :math:`n_{[s,t]}(a) := \#\{\tau : s \leq \tau \leq t, a_{\tau} = a \}`, naively by using the dictionary of all plays :attr:`all_rewards`."""
        assert s <= t, "Error: n_s_t only exists for s <= t, but here s = {} and t = {} for arm a = {}.".format(s, t, arm)  # DEBUG
        all_rewards_of_that_arm = self.all_rewards[arm]
        all_rewards_s_to_t = [r for (tau, r) in all_rewards_of_that_arm.items() if s <= tau <= t]
        return len(all_rewards_s_to_t)

    def mu_hat_s_t(self, arm, s, t):
        r""" Compute :math:`\hat{\tau}_{[s,t]}(a) := \frac{1}{n_{[s,t]}(a)} \sum_{\tau : s \leq \tau \leq t, a_{\tau} = a} r_t`, naively by using the dictionary of all plays :attr:`all_rewards`."""
        assert s <= t, "Error: mu_hat_s_t only exists for s <= t, but here s = {} and t = {} for arm a = {}.".format(s, t, arm)  # DEBUG
        all_rewards_of_that_arm = self.all_rewards[arm]
        all_rewards_s_to_t = [r for (tau, r) in all_rewards_of_that_arm.items() if s <= tau <= t]
        return mymean(all_rewards_s_to_t)

    def find_max_i(self, gap):
        r""" Follow the algorithm and, with a gap estimate :math:`\widehat{\Delta_k}`, find :math:`I_k = \max\{ i : d_i \geq \widehat{\Delta_k} \}`, where :math:`d_i := 2^{-i}`. There is no need to do an exhaustive search:

        .. math:: I_k := \lfloor - \log_2(\widehat{\Delta_k}) \rfloor.
        """
        assert gap >= 0, "Error: cannot find Ik if gap = {} is not > 0.".format(gap)  # DEBUG
        if np.isclose(gap, 0):
            Ik = 1
        else:
            Ik = max(1, int(np.floor(- np.log2(gap))))  # XXX Direct formula!
        # print("DEBUG: at time t = {}, with gap Delta_k = {}, I_k = {}".format(self.t, self.current_estimated_gap, Ik))  # DEBUG
        return Ik

    def choice(self):
        """ Choose an arm following the different phase of growing lengths according to the AdSwitchNew algorithm."""
        # 1. Add checks for bad arms:
        for bad_arm in self.set_BAD:
            gap_Delta_hat_of_l_a = self.gap_Delta_tilde_of_l[bad_arm]
            for i in range(1, self.find_max_i(gap_Delta_hat_of_l_a) + 1):
                # ell, K, T = self.ell, self.nbArms, self.horizon
                probability_to_add_this_triplet = 2**(-i) * np.sqrt(self.ell / (self.nbArms * self.horizon * np.log(self.horizon)))
                if with_proba(probability_to_add_this_triplet):
                    triplet = (2**(-i), np.floor(2**(2*i+1) * np.log(self.horizon)), self.t)
                    self.set_S[bad_arm].add(triplet)
        # 2. Select an arm:
        these_times_taus = [float('+inf') for arm in range(self.nbArms)]
        for arm in self.set_GOOD:
            if self.set_S[arm]:  # not empty set S_t(a)
                look_ahead_in_past = 1
                while look_ahead_in_past < len(self.history_of_plays) and self.history_of_plays[-look_ahead_in_past] != arm:
                    look_ahead_in_past += 1
                these_times_taus[arm] = self.t - look_ahead_in_past
        chosen_arm = np.argmin(these_times_taus)
        self.history_of_plays.append(chosen_arm)
        return chosen_arm
