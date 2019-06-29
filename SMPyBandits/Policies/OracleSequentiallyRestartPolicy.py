# -*- coding: utf-8 -*-
r""" An oracle policy for non-stationary bandits, restarting an underlying stationary bandit policy at each breakpoint.

- It runs on top of a simple policy, e.g., :class:`UCB`, and :class:`OracleSequentiallyRestartPolicy` is a wrapper:

    >>> policy = OracleSequentiallyRestartPolicy(nbArms, UCB)
    >>> # use policy as usual, with policy.startGame(), r = policy.choice(), policy.getReward(arm, r)

- It uses the knowledge of the breakpoints to restart the underlying algorithm at each breakpoint.
- It is very simple but impractical: in any real problem it is impossible to know the locations of the breakpoints, but it acts as an efficient baseline.

.. warning:: It is an efficient baseline, but it has no reason to be the best algorithm on a given problem (empirically)! I found that :class:`Policy.DiscountedThompson.DiscountedThompson` is usually the most efficient.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


import numpy as np

try:
    from .BaseWrapperPolicy import BaseWrapperPolicy
except ImportError:
    from BaseWrapperPolicy import BaseWrapperPolicy


#: Should we reset one arm empirical average or all? Default is ``False`` for this algorithm.
PER_ARM_RESTART = False
PER_ARM_RESTART = True

#: Should we fully restart the algorithm or simply reset one arm empirical average? Default is ``False``, it's usually more efficient!
FULL_RESTART_WHEN_REFRESH = True
FULL_RESTART_WHEN_REFRESH = False

#: ``True`` if the algorithm reset one/all arm memories when a change occur on any arm.
#: ``False``` if the algorithms only resets one arm memories when a change occur on *this arm* (needs to know ``listOfMeans``) (default, it should be more efficient).
RESET_FOR_ALL_CHANGE = False

#: ``True`` if the algorithms resets memories of *this arm* no matter if it stays optimal/suboptimal (default, it should be more efficient).
#: ``False`` if the algorithm reset memories only when a change make the previously best arm become suboptimal.
RESET_FOR_SUBOPTIMAL_CHANGE = True


# --- The very generic class

class OracleSequentiallyRestartPolicy(BaseWrapperPolicy):
    r""" An oracle policy for non-stationary bandits, restarting an underlying stationary bandit policy at each breakpoint.
    """
    def __init__(self, nbArms,
            changePoints=None,
            listOfMeans=None,
            reset_for_all_change=RESET_FOR_ALL_CHANGE,
            reset_for_suboptimal_change=RESET_FOR_SUBOPTIMAL_CHANGE,
            full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
            per_arm_restart=PER_ARM_RESTART,
            *args, **kwargs
        ):
        super(OracleSequentiallyRestartPolicy, self).__init__(nbArms, *args, **kwargs)

        if changePoints is None:
            changePoints = []
        changePoints = sorted([tau for tau in changePoints if tau > 0])
        if len(changePoints) == 0:
            print("WARNING: it is useless to use the wrapper OracleSequentiallyRestartPolicy when changePoints = {} is empty, just use the base policy without the wrapper!".format(changePoints))  # DEBUG
        changePoints = [changePoints for _ in range(nbArms)]

        self.reset_for_all_change = reset_for_all_change  #: See :data:`RESET_FOR_ALL_CHANGE`
        self.reset_for_suboptimal_change = reset_for_suboptimal_change  #: See :data:`RESET_FOR_SUBOPTIMAL_CHANGE`
        self.changePoints = self.compute_optimized_changePoints(changePoints=changePoints, listOfMeans=listOfMeans)  #: Locations of the break points (or change points) of the switching bandit problem, for each arm. If ``None``, an empty list is used.

        self._full_restart_when_refresh = full_restart_when_refresh  # Should we fully restart the algorithm or simply reset one arm empirical average ?
        self._per_arm_restart = per_arm_restart  # Should we reset one arm empirical average or all?

        # Internal memory
        self.all_rewards = [[] for _ in range(self.nbArms)]  #: Keep in memory all the rewards obtained since the last restart on that arm.
        self.last_pulls = np.zeros(nbArms, dtype=int)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
        print("Info: creating a new policy {}, with change points = {}...".format(self, changePoints))  # DEBUG

    def compute_optimized_changePoints(self, changePoints=None, listOfMeans=None):
        """ Compute the list of change points for each arm.

        - If :attr:`reset_for_all_change` is ``True``, all change points concern all arms (sub optimal)!
        - If :attr:`reset_for_all_change` is ``False``,
            + If :attr:`reset_for_suboptimal_change` is ``True``, all change points were the mean of an arm change concern it (still sub optimal)!
            + If :attr:`reset_for_suboptimal_change` is ``False``, only the change points were an arm goes from optimal to sub-optimal or sub-optimal to optimal concern it (optimal!)!
        """
        optimized_changePoints = [ [] for _ in range(self.nbArms) ]
        if listOfMeans is None:
            return changePoints
        elif listOfMeans is not None and len(listOfMeans) > 0:
            listOfMeans = np.array(listOfMeans)
            for arm in range(self.nbArms):
                taus = changePoints[arm]
                mus = listOfMeans[:, arm]
                m = 0
                last_mu = mus[m]
                last_best_mu = np.max(listOfMeans[m, :])
                for m, (mu_m, tau_m) in enumerate(zip(mus, taus)):
                    if self.reset_for_all_change:
                        # this breakpoint location concerns all arm, for this option
                        optimized_changePoints[arm].append(tau_m)
                    elif last_mu != mu_m:
                        if self.reset_for_suboptimal_change:
                            # this breakpoint location concerns this arm because its mean changed, for this option
                            optimized_changePoints[arm].append(tau_m)
                        else:
                            best_mu = np.max(listOfMeans[m, :])
                            if (
                                (last_mu == last_best_mu and mu_m < best_mu)  # it's not the best anymore!
                                or (last_mu < last_best_mu and mu_m == best_mu)  # it's now the best!
                                ):
                                # this breakpoint location concerns this arm because its mean changed and it is not the best anymore, for this option
                                optimized_changePoints[arm].append(tau_m)
                            last_best_mu = best_mu
                        last_mu = mu_m
        return optimized_changePoints

    def __str__(self):
        quality = "reset for optimal changes"
        if self.reset_for_all_change: quality = "reset for all changes"
        if self.reset_for_all_change: quality = ""
        sub = not self.reset_for_all_change and not self.reset_for_suboptimal_change
        if sub: quality = "sub-optimal"
        args = "{}{}".format("" if self._per_arm_restart else "global", ", Restart-with-new-Object" if self._full_restart_when_refresh else "")
        args = "{}, {}".format(args, quality) if args else quality
        args = "({})".format(args) if args else ""
        # opt = not self.reset_for_all_change and self.reset_for_suboptimal_change
        return r"Oracle-{}{}".format(self._policy.__name__, args)

    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the whole empirical average if the current time step is in the list of change points.
        """
        super(OracleSequentiallyRestartPolicy, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_pulls[arm] += 1
        # Store it in place for the empirical average of that arm
        self.all_rewards[arm].append(reward)

        if self.detect_change(arm):
            # print("For a player {} a change was detected at time {} for arm {}, because this time step is in its list of change points!".format(self, self.t, arm))  # DEBUG

            if not self._per_arm_restart:
                # or reset current memory for ALL THE arms
                for other_arm in range(self.nbArms):
                    self.last_pulls[other_arm] = 0
                    self.all_rewards[other_arm] = []
            # reset current memory for THIS arm
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
                        if hasattr(self.policy, 'posterior'): self.policy.posterior[other_arm].reset()  # XXX Posterior to reset, for Bayesian policy
                # reset current memory for THIS arm
                self.policy.rewards[arm] = np.sum(self.all_rewards[arm])
                self.policy.pulls[arm] = len(self.all_rewards[arm])
                if hasattr(self.policy, 'posterior'): self.policy.posterior[arm].reset()  # XXX Posterior to reset, for Bayesian policy

        # we update the total number of samples available to the underlying policy
        # self.policy.t = np.sum(self.last_pulls)  # XXX SO NOT SURE HERE

    def detect_change(self, arm):
        """ Try to detect a change in the current arm."""
        return self.t in self.changePoints[arm], None
