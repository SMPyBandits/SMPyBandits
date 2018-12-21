# -*- coding: utf-8 -*-
r""" An experimental policy, using a sliding window (of for instance :math:`\tau=100` *draws* of each arm), and reset the algorithm as soon as the small empirical average is too far away from the long history empirical average (or just restart for one arm, if possible).

- Reference: none yet, idea from Rémi Bonnefoi and Lilian Besson.
- It runs on top of a simple policy, e.g., :class:`UCB`, and :func:`SlidingWindowRestart` is a generic policy using any simple policy with this "sliding window" trick:

    >>> policy = SlidingWindowRestart(nbArms, UCB, tau=100, threshold=0.1)
    >>> # use policy as usual, with policy.startGame(), r = policy.choice(), policy.getReward(arm, r)

- It uses an additional :math:`\mathcal{O}(\tau)` memory but do not cost anything else in terms of time complexity (the average is done with a sliding window, and costs :math:`\mathcal{O}(1)` at every time step).

.. warning:: This is very experimental!
.. warning:: It can only work on basic index policy based on empirical averages (and an exploration bias), like :class:`UCB`, and cannot work on any Bayesian policy (for which we would have to remember all previous observations in order to reset the history with a small history)! Note that it works on :class:`Policies.Thompson.Thompson`.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Rémi Bonnefoi and Lilian Besson"
__version__ = "0.9"


import numpy as np

try:
    from .BaseWrapperPolicy import BaseWrapperPolicy
    from .UCB import UCB as DefaultPolicy, UCB
    from .UCBalpha import UCBalpha, ALPHA
    from .klUCB import klUCB, klucbBern, c
except ImportError:
    from BaseWrapperPolicy import BaseWrapperPolicy
    from UCB import UCB as DefaultPolicy, UCB
    from UCBalpha import UCBalpha, ALPHA
    from klUCB import klUCB, klucbBern, c


#: Size of the sliding window.
TAU = 100

#: Threshold to know when to restart the base algorithm.
THRESHOLD = 0.005

#: Should we fully restart the algorithm or simply reset one arm empirical average ?
FULL_RESTART_WHEN_REFRESH = True


# --- Class

class SlidingWindowRestart(BaseWrapperPolicy):
    r""" An experimental policy, using a sliding window of for instance :math:`\tau=100` draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).
    """

    def __init__(self, nbArms, policy=DefaultPolicy,
            tau=TAU, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
            *args, **kwargs
        ):
        super(SlidingWindowRestart, self).__init__(nbArms, policy=policy, *args, **kwargs)
        # New parameters
        assert 1 <= tau, "Error: parameter 'tau' for class SlidingWindowRestart has to be >= 1, but was {}.".format(tau)  # DEBUG
        self._tau = int(tau)  # Size of the sliding window.
        assert 0 < threshold <= 1, "Error: parameter 'threshold' for class SlidingWindowRestart has to be 0 < threshold <= 1, but was {}.".format(threshold)  # DEBUG
        self._threshold = threshold  # Threshold to know when to restart the base algorithm.
        self._full_restart_when_refresh = full_restart_when_refresh  # Should we fully restart the algorithm or simply reset one arm empirical average ?
        # Internal memory
        self.last_rewards = np.zeros((nbArms, self._tau))  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        self.last_pulls = np.zeros(nbArms, dtype=int)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)

    def __str__(self):
        return r"SW-Restart({}, $\tau={}$, $\varepsilon={:.3g}$)".format(self._policy.__name__, self._tau, self._threshold)

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the whole empirical average if the small average is too far away from it.
        """
        super(SlidingWindowRestart, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_pulls[arm] += 1
        # Store it in place for the empirical average of that arm
        self.last_rewards[arm, self.last_pulls[arm] % self._tau] = reward
        if self.last_pulls[arm] >= self._tau \
            and self.pulls[arm] >= self._tau:
            # Compute the empirical average for that arm
            empirical_average = self.rewards[arm] / self.pulls[arm]
            # And the small empirical average for that arm
            small_empirical_average = np.mean(self.last_rewards[arm])
            if np.abs(empirical_average - small_empirical_average) >= self._threshold:
                # Fully restart the algorithm ?!
                if self._full_restart_when_refresh:
                    self.startGame(createNewPolicy=False)
                # Or simply reset one of the empirical averages?
                else:
                    self.rewards[arm] = np.sum(self.last_rewards[arm])
                    self.pulls[arm] = 1 + (self.last_pulls[arm] % self._tau)


# --- Manually written

class SWR_UCB(UCB):
    r""" An experimental policy, using a sliding window of for instance :math:`\tau=100` draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).

    .. warning:: FIXME I should remove this code, it's useless now that the generic wrapper :class:`SlidingWindowRestart` works fine.
    """

    def __init__(self, nbArms, tau=TAU, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH, *args, **kwargs):
        super(SWR_UCB, self).__init__(nbArms, *args, **kwargs)
        # New parameters
        assert 1 <= tau, "Error: parameter 'tau' for class SWR_UCB has to be >= 1, but was {}.".format(tau)  # DEBUG
        self.tau = int(tau)  #: Size of the sliding window.
        assert 0 < threshold <= 1, "Error: parameter 'threshold' for class SWR_UCB has to be 0 < threshold <= 1, but was {}.".format(threshold)  # DEBUG
        self.threshold = threshold  #: Threshold to know when to restart the base algorithm.
        self.last_rewards = np.zeros((nbArms, tau))  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        self.last_pulls = np.zeros(nbArms, dtype=int)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
        self.full_restart_when_refresh = full_restart_when_refresh  #: Should we fully restart the algorithm or simply reset one arm empirical average ?

    def __str__(self):
        return r"SW-Restart({}, $\tau={}$, $\varepsilon={:.3g}$)".format(super(SWR_UCB, self).__str__(), self.tau, self.threshold)

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the whole empirical average if the small average is too far away from it.
        """
        super(SWR_UCB, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_pulls[arm] += 1
        # Store it in place for the empirical average of that arm
        self.last_rewards[arm, self.last_pulls[arm] % self.tau] = reward
        if self.last_pulls[arm] >= self.tau \
            and self.pulls[arm] >= self.tau:
            # Compute the empirical average for that arm
            empirical_average = self.rewards[arm] / self.pulls[arm]
            # And the small empirical average for that arm
            small_empirical_average = np.mean(self.last_rewards[arm])
            if np.abs(empirical_average - small_empirical_average) >= self.threshold:
                # Fully restart the algorithm ?!
                if self.full_restart_when_refresh:
                    self.startGame()
                # Or simply reset one of the empirical averages?
                else:
                    self.rewards[arm] = np.sum(self.last_rewards[arm])
                    self.pulls[arm] = 1 + (self.last_pulls[arm] % self.tau)


class SWR_UCBalpha(UCBalpha):
    r""" An experimental policy, using a sliding window of for instance :math:`\tau=100` draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).

    .. warning:: FIXME I should remove this code, it's useless now that the generic wrapper :class:`SlidingWindowRestart` works fine.
    """

    def __init__(self, nbArms, tau=TAU, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH, alpha=ALPHA, *args, **kwargs):
        super(SWR_UCBalpha, self).__init__(nbArms, alpha=alpha, *args, **kwargs)
        # New parameters
        assert 1 <= tau, "Error: parameter 'tau' for class SWR_UCBalpha has to be >= 1, but was {}.".format(tau)  # DEBUG
        self.tau = int(tau)  #: Size of the sliding window.
        assert 0 < threshold <= 1, "Error: parameter 'threshold' for class SWR_UCBalpha has to be 0 < threshold <= 1, but was {}.".format(threshold)  # DEBUG
        self.threshold = threshold  #: Threshold to know when to restart the base algorithm.
        self.last_rewards = np.zeros((nbArms, tau))  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        self.last_pulls = np.zeros(nbArms, dtype=int)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
        self.full_restart_when_refresh = full_restart_when_refresh  #: Should we fully restart the algorithm or simply reset one arm empirical average ?

    def __str__(self):
        return r"SW-Restart({}, $\tau={}$, $\varepsilon={:.3g}$)".format(super(SWR_UCBalpha, self).__str__(), self.tau, self.threshold)

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the whole empirical average if the small average is too far away from it.
        """
        super(SWR_UCBalpha, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_pulls[arm] += 1
        # Store it in place for the empirical average of that arm
        self.last_rewards[arm, self.last_pulls[arm] % self.tau] = reward
        if self.last_pulls[arm] >= self.tau \
            and self.pulls[arm] >= self.tau:
            # Compute the empirical average for that arm
            empirical_average = self.rewards[arm] / self.pulls[arm]
            # And the small empirical average for that arm
            small_empirical_average = np.mean(self.last_rewards[arm])
            if np.abs(empirical_average - small_empirical_average) >= self.threshold:
                # Fully restart the algorithm ?!
                if self.full_restart_when_refresh:
                    self.startGame()
                # Or simply reset one of the empirical averages?
                else:
                    self.rewards[arm] = np.sum(self.last_rewards[arm])
                    self.pulls[arm] = 1 + (self.last_pulls[arm] % self.tau)


class SWR_klUCB(klUCB):
    r""" An experimental policy, using a sliding window of for instance :math:`\tau=100` draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).

    .. warning:: FIXME I should remove this code, it's useless now that the generic wrapper :class:`SlidingWindowRestart` works fine.
    """

    def __init__(self, nbArms, tau=TAU, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH, tolerance=1e-4, klucb=klucbBern, c=c, *args, **kwargs):
        super(SWR_klUCB, self).__init__(nbArms, tolerance=tolerance, klucb=klucb, c=c, *args, **kwargs)
        # New parameters
        assert 1 <= tau, "Error: parameter 'tau' for class SWR_klUCB has to be >= 1, but was {}.".format(tau)  # DEBUG
        self.tau = int(tau)  #: Size of the sliding window.
        assert 0 < threshold <= 1, "Error: parameter 'threshold' for class SWR_klUCB has to be 0 < threshold <= 1, but was {}.".format(threshold)  # DEBUG
        self.threshold = threshold  #: Threshold to know when to restart the base algorithm.
        self.last_rewards = np.zeros((nbArms, tau))  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        self.last_pulls = np.zeros(nbArms, dtype=int)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
        self.full_restart_when_refresh = full_restart_when_refresh  #: Should we fully restart the algorithm or simply reset one arm empirical average ?

    def __str__(self):
        return r"SW-Restart({}, $\tau={}$, $\varepsilon={:.3g}$)".format(super(SWR_klUCB, self).__str__(), self.tau, self.threshold)

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the whole empirical average if the small average is too far away from it.
        """
        super(SWR_klUCB, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_pulls[arm] += 1
        # Store it in place for the empirical average of that arm
        self.last_rewards[arm, self.last_pulls[arm] % self.tau] = reward
        if self.last_pulls[arm] >= self.tau \
            and self.pulls[arm] >= self.tau:
            # Compute the empirical average for that arm
            empirical_average = self.rewards[arm] / self.pulls[arm]
            # And the small empirical average for that arm
            small_empirical_average = np.mean(self.last_rewards[arm])
            if np.abs(empirical_average - small_empirical_average) >= self.threshold:
                # Fully restart the algorithm ?!
                if self.full_restart_when_refresh:
                    self.startGame()
                # Or simply reset one of the empirical averages?
                else:
                    self.rewards[arm] = np.sum(self.last_rewards[arm])
                    self.pulls[arm] = 1 + (self.last_pulls[arm] % self.tau)
