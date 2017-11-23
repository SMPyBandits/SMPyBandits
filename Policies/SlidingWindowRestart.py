# -*- coding: utf-8 -*-
r""" An experimental policy, using a sliding window (of for instance :math:`\tau=100` *draws* of each arm), and reset the algorithm as soon as the small empirical average is too far away from the long history empirical average (or just restart for one arm, if possible).

- Reference: none yet, idea from Rémi Bonnefoi and Lilian Besson.
- It runs on top of a simple policy, e.g., :class:`Policy.UCB.UCB`, and :func:`SlidingWindowRestart` is a function, transforming a base policy into a version implementing this "sliding window" trick:

    >>> SWR_UCB = SlidingWindowRestart(UCB, tau=100, threshold=0.1)
    >>> policy = SWR_UCB(nbArms)
    >>> # use policy as usual, with policy.startGame(), r = policy.choice(), policy.getReward(arm, r)

- It uses an additional :math:`\mathcal{O}(\tau)` memory but do not cost anything else in terms of time complexity (the average is done with a sliding window, and costs :math:`\mathcal{O}(1)` at every time step).

.. warning:: This is very experimental!
.. warning:: It can only work on basic index policy based on empirical averages (and an exploration bias), like :class:`Policy.UCB.UCB`, and cannot work on Bayesian policy (for which we would have to remember all previous observations in order to reset the history with a small history) !
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Rémi Bonnefoi and Lilian Besson"
__version__ = "0.6"


import numpy as np

from .UCB import UCB as DefaultPolicy
from .UCB import UCB
from .UCBalpha import UCBalpha, ALPHA
from .klUCB import klUCB, klucbBern, c


#: Size of the sliding window.
TAU = 300

#: Threshold to know when to restart the base algorithm.
THRESHOLD = 5e-3

#: Should we fully restart the algorithm or simply reset one arm empirical average ?
FULL_RESTART_WHEN_REFRESH = True


# --- Main function

def SlidingWindowRestart(Policy=DefaultPolicy,
                          tau=TAU, threshold=THRESHOLD,
                          full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
                          ):
    """
    Function implementing this algorithm, for a generic base policy ``Policy``.

    .. warning::

        It works fine, but the return class is not a manually defined class, and it is not pickable, so cannot be used with joblib.Parallel::

            AttributeError: Can't pickle local object 'SlidingWindowRestart.<locals>.SlidingWindowsRestart_Policy'
    """

    class SlidingWindowsRestart_Policy(Policy):
        """ An experimental policy, using a sliding window of for instance 100 draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).
        """

        def __init__(self, nbArms, lower=0., amplitude=1., *args, **kwargs):
            super(SlidingWindowsRestart_Policy, self).__init__(nbArms, lower=lower, amplitude=amplitude, *args, **kwargs)
            # New parameters
            assert 1 <= tau, "Error: parameter 'tau' for class SlidingWindowsRestart_Policy has to be >= 1, but was {}.".format(tau)  # DEBUG
            self.tau = int(tau)  #: Size of the sliding window.
            assert 0 < threshold <= 1, "Error: parameter 'threshold' for class SlidingWindowsRestart_Policy has to be 0 < threshold <= 1, but was {}.".format(threshold)  # DEBUG
            self.threshold = threshold  #: Threshold to know when to restart the base algorithm.
            self.last_rewards = np.zeros((nbArms, tau))  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
            self.last_pulls = np.full(nbArms, -1)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
            self.full_restart_when_refresh = full_restart_when_refresh  #: Should we fully restart the algorithm or simply reset one arm empirical average ?

        def __str__(self):
            return r"SlidingWindowRestart({}, $\tau={}$, $\varepsilon={:.3g}$)".format(super(SlidingWindowsRestart_Policy, self).__str__(), self.tau, self.threshold)

        def getReward(self, arm, reward):
            """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

            - Reset the whole empirical average if the small average is too far away from it.
            """
            super(SlidingWindowsRestart_Policy, self).getReward(arm, reward)
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
            # DONE

    return SlidingWindowsRestart_Policy


# --- Some basic ones

# SWR_UCB = SlidingWindowRestart(Policy=UCB)
# SWR_UCBalpha = SlidingWindowRestart(Policy=UCBalpha, tau=TAU, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH)
# SWR_klUCB = SlidingWindowRestart(Policy=klUCB, tau=TAU, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH)


# --- Manually written

class SWR_UCB(UCB):
    """ An experimental policy, using a sliding window of for instance 100 draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).
    """

    def __init__(self, nbArms, tau=TAU, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH, lower=0., amplitude=1., *args, **kwargs):
        super(SWR_UCB, self).__init__(nbArms, lower=lower, amplitude=amplitude, *args, **kwargs)
        # New parameters
        assert 1 <= tau, "Error: parameter 'tau' for class SWR_UCB has to be >= 1, but was {}.".format(tau)  # DEBUG
        self.tau = int(tau)  #: Size of the sliding window.
        assert 0 < threshold <= 1, "Error: parameter 'threshold' for class SWR_UCB has to be 0 < threshold <= 1, but was {}.".format(threshold)  # DEBUG
        self.threshold = threshold  #: Threshold to know when to restart the base algorithm.
        self.last_rewards = np.zeros((nbArms, tau))  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        self.last_pulls = np.full(nbArms, -1)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
        self.full_restart_when_refresh = full_restart_when_refresh  #: Should we fully restart the algorithm or simply reset one arm empirical average ?

    def __str__(self):
        return r"SlidingWindowRestart({}, $\tau={}$, $\varepsilon={:.3g}$)".format(super(SWR_UCB, self).__str__(), self.tau, self.threshold)

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
        # DONE


class SWR_UCBalpha(UCBalpha):
    """ An experimental policy, using a sliding window of for instance 100 draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).
    """

    def __init__(self, nbArms, tau=TAU, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH, alpha=ALPHA, lower=0., amplitude=1., *args, **kwargs):
        super(SWR_UCBalpha, self).__init__(nbArms, alpha=alpha, lower=lower, amplitude=amplitude, *args, **kwargs)
        # New parameters
        assert 1 <= tau, "Error: parameter 'tau' for class SWR_UCBalpha has to be >= 1, but was {}.".format(tau)  # DEBUG
        self.tau = int(tau)  #: Size of the sliding window.
        assert 0 < threshold <= 1, "Error: parameter 'threshold' for class SWR_UCBalpha has to be 0 < threshold <= 1, but was {}.".format(threshold)  # DEBUG
        self.threshold = threshold  #: Threshold to know when to restart the base algorithm.
        self.last_rewards = np.zeros((nbArms, tau))  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        self.last_pulls = np.full(nbArms, -1)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
        self.full_restart_when_refresh = full_restart_when_refresh  #: Should we fully restart the algorithm or simply reset one arm empirical average ?

    def __str__(self):
        return r"SlidingWindowRestart({}, $\tau={}$, $\varepsilon={:.3g}$)".format(super(SWR_UCBalpha, self).__str__(), self.tau, self.threshold)

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
        # DONE


class SWR_klUCB(klUCB):
    """ An experimental policy, using a sliding window of for instance 100 draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).
    """

    def __init__(self, nbArms, tau=TAU, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH, tolerance=1e-4, klucb=klucbBern, c=c, lower=0., amplitude=1., *args, **kwargs):
        super(SWR_klUCB, self).__init__(nbArms, tolerance=tolerance, klucb=klucb, c=c, lower=lower, amplitude=amplitude, *args, **kwargs)
        # New parameters
        assert 1 <= tau, "Error: parameter 'tau' for class SWR_klUCB has to be >= 1, but was {}.".format(tau)  # DEBUG
        self.tau = int(tau)  #: Size of the sliding window.
        assert 0 < threshold <= 1, "Error: parameter 'threshold' for class SWR_klUCB has to be 0 < threshold <= 1, but was {}.".format(threshold)  # DEBUG
        self.threshold = threshold  #: Threshold to know when to restart the base algorithm.
        self.last_rewards = np.zeros((nbArms, tau))  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        self.last_pulls = np.full(nbArms, -1)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
        self.full_restart_when_refresh = full_restart_when_refresh  #: Should we fully restart the algorithm or simply reset one arm empirical average ?

    def __str__(self):
        return r"SlidingWindowRestart({}, $\tau={}$, $\varepsilon={:.3g}$)".format(super(SWR_klUCB, self).__str__(), self.tau, self.threshold)

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
        # DONE
