# -*- coding: utf-8 -*-
r""" An experimental policy, using a sliding window (of for instance :math:`T_0=100` *draws* of each arm), and reset the algorithm as soon as the small empirical average is too far away from the long history empirical average (or just restart for one arm, if possible).

- Reference: none yet, idea from Rémi Bonnefoi and Lilian Besson
- It runs on top of a simple policy, e.g., :class:`Policy.UCB.UCB`, and :func:`SlidingWindowRestart` is a *function*, transforming a base policy into a version implementing this "sliding window" trick:

>>> SWUCB = SlidingWindowRestart(UCB, smallHistory=100, threshold=0.1)
>>> policy = SWUCB(nbArms)
>>> # use policy as usual, with policy.startGame(), r = policy.choice(), policy.getReward(arm, r)

- It uses an additional :math:`\mathcal{O}(T_0)` memory but do not cost anything else in terms of time complexity (the average is done with a sliding window, and costs :math:`\mathcal{O}(1)` at every time step).

.. warning:: This is very experimental!
.. warning:: It can only work on basic index policy based on empirical averages (and an exploration bias), like :class:`Policy.UCB.UCB`, and cannot work on Bayesian policy (for which we would have to remember all previous observations in order to reset the history with a small history) !
"""

__author__ = "Rémi Bonnefoi and Lilian Besson"
__version__ = "0.6"


import numpy as np

from .UCB import UCB as DefaultPolicy
from .UCB import UCB
from .UCBalpha import UCBalpha, ALPHA
# from .UCBalpha import UCBalpha as DefaultPolicy
from .klUCB import klUCB, klucbBern, c
# from .klUCB import klUCB as DefaultPolicy


#: Size of the sliding window.
SMALLHISTORY = 300

#: Threshold to know when to restart the base algorithm.
THRESHOLD = 5e-3

#: Should we fully restart the algorithm or simply reset one arm empirical average ?
FULL_RESTART_WHEN_REFRESH = True


# --- Main function

def SlidingWindowRestart(Policy=DefaultPolicy,
                          smallHistory=SMALLHISTORY,
                          threshold=THRESHOLD,
                          full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
                          ):
    """
    Function implementing this algorithm, for a generic base policy ``Policy``.

    .. warning:: it works fine, but the return class is not a manually defined class, and it is not pickable, so cannot be used with joblib.Parallel::

        AttributeError: Can't pickle local object 'SlidingWindowRestart.<locals>.SlidingWindowsRestart_Policy'
    """

    class SlidingWindowsRestart_Policy(Policy):
        """ An experimental policy, using a sliding window of for instance 100 draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).
        """

        def __init__(self, nbArms, lower=0., amplitude=1., *args, **kwargs):
            super(SlidingWindowsRestart_Policy, self).__init__(nbArms, lower=lower, amplitude=amplitude, *args, **kwargs)
            # New parameters
            assert 1 <= smallHistory, "Error: parameter 'smallHistory' for class SlidingWindowsRestart_Policy has to be >= 1, but was {}.".format(smallHistory)  # DEBUG
            self.smallHistory = int(smallHistory)  #: Size of the sliding window.
            assert 0 < threshold <= 1, "Error: parameter 'threshold' for class SlidingWindowsRestart_Policy has to be 0 < threshold <= 1, but was {}.".format(threshold)  # DEBUG
            self.threshold = threshold  #: Threshold to know when to restart the base algorithm.
            self.small_rewards = np.zeros((nbArms, smallHistory))  #: Keep in memory all the rewards obtained in the last :math:`T_0` steps.
            self.last_selections = np.full(nbArms, -1)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
            self.full_restart_when_refresh = full_restart_when_refresh  #: Should we fully restart the algorithm or simply reset one arm empirical average ?

        def __str__(self):
            return r"SlidingWindowRestart({}, $T_0={}$, $\varepsilon={:.3g}$)".format(super(SlidingWindowsRestart_Policy, self).__str__(), self.smallHistory, self.threshold)

        def getReward(self, arm, reward):
            """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

            - Reset the empirical average
            """
            super(SlidingWindowsRestart_Policy, self).getReward(arm, reward)
            # Get reward
            reward = (reward - self.lower) / self.amplitude
            # We seen it one more time
            self.last_selections[arm] += 1
            # Store it in place for the empirical average of that arm
            self.small_rewards[arm, self.last_selections[arm] % self.smallHistory] = reward
            if self.last_selections[arm] >= self.smallHistory \
                     and self.pulls[arm] >= self.smallHistory:
                # Compute the empirical average for that arm
                empirical_average = self.rewards[arm] / self.pulls[arm]
                # And the small empirical average for that arm
                small_empirical_average = np.mean(self.small_rewards[arm])
                if np.abs(empirical_average - small_empirical_average) >= self.threshold:
                    # Fully restart the algorithm ?!
                    if self.full_restart_when_refresh:
                        self.startGame()
                    # Or simply reset one of the empirical averages?
                    else:
                        self.rewards[arm] = np.sum(self.small_rewards[arm])
                        self.pulls[arm] = self.last_selections
            # DONE

    return SlidingWindowsRestart_Policy


# --- Some basic ones

# SWUCB = SlidingWindowRestart(Policy=UCB)
# SWUCBalpha = SlidingWindowRestart(Policy=UCBalpha, smallHistory=SMALLHISTORY, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH)
# SWklUCB = SlidingWindowRestart(Policy=klUCB, smallHistory=SMALLHISTORY, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH)


# --- Manually written

class SWUCB(UCB):
    """ An experimental policy, using a sliding window of for instance 100 draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).
    """

    def __init__(self, nbArms, smallHistory=SMALLHISTORY, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH, lower=0., amplitude=1., *args, **kwargs):
        super(SWUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude, *args, **kwargs)
        # New parameters
        assert 1 <= smallHistory, "Error: parameter 'smallHistory' for class SWUCB has to be >= 1, but was {}.".format(smallHistory)  # DEBUG
        self.smallHistory = int(smallHistory)  #: Size of the sliding window.
        assert 0 < threshold <= 1, "Error: parameter 'threshold' for class SWUCB has to be 0 < threshold <= 1, but was {}.".format(threshold)  # DEBUG
        self.threshold = threshold  #: Threshold to know when to restart the base algorithm.
        self.small_rewards = np.zeros((nbArms, smallHistory))  #: Keep in memory all the rewards obtained in the last :math:`T_0` steps.
        self.last_selections = np.full(nbArms, -1)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
        self.full_restart_when_refresh = full_restart_when_refresh  #: Should we fully restart the algorithm or simply reset one arm empirical average ?

    def __str__(self):
        return r"SlidingWindowRestart({}, $T_0={}$, $\varepsilon={:.3g}$)".format(super(SWUCB, self).__str__(), self.smallHistory, self.threshold)

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the empirical average
        """
        super(SWUCB, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_selections[arm] += 1
        # Store it in place for the empirical average of that arm
        self.small_rewards[arm, self.last_selections[arm] % self.smallHistory] = reward
        if self.last_selections[arm] >= self.smallHistory \
                 and self.pulls[arm] >= self.smallHistory:
            # Compute the empirical average for that arm
            empirical_average = self.rewards[arm] / self.pulls[arm]
            # And the small empirical average for that arm
            small_empirical_average = np.mean(self.small_rewards[arm])
            if np.abs(empirical_average - small_empirical_average) >= self.threshold:
                # Fully restart the algorithm ?!
                if self.full_restart_when_refresh:
                    self.startGame()
                # Or simply reset one of the empirical averages?
                else:
                    self.rewards[arm] = np.sum(self.small_rewards[arm])
                    self.pulls[arm] = self.last_selections
        # DONE


class SWUCBalpha(UCBalpha):
    """ An experimental policy, using a sliding window of for instance 100 draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).
    """

    def __init__(self, nbArms, smallHistory=SMALLHISTORY, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH, alpha=ALPHA, lower=0., amplitude=1., *args, **kwargs):
        super(SWUCBalpha, self).__init__(nbArms, alpha=alpha, lower=lower, amplitude=amplitude, *args, **kwargs)
        # New parameters
        assert 1 <= smallHistory, "Error: parameter 'smallHistory' for class SWUCBalpha has to be >= 1, but was {}.".format(smallHistory)  # DEBUG
        self.smallHistory = int(smallHistory)  #: Size of the sliding window.
        assert 0 < threshold <= 1, "Error: parameter 'threshold' for class SWUCBalpha has to be 0 < threshold <= 1, but was {}.".format(threshold)  # DEBUG
        self.threshold = threshold  #: Threshold to know when to restart the base algorithm.
        self.small_rewards = np.zeros((nbArms, smallHistory))  #: Keep in memory all the rewards obtained in the last :math:`T_0` steps.
        self.last_selections = np.full(nbArms, -1)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
        self.full_restart_when_refresh = full_restart_when_refresh  #: Should we fully restart the algorithm or simply reset one arm empirical average ?

    def __str__(self):
        return r"SlidingWindowRestart({}, $T_0={}$, $\varepsilon={:.3g}$)".format(super(SWUCBalpha, self).__str__(), self.smallHistory, self.threshold)

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the empirical average
        """
        super(SWUCBalpha, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_selections[arm] += 1
        # Store it in place for the empirical average of that arm
        self.small_rewards[arm, self.last_selections[arm] % self.smallHistory] = reward
        if self.last_selections[arm] >= self.smallHistory \
                 and self.pulls[arm] >= self.smallHistory:
            # Compute the empirical average for that arm
            empirical_average = self.rewards[arm] / self.pulls[arm]
            # And the small empirical average for that arm
            small_empirical_average = np.mean(self.small_rewards[arm])
            if np.abs(empirical_average - small_empirical_average) >= self.threshold:
                # Fully restart the algorithm ?!
                if self.full_restart_when_refresh:
                    self.startGame()
                # Or simply reset one of the empirical averages?
                else:
                    self.rewards[arm] = np.sum(self.small_rewards[arm])
                    self.pulls[arm] = self.last_selections
        # DONE


class SWklUCB(klUCB):
    """ An experimental policy, using a sliding window of for instance 100 draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).
    """

    def __init__(self, nbArms, smallHistory=SMALLHISTORY, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH, tolerance=1e-4, klucb=klucbBern, c=c, lower=0., amplitude=1., *args, **kwargs):
        super(SWklUCB, self).__init__(nbArms, tolerance=tolerance, klucb=klucb, c=c, lower=lower, amplitude=amplitude, *args, **kwargs)
        # New parameters
        assert 1 <= smallHistory, "Error: parameter 'smallHistory' for class SWklUCB has to be >= 1, but was {}.".format(smallHistory)  # DEBUG
        self.smallHistory = int(smallHistory)  #: Size of the sliding window.
        assert 0 < threshold <= 1, "Error: parameter 'threshold' for class SWklUCB has to be 0 < threshold <= 1, but was {}.".format(threshold)  # DEBUG
        self.threshold = threshold  #: Threshold to know when to restart the base algorithm.
        self.small_rewards = np.zeros((nbArms, smallHistory))  #: Keep in memory all the rewards obtained in the last :math:`T_0` steps.
        self.last_selections = np.full(nbArms, -1)  #: Keep in memory the times where each arm was last seen. Start with -1 (never seen)
        self.full_restart_when_refresh = full_restart_when_refresh  #: Should we fully restart the algorithm or simply reset one arm empirical average ?

    def __str__(self):
        return r"SlidingWindowRestart({}, $T_0={}$, $\varepsilon={:.3g}$)".format(super(SWklUCB, self).__str__(), self.smallHistory, self.threshold)

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the empirical average
        """
        super(SWklUCB, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_selections[arm] += 1
        # Store it in place for the empirical average of that arm
        self.small_rewards[arm, self.last_selections[arm] % self.smallHistory] = reward
        if self.last_selections[arm] >= self.smallHistory \
                 and self.pulls[arm] >= self.smallHistory:
            # Compute the empirical average for that arm
            empirical_average = self.rewards[arm] / self.pulls[arm]
            # And the small empirical average for that arm
            small_empirical_average = np.mean(self.small_rewards[arm])
            if np.abs(empirical_average - small_empirical_average) >= self.threshold:
                # Fully restart the algorithm ?!
                if self.full_restart_when_refresh:
                    self.startGame()
                # Or simply reset one of the empirical averages?
                else:
                    self.rewards[arm] = np.sum(self.small_rewards[arm])
                    self.pulls[arm] = self.last_selections
        # DONE
