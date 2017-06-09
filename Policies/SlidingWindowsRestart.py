# -*- coding: utf-8 -*-
r""" An experimental policy, using a sliding window of for instance 100 draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible).

- Reference: none yet, idea from Rémi Bonnefoi and Lilian Besson
- It runs on top of a simple policy, e.g., :class:`Policy.UCB.UCB`, and :func:`SlidingWindowsRestart` is a *function*, transforming a base policy into a version implementing this "sliding window" trick:

>>> SlidingUCB = SlidingWindowsRestart(UCB, smallHistory=100, threshold=0.1)
>>> policy = SlidingUCB(nbArms)
>>> # use policy as usual, with policy.startGame(), r = policy.choice(), policy.getReward(arm, r)

- It uses an additional :math:`\mathcal{O}(T_0)` memory but do not cost anything else in terms of time complexity (the average is done with a sliding window, and costs :math`\mathcal{O}(1)` at every time step).

.. warning:: This is very experimental!
.. warning:: It can only work on basic index policy based on empirical averages (and an exploration bias), like :class:`Policy.UCB.UCB`, and cannot work on Bayesian policy (for which we would have to remember all previous observations in order to reset the history with a small history) !
"""

__author__ = "Rémi Bonnefoi and Lilian Besson"
__version__ = "0.6"


import numpy as np

from .UCB import UCB as DefaultPolicy
from .UCB import UCB
from .UCBalpha import UCBalpha
# from .UCBalpha import UCBalpha as DefaultPolicy
from .klUCB import klUCB
# from .klUCB import klUCB as DefaultPolicy


#: Size of the sliding window.
SMALLHISTORY = 100

#: Threshold to know when to restart the base algorithm.
THRESHOLD = 1e-2

#: Should we fully restart the algorithm or simply reset one arm empirical average ?
FULL_RESTART_WHEN_REFRESH = False


# --- Main function

def SlidingWindowsRestart(Policy=DefaultPolicy,
                          smallHistory=SMALLHISTORY,
                          threshold=THRESHOLD,
                          full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
                          ):

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
            return r"SlidingWindowsRestart({}, $T_0={}$, $\varepsilon={:.3g}$)".format(super(SlidingWindowsRestart_Policy, self).__str__(), self.smallHistory, self.threshold)

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

SlidingUCB = SlidingWindowsRestart(Policy=UCB, smallHistory=SMALLHISTORY, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH)
SlidingUCBalpha = SlidingWindowsRestart(Policy=UCBalpha, smallHistory=SMALLHISTORY, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH)
SlidingklUCB = SlidingWindowsRestart(Policy=klUCB, smallHistory=SMALLHISTORY, threshold=THRESHOLD, full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH)
