# -*- coding: utf-8 -*-
""" Base class for any policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""

__author__ = "Lilian Besson"
__version__ = "0.3"

import numpy as np


class BasePolicy(object):
    """ Base class for any policy."""

    def __init__(self, nbArms, lower=0., amplitude=1.):
        # Parameters
        assert nbArms > 0, "Error: the 'nbArms' parameter of a {} object cannot be <= 0.".format(self)
        self.nbArms = nbArms
        self.lower = lower
        assert amplitude > 0, "Error: the 'amplitude' parameter of a {} object cannot be <= 0.".format(self)
        self.amplitude = amplitude
        # Internal memory
        self.t = -1  # special value
        self.pulls = np.zeros(nbArms, dtype=int)
        self.rewards = np.zeros(nbArms)

    def __str__(self):
        return self.__class__.__name__

    # --- Start game, and receive rewards

    def startGame(self):
        self.t = 0
        self.pulls.fill(0)
        self.rewards.fill(0)

    def getReward(self, arm, reward, checkBounds=False):
        self.t += 1
        self.pulls[arm] += 1
        # XXX we could check here if the reward is outside the bounds
        if checkBounds:
            if not 0 <= reward - self.lower <= self.amplitude:
                print("[Warning] {} received on arm {} a reward = {} that is outside the interval [{}, {}] : the policy will probably fail to work correctly...".format(self, arm, reward, self.lower, self.lower + self.amplitude))
        reward = (reward - self.lower) / self.amplitude
        self.rewards[arm] += reward

    # --- Basic choice() method

    def choice(self):
        raise NotImplementedError("This method choice() has to be implemented in the child class inheriting from BasePolicy.")

    # --- Others choice...() methods, partly implemented

    def choiceWithRank(self, rank=1):
        if rank == 1:
            return self.choice()
        else:
            raise NotImplementedError("This method choiceWithRank(rank) has to be implemented in the child class inheriting from BasePolicy.")

    def choiceFromSubSet(self, availableArms='all'):
        if availableArms == 'all':
            return self.choice()
        else:
            raise NotImplementedError("This method choiceFromSubSet(availableArms) has to be implemented in the child class inheriting from BasePolicy.")

    def choiceMultiple(self, nb=1):
        if nb == 1:
            return self.choice()
        else:
            raise NotImplementedError("This method choiceMultiple(nb) has to be implemented in the child class inheriting from BasePolicy.")
