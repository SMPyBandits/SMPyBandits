# -*- coding: utf-8 -*-
""" The UCB index policy, modified to take a random permutation order for the initial exploration of each arm (reduce collisions in the multi-players setting).
Reference: [Lai & Robbins, 1985].
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

from .UCB import UCB
import numpy as np


class UCBrandomInit(UCB):
    """ The UCB index policy, modified to take a random permutation order for the initial exploration of each arm (reduce collisions in the multi-players setting).
    Reference: [Lai & Robbins, 1985].
    """

    def __init__(self, nbArms, lower=0., amplitude=1.):
        super(UCBrandomInit, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # XXX trying to randomize the order of the initial visit to each arm; as this determinism breaks its habitility to play efficiently in multi-players games
        # XXX do even more randomized, take a random permutation of the arm
        self._initial_exploration = np.random.permutation(nbArms)
        # The proba that another player has the same is nbPlayers / factorial(nbArms) : should be SMALL !
        # print("One UCB player with _initial_exploration =", self._initial_exploration)  # DEBUG

    def choice(self):
        if self.t < self.nbArms:  # Force to first visit each arm in a certain random order
            return self._initial_exploration[self.t]  # Better: random permutation!
        else:
            return super(UCBrandomInit, self).choice()
