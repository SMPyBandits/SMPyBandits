# -*- coding: utf-8 -*-
""" MEGA: implementation of the single-player policy from [Concurrent bandits and cognitive radio network, O.Avner & S.Mannor, 2014](https://arxiv.org/abs/1404.5421).

FIXME description
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np

# --- Help function

def epsilon_t(c, d, K, t):
    """ Cf. Algorithm 1 in [Avner & Mannor, 2014](https://arxiv.org/abs/1404.5421)."""
    return min(1, (c * K**2) / (d**2 * (K - 1) * t))


# --- Class MEGA

class MEGA(object):
    """ MEGA: implementation of the single-player policy from [Concurrent bandits and cognitive radio network, O.Avner & S.Mannor, 2014](https://arxiv.org/abs/1404.5421).
    """

    def __init__(self, nbArms, c=None, d=None, p0=0.5, alpha=None, beta=None):  # Named argument to give them in any order
        """
        - nbArms: number of arms.
        - FIXME other parameters

        Example:
        >>> nbArms, XXX = 17, XXX
        >>> player1 = MEGA(nbArms, XXX)

        For multi-players use:
        >>> configuration["players"] = Selfish(NB_PLAYERS, MEGA, nbArms, XXX).childs
        """
        # Store parameters
        self.nbArms = nbArms
        self.c = c
        self.d = d
        assert 0 <= p0 <= 1, "Error: parameter 'p0' for a MEGA player should be in [0, 1]."
        self.p0 = p0
        self.p = p0
        assert 0 <= alpha <= 1, "Error: parameter 'alpha' for a MEGA player should be in [0, 1]."
        self.alpha = alpha
        assert 0 <= beta <= 1, "Error: parameter 'beta' for a MEGA player should be in [0, 1]."
        self.beta = beta
        # Internal memory
        self._tnext = np.ones(nbArms, dtype=int)
        # Implementation details
        self.t = -1

    def __str__(self):
        return "MEGA(p0: {}, alpha: {}, beta: {}, )".format(self.p0, self.alpha, self.beta)

    def startGame(self):
        """ Just reinitialize all the internal memory."""
        self.t = 0
        self.p = self.p0

    def choice(self):
        self.t += 1
        raise ValueError("FIXME MEGA.choice()")

    def getReward(self, arm, reward):
        # print("- A MEGA player receive reward = {} on arm {}, in state {} and time t = {}...".format(reward, arm, self.state, self.t))  # DEBUG
        # If not collision, receive a reward after pulling the arm
        raise ValueError("FIXME MEGA.getReward()")

    def handleCollision(self, arm):
        """ Handle a collision, on arm of index 'arm'.

        - Warning: this method has to be implemented in the collision model, it is NOT implemented in the EvaluatorMultiPlayers.
        """
        # print("- A MEGA player saw a collision on arm {}, in state {}, and time t = {} ...".format(arm, self.state, self.t))  # DEBUG
        raise ValueError("FIXME MEGA.handleCollision()")
