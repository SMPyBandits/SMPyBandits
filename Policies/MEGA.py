# -*- coding: utf-8 -*-
""" MEGA: implementation of the single-player policy from [Concurrent bandits and cognitive radio network, O.Avner & S.Mannor, 2014](https://arxiv.org/abs/1404.5421).

The Multi-user epsilon-Greedy collision Avoiding (MEGA) algorithm is based on the epsilon-greedy algorithm introduced in [2], augmented by a collision avoidance mechanism that is inspired by the classical ALOHA protocol.

- [2]: Finite-time analysis of the multi-armed bandit problem, P.Auer & N.Cesa-Bianchi & P.Fischer, 2002
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from random import random
import numpy as np
import numpy.random as rn
from .BasePolicy import BasePolicy


# --- Class MEGA

class MEGA(BasePolicy):
    """ MEGA: implementation of the single-player policy from [Concurrent bandits and cognitive radio network, O.Avner & S.Mannor, 2014](https://arxiv.org/abs/1404.5421).
    """

    def __init__(self, nbArms, p0=0.5, alpha=0.5, beta=0.5, c=0.1, d=0.01, lower=0., amplitude=1.):  # Named argument to give them in any order
        """
        - nbArms: number of arms.
        - p0: initial probability p(0); p(t) is the probability of persistance on the chosenArm at time t
        - alpha: scaling in the update for p(t+1) <- alpha p(t) + (1 - alpha(t))
        - beta: exponent used in the interval [t, t + t^beta], from where to sample a random time t_next(k), until when the chosenArm is unavailable
        - c, d: used to compute the exploration probability epsilon_t, cf the function :func:`_epsilon_t`.

        Example:

        >>> nbArms, p0, alpha, beta, c, d = 17, 0.5, 0.5, 0.5, 0.1, 0.01
        >>> player1 = MEGA(nbArms, p0, alpha, beta, c, d)

        For multi-players use:

        >>> configuration["players"] = Selfish(NB_PLAYERS, MEGA, nbArms, p0, alpha, beta, c, d).children
        """
        # Store parameters
        super(MEGA, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.c = c  #: Parameter c
        # FIXME we should not require two parameters, as they are just used in the formula for epsilon_t
        self.d = d  #: Parameter d
        assert 0 <= p0 <= 1, "Error: parameter 'p0' for a MEGA player should be in [0, 1]."  # DEBUG
        self.p0 = p0  #: Parameter p0, should not be modified
        self.p = p0  #: Parameter p, can be modified
        assert 0 < alpha <= 1, "Error: parameter 'alpha' for a MEGA player should be in (0, 1]."  # DEBUG
        self.alpha = alpha  #: Parameter alpha
        assert 0 < beta <= 1, "Error: parameter 'beta' for a MEGA player should be in (0, 1]."  # DEBUG
        self.beta = beta  #: Parameter beta
        # Internal memory
        self.chosenArm = None  #: Last chosen arm
        self.tnext = np.ones(nbArms, dtype=int)  #: Only store the delta time
        self.meanRewards = np.zeros(nbArms)  #: Mean rewards

    def __str__(self):
        return r"MEGA($c={:.3g}$, $d={:.3g}$, $p_0={:.3g}$, $\alpha={:.3g}$, $\beta={:.3g}$)".format(self.c, self.d, self.p0, self.alpha, self.beta)

    def startGame(self):
        """ Just reinitialize all the internal memory."""
        super(MEGA, self).startGame()
        self.p = self.p0
        self.chosenArm = rn.randint(self.nbArms)  # Start on a random arm
        self.tnext.fill(1)
        self.meanRewards.fill(float('-inf'))  # Null reward if not pulled

    def choice(self):
        """ Choose an arm, as described by the MEGA algorithm."""
        self.t += 1
        if self.chosenArm is not None:  # We can still exploit that arm
            return self.chosenArm
        else:  # We have to chose a new arm
            # Identify available arms
            availableArms = np.nonzero(self.tnext <= self.t)[0]
            if len(availableArms) == 0:
                print("Error: MEGA.choice() should 'Refrain from transmitting in this round' but my model does not allow this - YET ... Choosing a random arm.")  # DEBUG
                self.chosenArm = rn.randint(self.nbArms)  # XXX Choose a random arm
                # raise ValueError("FIXME MEGA.choice() should 'Refrain from transmitting in this round' but my model does not allow this - YET")
            else:  # There is some available arms
                epsilon = self._epsilon_t()
                if random() < epsilon:  # With proba epsilon_t
                    newArm = rn.choice(availableArms)  # Explore valid arms
                    if self.chosenArm != newArm:
                        self.p = self.p0  # Reinitialize proba p
                else:  # Exploit: select the arm with highest meanRewards
                    self.meanRewards[self.pulls != 0] = self.rewards[self.pulls != 0] / self.pulls[self.pulls != 0]
                    # newArm = np.argmax(self.meanRewards)
                    # Uniformly chosen if more than one arm has the highest index, but that's unlikely
                    newArm = np.random.choice(np.nonzero(self.meanRewards == np.max(self.meanRewards))[0])
                self.chosenArm = newArm
            return self.chosenArm

    def getReward(self, arm, reward):
        """ Receive a reward on arm of index 'arm', as described by the MEGA algorithm.

        - If not collision, receive a reward after pulling the arm.
        """
        assert self.chosenArm == arm, "Error: a MEGA player can only get a reward on her chosenArm. Here, arm = {} != chosenArm = {} ...".format(arm, self.chosenArm)  # DEBUG
        # print("- A MEGA player receive reward = {} on arm {}, and time t = {}...".format(reward, arm, self.t))  # DEBUG
        self.rewards[arm] += (reward - self.lower) / self.amplitude
        self.pulls[arm] += 1
        self.p = self.p * self.alpha + (1 - self.alpha)  # Update proba p

    def handleCollision(self, arm, reward=None):
        """ Handle a collision, on arm of index 'arm'.

        - Warning: this method has to be implemented in the collision model, it is NOT implemented in the EvaluatorMultiPlayers.

        .. note:: We do not care on which arm the collision occured.

        """
        assert self.chosenArm == arm, "Error: a MEGA player can only see a collision on her chosenArm. Here, arm = {} != chosenArm = {} ...".format(arm, self.chosenArm)  # DEBUG
        # print("- A MEGA player saw a collision on arm {}, and time t = {} ...".format(arm, self.t))  # DEBUG
        # 1. With proba p, persist
        # if random() < self.p:
        #     self.chosenArm = self.chosenArm
        # 2. With proba 1 - p, give up
        if random() >= self.p:
            # Random time offset until when this arm self.chosenArm is not sampled
            delta_tnext_k = rn.randint(low=0, high=1 + int(self.t**self.beta))
            self.tnext[self.chosenArm] = self.t + delta_tnext_k
            # Reinitialize the proba p
            self.p = self.p0
            # We give up this arm
            self.chosenArm = None

    # --- Internal methods

    def _epsilon_t(self):
        """Compute the value of decreasing epsilon(t), cf. Algorithm 1 in [Avner & Mannor, 2014](https://arxiv.org/abs/1404.5421)."""
        return min(1, (self.c * self.nbArms**2) / (self.d**2 * (self.nbArms - 1) * self.t))
        # assert 0 <= epsilon <= 1, "Error, epsilon_t({}, {}, {}, {}) computed an epsilon = {} which is NOT in [0, 1] ...".format(c, d, nbArms, t, epsilon)  # DEBUG
