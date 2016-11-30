# -*- coding: utf-8 -*-
""" MusicalChair: implementation of the single-player policy from https://arxiv.org/abs/1512.02866.

- FIXME write documentation
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np


# Possible states of the player
STATE_0__NotStarted = 0
STATE_1__InitialPhase = 1
STATE_2__MusicalChair = 2
STATES = [STATE_0__NotStarted, STATE_1__InitialPhase, STATE_2__MusicalChair]


class MusicalChair(object):
    """ MusicalChair: implementation of the single-player policy from https://arxiv.org/abs/1512.02866.
    """

    def __init__(self, nbArms, T0=0.25, T1=None, nbPlayers=None):  # Named argument to give them in any order
        """
        - nbArms: number of arms.
        - nbPlayers: number of players to create (in self._players). Warning: each child player should NOT use this knowledge!

        Example:
        >>> nbArms, T0, T1, nbPlayers = 17, 0.1, 10000, 6
        >>> player1 = MusicalChair(nbArms, T0, T1, nbPlayers)

        For multi-players use:
        >>> configuration["players"] = Selfish(NB_PLAYERS, MusicalChair, nbArms, T0=0.25, T1=HORIZON, nbPlayers=NB_PLAYERS).childs
        """
        self.state = STATE_0__NotStarted
        if 0 < T0 < 1:  # T0 is a fraction of the horizon T1
            T0 = int(T0 * T1)  # Lower bound
        assert T0 < T1, "Error, T0 should be < than T1 for MusicalChair class."
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for MusicalChair class has to be > 0."
        # Store parameters
        self.nbArms = nbArms
        self.T0 = T0
        self.T1 = T1
        self.nbPlayers = nbPlayers
        # Internal memory
        self._chair = None  # Not sited yet
        self._cumulatedRewards = np.zeros(nbArms)  # That's the s_i(t) of the paper
        self._empiricalMeans = np.zeros(nbArms)    # TODO use only one array for both, as they are never used as the same time!
        self._nbObservations = np.zeros(nbArms, dtype=int)  # That's the o_i of the paper
        self._A = np.zeros(nbArms, dtype=int)
        self._nbCollision = 0  # That's the C_T0 of the paper
        # Implementation details for the common API
        self.params = ''
        self.t = -1

    def __str__(self):
        self.params = 'N*: {}'.format(self.nbPlayers)  # Update current estimate
        return "MusicalChair({})".format(self.params)

    def startGame(self):
        """ Just reinitialize all the internal memory, and decide how to start (state 1 or 2)."""
        self.t = 0
        self._chair = None  # Not sited yet
        self._cumulatedRewards.fill(0)
        self._empiricalMeans.fill(0)
        self._nbObservations.fill(0)
        self._A.fill(0)
        self._nbCollision = 0
        # if nbPlayers is None, start by estimating it to N*, with the initial phase procedure
        if self.nbPlayers is None:
            self.state = STATE_1__InitialPhase
        else:  # No need for an initial phase if nbPlayers is known (given)
            self.T0 = 0
            self.state = STATE_2__MusicalChair

    def choice(self):
        if self._chair is not None:  # If the player is already sit, nothing to do
            return self._chair
        elif self.state == STATE_1__InitialPhase:
            # Play as initial phase: chose a random arm, uniformly among all the K arms
            i = np.random.randint(self.nbArms)
            return i
        elif self.state == STATE_2__MusicalChair:
            # Play as musical chair: chose a random arm, among the M bests
            k = np.random.randint(self.nbPlayers)
            i = self._A[k]  # Random arm among the M bests
            self._chair = i  # Assume that it would be a good chair
            return i
        else:  # XXX remove this
            raise ValueError("MusicalChair.choice() should never be in this case. Fix this code, quickly!")

    def getReward(self, arm, reward):
        # if not collision, receive a reward after pulling the arm
        if self.state == STATE_1__InitialPhase:
            # Count the observation, update arm cumulated reward
            self._nbObservations[arm] += 1      # One observation of this arm
            self._cumulatedRewards[arm] += reward  # More reward
        elif self.state == STATE_2__MusicalChair:
            pass  # Nothing to do in this second phase
        # Maybe we are do with the initial phase?
        if self.t >= self.TO and self.state == STATE_1__InitialPhase:
            self.state = STATE_2__MusicalChair
            # First, we compute the empirical means mu_i
            self._empiricalMeans = self._cumulatedRewards / self._nbObservations
            # Then, sort their index by empirical means
            self._A = np.argsort(self._empiricalMeans)
            # Finally, we compute the final estimate of N* = nbPlayers
            if self._nbCollision == self.T0:  # 1st case, we only saw collisions!
                self.nbPlayers = self.nbArms  # Worst case, pessimist estimate of the nb of players
            else:  # 2nd case, we didn't see only collisions
                self.nbPlayers = round(1 + np.log((self.TO - self._nbCollision) / self.TO) / np.log(1. - 1. / self.nbArms))
        # Finish, next step
        self.t += 1

    def handleCollision(self, t, arm):
        if self.state == STATE_1__InitialPhase:
            # count one more collision in this initial phase (no matter the arm)
            self._nbCollision += 1
        elif self.state == STATE_2__MusicalChair:
            self._chair = None  # Cannot stay sitted here
            # FIXME should be different!
