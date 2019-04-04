# -*- coding: utf-8 -*-
r""" MusicalChairNoSensing: implementation of the decentralized multi-player policy from [["Multiplayer bandits without observing collision information", by Gabor Lugosi and Abbas Mehrabian]](https://arxiv.org/abs/1808.08416).

.. note:: The algorithm implemented here is Algorithm 1 (page 8) in the article, but the authors did not named it. I will refer to it as the Musical Chair algorithm with no sensing, or :class:`MusicalChairNoSensing` in the code.
"""
from __future__ import division, print_function  # Python 2 compatibility, division

__author__ = "Lilian Besson"
__version__ = "0.9"

from enum import Enum  # For the different states
import numpy as np
from scipy.special import lambertw

try:
    from .BasePolicy import BasePolicy
except ImportError:
    from BasePolicy import BasePolicy


# --- Utility functions

#: A *crazy* large constant to get all the theoretical results working. The paper suggests :math:`C = 128`.
#:
#: .. warning:: One can choose a much smaller value in order to (try to) have reasonable empirical performances! I have tried :math:`C = 1`. *BUT* the algorithm DOES NOT work better with a much smaller constant: every single simulations I tried end up with a linear regret for :class:`MusicalChairNoSensing`.
ConstantC = 128
ConstantC = 10
ConstantC = 1


def parameter_g(K=9, m=3, T=1000, constant_c=ConstantC):
    r""" Length :math:`g` of the phase 1, from parameters ``K``, ``m`` and ``T``.

    .. math:: g = 128 K \log(3 K m^2 T^2).

    Examples:

    >>> parameter_g(m=2, K=2, T=100)  # DOCTEST: +ELLIPSIS
    3171.428...
    >>> parameter_g(m=2, K=2, T=1000)  # DOCTEST: +ELLIPSIS
    4350.352...
    >>> parameter_g(m=2, K=3, T=100)  # DOCTEST: +ELLIPSIS
    4912.841...
    >>> parameter_g(m=3, K=3, T=100)  # DOCTEST: +ELLIPSIS
    5224.239...
    """
    return (np.log(3) + np.log(K) + 2*np.log(m) + 2*np.log(T)) * constant_c * K


def estimate_length_phases_12(K=3, m=9, Delta=0.1, T=1000):
    """ Estimate the length of phase 1 and 2 from the parameters of the problem.

    Examples:

    >>> estimate_length_phases_12(m=2, K=2, Delta=0.1, T=100)
    198214307
    >>> estimate_length_phases_12(m=2, K=2, Delta=0.01, T=100)
    19821430723
    >>> estimate_length_phases_12(m=2, K=2, Delta=0.1, T=1000)
    271897030
    >>> estimate_length_phases_12(m=2, K=3, Delta=0.1, T=100)
    307052623
    >>> estimate_length_phases_12(m=2, K=5, Delta=0.1, T=100)
    532187397
    """
    assert Delta > 0, "Error: estimate_length_phases_12 needs a non zero gap."  # DEBUG
    return int(625/128 * ConstantC * parameter_g(K=K, m=m, T=T) / Delta**2)


def smallest_T_from_where_length_phases_12_is_larger(K=3, m=9, Delta=0.1, Tmax=1e9):
    """ Compute the smallest horizon T from where the (estimated) length of phases 1 and 2 is larger than T.

    Examples:

    >>> smallest_T_from_where_length_phases_12_is_larger(K=2, m=1)
    687194767
    >>> smallest_T_from_where_length_phases_12_is_larger(K=3, m=2)
    1009317314
    >>> smallest_T_from_where_length_phases_12_is_larger(K=3, m=3)
    1009317314

    Examples with even longer phase 1:

    >>> smallest_T_from_where_length_phases_12_is_larger(K=10, m=5)
    1009317314
    >>> smallest_T_from_where_length_phases_12_is_larger(K=10, m=10)
    1009317314

    With :math:`K=100` arms, it starts to be crazy:

    >>> smallest_T_from_where_length_phases_12_is_larger(K=100, m=10)
    1009317314
    """
    T = 1
    while estimate_length_phases_12(K=K, m=m, Delta=Delta, T=T) > T and T < Tmax:
        T *= 2
    maxT = T
    T /= 2
    while estimate_length_phases_12(K=K, m=m, Delta=Delta, T=T) > T and T < Tmax:
        T += maxT/100
    return int(T)


#: Different states during the Musical Chair with no sensing algorithm
State = Enum('State', [
    'NotStarted',
    'InitialPhase',
    'UniformWaitPhase2',
    'MusicalChair',
    'Sitted'
])


# --- Class MusicalChairNoSensing

class MusicalChairNoSensing(BasePolicy):
    """ MusicalChairNoSensing: implementation of the decentralized multi-player policy from [["Multiplayer bandits without observing collision information", by Gabor Lugosi and Abbas Mehrabian]](https://arxiv.org/abs/1808.08416).
    """

    def __init__(self,
                nbPlayers=1, nbArms=1, horizon=1000,
                constant_c=ConstantC,
                lower=0., amplitude=1.
        ):  # Named argument to give them in any order
        """
        - nbArms: number of arms (``K`` in the paper),
        - nbPlayers: number of players (``m`` in the paper),
        - horizon: horizon (length) of the game (``T`` in the paper),

        Example:

        >>> nbPlayers, nbArms, horizon = 3, 9, 10000
        >>> player1 = MusicalChairNoSensing(nbPlayers, nbArms, horizon)

        For multi-players use:

        >>> configuration["players"] = Selfish(NB_PLAYERS, MusicalChairNoSensing, nbArms, nbPlayers=nbPlayers, horizon=horizon).children

        or

        >>> configuration["players"] = [ MusicalChairNoSensing(nbPlayers=nbPlayers, nbArms=nbArms, horizon=horizon) for _ in range(NB_PLAYERS) ]
        """
        super(MusicalChairNoSensing, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert 0 < nbPlayers <= nbArms, "Error, the parameter 'nbPlayers' = {} for MusicalChairNoSensing class has to be None or > 0.".format(nbPlayers)  # DEBUG
        self.state = State.NotStarted  #: Current state
        # Store parameters
        self.nbPlayers = nbPlayers  #: Number of players
        self.nbArms = nbArms  #: Number of arms
        self.horizon = horizon  #: Parameter T (horizon)

        # Internal memory
        self.chair = None  #: Current chair. Not sited yet.
        self.cumulatedRewards = np.zeros(nbArms)  #: That's the s_i(t) of the paper
        self.nbObservations = np.zeros(nbArms, dtype=int)  #: That's the o_i of the paper
        self.A = np.random.permutation(nbArms)  #: A random permutation of arms, it will then be of size nbPlayers!

        # Parameters
        self.constant_c = constant_c
        g = parameter_g(K=nbArms, m=nbArms, T=horizon, constant_c=constant_c)  #: Used for the stopping criteria of phase 1
        self.constant_g = g
        self.constant_in_testing_the_gap = (1 - 1.0/self.nbArms)**(self.nbPlayers - 1) * 3 * np.sqrt(g)

        # Implementation details
        self.tau_phase_2 = -1  #: Time when phase 2 starts
        self.t = -1  #: Internal times

    def __str__(self):
        # return r"MCNoSensing($M={}$, $T={}$)".format(self.nbPlayers, self.horizon)  # Use current estimate
        return r"MCNoSensing($M={}$, $T={}$, $c={:.3g}$, $g={:.3g}$)".format(self.nbPlayers, self.horizon, self.constant_c, self.constant_g)  # Use current estimate

    def startGame(self):
        """ Just reinitialize all the internal memory, and decide how to start (state 1 or 2)."""
        self.t = -1  # -1 because t += 1 is done in self.choice()
        self.chair = None  # Not sited yet
        self.cumulatedRewards.fill(0)
        self.nbObservations.fill(0)
        self.A = np.random.permutation(self.nbArms)  # We have to select a random permutation, instead of fill(0), in case the initial phase was too short, the player is not too stupid
        self.state = State.InitialPhase

    def choice(self):
        """ Choose an arm, as described by the Musical Chair with no Sensing algorithm."""
        self.t += 1
        if self.chair is not None:  # and self.state == State.Sitted:
            # If the player is already sit, nothing to do
            self.state = State.Sitted  # We can stay sitted: no collision right after we sit
            # If we can choose this chair like this, it's because we were already sitted, without seeing a collision
            # print("\n- A MusicalChairNoSensing player chose arm {} because it's his chair, and time t = {} ...".format(self.chair, self.t))  # DEBUG
            return self.chair
        elif self.state == State.InitialPhase or self.state == State.UniformWaitPhase2:
            # Play as initial phase: choose a random arm, uniformly among all the K arms
            i = np.random.randint(self.nbArms)
            # print("\n- A MusicalChairNoSensing player chose a random arm {} among [1,...,{}] as it is in state InitialPhase, and time t = {} ...".format(i, self.nbArms, self.t))  # DEBUG
            return i
        elif self.state == State.MusicalChair:
            # Play as musical chair: choose a random arm, among the M bests
            i = np.random.choice(self.A)  # Random arm among the M bests
            self.chair = i  # Assume that it would be a good chair
            # print("\n- A MusicalChairNoSensing player chose a random arm i={} among the {}-best arms in [1,...,{}] as it is in state MusicalChairNoSensing, and time t = {} ...".format(i, self.nbPlayers, self.nbArms, self.t))  # DEBUG
            return i
        else:
            raise ValueError("MusicalChairNoSensing.choice() should never be in this case. Fix this code, quickly!")

    def getReward(self, arm, reward):
        """ Receive a reward on arm of index 'arm', as described by the Musical Chair  with no Sensing algorithm.

        - If not collision, receive a reward after pulling the arm.
        """
        # print("- A MusicalChairNoSensing player receive reward = {} on arm {}, in state {} and time t = {}...".format(reward, arm, self.state, self.t))  # DEBUG
        # If not collision, receive a reward after pulling the arm
        if self.state == State.InitialPhase:
            # Count the observation, update arm cumulated reward
            self.nbObservations[arm] += 1      # One observation of this arm
            self.cumulatedRewards[arm] += (reward - self.lower) / self.amplitude  # More reward

            # we sort the empirical means, and compare the m-th and (m+1)-th ones
            empiricalMeans = self.cumulatedRewards / self.nbObservations
            sortedMeans = np.sort(empiricalMeans)[::-1]  # XXX decreasing order!
            # print("Sorting empirical means… sortedMeans = {}".format(sortedMeans))  # DEBUG
            if self.nbPlayers < self.nbArms:
                gap_Mbest_Mworst = abs(sortedMeans[self.nbPlayers] - sortedMeans[self.nbPlayers + 1])
            else:
                gap_Mbest_Mworst = 0
            # print("Gap between M-best and M-worst set (with M = {}) is {}, compared to {}...".format(self.nbPlayers, gap_Mbest_Mworst, self.constant_in_testing_the_gap / np.sqrt(self.t)))
            if gap_Mbest_Mworst >= self.constant_in_testing_the_gap / np.sqrt(self.t):
                # print("Gap was larger than the threshold, so this player switch to uniform phase 2!")
                self.state = State.UniformWaitPhase2
                self.tau_phase_2 = self.t

        # And if t = Time0, we are done with the phase 2
        elif self.state == State.UniformWaitPhase2 and (self.t - self.tau_phase_2) >= 24 * self.tau_phase_2:
            self._endPhase2()
        elif self.state == State.MusicalChair:
            assert self.chair is not None, "Error: bug in my code in handleCollision() for MusicalChair class."  # DEBUG
            if reward <= 0:
                self.chair = None  # Cannot stay sit here

    def _endPhase2(self):
        """ Small computation needed at the end of the initial random exploration phase."""
        # print("\n- A MusicalChairNoSensing player has to switch from InitialPhase to MusicalChairNoSensing ...")  # DEBUG
        self.state = State.MusicalChair  # Switch ONCE to phase 3

        # First, we compute the empirical means mu_i
        empiricalMeans = (1 + self.cumulatedRewards) / (1 + self.nbObservations)

        # Finally, sort their index by empirical means, decreasing order
        self.A = np.argsort(-empiricalMeans)[:self.nbPlayers]  # among the best M arms!

    def handleCollision(self, arm, reward=None):
        """ Handle a collision, on arm of index 'arm'.

        - Here, as its name suggests it, the :class:`MusicalChairNoSensing` algorithm does *not* use any collision information, hence this method is empty.
        - Warning: this method has to be implemented in the collision model, it is NOT implemented in the EvaluatorMultiPlayers.
        """
        pass
