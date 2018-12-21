# -*- coding: utf-8 -*-
r""" TrekkingTSN: implementation of the decentralized multi-player policy from [R.Kumar, A.Yadav, S.J.Darak, M.K.Hanawal, Trekking based Distributed Algorithm for Opportunistic Spectrum Access in Infrastructure-less Network, 2018](XXX).

- Each player has 3 states, 1st is channel characterization, 2nd is Trekking phase
- 1st step
    + FIXME
- 2nd step:
    + FIXME
"""
from __future__ import division, print_function  # Python 2 compatibility, division

__author__ = "Lilian Besson"
__version__ = "0.9"

from enum import Enum  # For the different states
import numpy as np

try:
    from .BasePolicy import BasePolicy
except ImportError:
    from BasePolicy import BasePolicy


# --- Functions to compute the optimal choice of Time0 proposed in [Kumar et al., 2018]

def special_times(nbArms=10, theta=0.01, epsilon=0.1, delta=0.05):
    r""" Compute the lower-bound suggesting "large-enough" values for the different parameters :math:`T_{RH}`, :math:`T_{SH}` and :math:`T_{TR}` that should guarantee constant regret with probability at least :math:`1 - \delta`, if the gap :math:`\Delta` is larger than :math:`\epsilon` and the smallest mean is larger than :math:`\theta`.

    .. math::

        T_{RH} &= \frac{\log(\frac{\delta}{3 K})}{\log(1 - \theta (1 - \frac{1}{K})^{K-1}))} \\
        T_{SH} &= (2 K / \varepsilon^2) \log(\frac{2 K^2}{\delta / 3}) \\
        T_{TR} &= \lceil\frac{\log((\delta / 3) K XXX)}{\log(1 - \theta)} \rceil \frac{(K - 1) K}{2}.

    - Cf. Theorem 1 of [Kumar et al., 2018](XXX).
    - Examples:

    >>> nbArms = 8
    >>> theta = Delta = 0.07
    >>> epsilon = theta
    >>> delta = 0.1
    >>> special_times(nbArms=nbArms, theta=theta, epsilon=epsilon, delta=delta)  # doctest: +ELLIPSIS
    (197, 26949, -280)
    >>> delta = 0.01
    >>> special_times(nbArms=nbArms, theta=theta, epsilon=epsilon, delta=delta)  # doctest: +ELLIPSIS
    (279, 34468, 616)
    >>> delta = 0.001
    >>> special_times(nbArms=nbArms, theta=theta, epsilon=epsilon, delta=delta)  # doctest: +ELLIPSIS
    (362, 41987, 1512)
    """
    K = nbArms
    XXX = K  # FIXME this is unspecified in their paper (see equation (5))
    T_RH = int((np.log(delta / (3 * K))) / (np.log(1 - theta * (1 - 1/K)**(K-1))))
    T_SH = int((2 * K / epsilon**2) * np.log(2 * K**2 / (delta / 3)))
    T_TR = int(np.ceil((np.log((delta / 3) * K * XXX)) / (np.log(1 - theta))) * ((K - 1) * K) / 2.)
    # XXX this T_TR CANNOT be negative !
    assert T_RH > 0, "Error: time T_RH cannot be <= 0 but was found = {}...".format(T_RH)  # DEBUG
    assert T_SH > 0, "Error: time T_SH cannot be <= 0 but was found = {}...".format(T_SH)  # DEBUG
    # assert T_TR > 0, "Error: time T_TR cannot be <= 0 but was found = {}...".format(T_TR)  # DEBUG
    return T_RH, T_SH, T_TR


def boundOnFinalRegret(T_RH, T_SH, T_TR, nbPlayers, nbArms):
    r""" Use the upper-bound on regret when :math:`T_{RH}`, :math:`T_{SH}` and :math:`T_{TR}` and :math:`M` are known.

    - The "constant" regret of course grows linearly with :math:`T_{RH}`, :math:`T_{SH}` and :math:`T_{TR}`, as:

        .. math:: \forall T \geq T_{RH} + T_{SH} + T_{TR}, \;\; R_T \leq M (T_{RH} + (1 - \frac{M}{K}) T_{SH} + T_{TR}).

    .. warning:: this bound is not a deterministic result, it is only value with a certain probability (at least :math:`1 - \delta`, if :math:`T_{RH}`, :math:`T_{SH}` and :math:`T_{TR}` is chosen as given by :func:`special_times`).


    - Cf. Theorem 1 of [Kumar et al., 2018](XXX).
    - Examples:

    >>> boundOnFinalRegret(197, 26949, -280, 2, 8)  # doctest: +ELLIPSIS
    40257.5
    >>> boundOnFinalRegret(279, 34468, 616, 2, 8)   # doctest: +ELLIPSIS
    53492.0
    >>> boundOnFinalRegret(362, 41987, 1512, 2, 8)  # doctest: +ELLIPSIS
    66728.5

        + For :math:`M=5`:

    >>> boundOnFinalRegret(197, 26949, -280, 5, 8)  # doctest: +ELLIPSIS
    50114.375
    >>> boundOnFinalRegret(279, 34468, 616, 5, 8)   # doctest: +ELLIPSIS
    69102.5
    >>> boundOnFinalRegret(362, 41987, 1512, 5, 8)  # doctest: +ELLIPSIS
    88095.625

        + For :math:`M=K=8`:

    >>> boundOnFinalRegret(197, 26949, -280, 8, 8)  # doctest: +ELLIPSIS
    -664.0  # there is something wrong with T_TR !
    >>> boundOnFinalRegret(279, 34468, 616, 8, 8)   # doctest: +ELLIPSIS
    7160.0
    >>> boundOnFinalRegret(362, 41987, 1512, 8, 8)  # doctest: +ELLIPSIS
    14992.0
    """
    return nbPlayers * (T_RH + T_SH * (1. - float(nbPlayers)/nbArms) + T_TR)


#: Different states during the Musical Chair algorithm
State = Enum('State', ['NotStarted', 'ChannelCharacterization', 'TrekkingTSN'])


# --- Class TrekkingTSN

class TrekkingTSN(BasePolicy):
    """ TrekkingTSN: implementation of the single-player policy from [R.Kumar, A.Yadav, S.J.Darak, M.K.Hanawal, Trekking based Distributed Algorithm for Opportunistic Spectrum Access in Infrastructure-less Network, 2018](XXX).
    """

    def __init__(self, nbArms,
                theta=0.01, epsilon=0.1, delta=0.05,
                lower=0., amplitude=1.
        ):  # Named argument to give them in any order
        """
        - nbArms: number of arms,

        Example:

        >>> nbArms = 8
        >>> theta, epsilon, delta = 0.01, 0.1, 0.05
        >>> player1 = TrekkingTSN(nbArms, theta=theta, epsilon=epsilon, delta=delta)

        For multi-players use:

        >>> configuration["players"] = Selfish(NB_PLAYERS, TrekkingTSN, nbArms, theta=theta, epsilon=epsilon, delta=delta).children
        """
        super(TrekkingTSN, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.state = State.NotStarted  #: Current state

        # compute times
        T_RH, T_SH, T_TR = special_times(nbArms=nbArms, theta=theta, epsilon=epsilon, delta=delta)

        self.theta = theta  #: Parameter :math:`\theta`.
        self.epsilon = epsilon  #: Parameter :math:`\epsilon`.
        self.delta = delta  #: Parameter :math:`\delta`.

        # Store parameters
        self.T_RH = T_RH  #: Parameter :math:`T_{RH}` computed from :func:`special_times`
        self.T_SH = T_SH  #: Parameter :math:`T_{SH}` computed from :func:`special_times`
        self.T_CC = T_RH + T_SH  #: Parameter :math:`T_{CC} = T_{RH} + T_{SH}`
        self.T_TR = T_TR  #: Parameter :math:`T_{TR}` computed from :func:`special_times`

        # Internal memory
        self.last_was_successful = False  #: That's the l of the paper
        self.last_choice = None  #: Keep memory of the last choice for CC phase
        self.cumulatedRewards = np.zeros(nbArms)  #: That's the V_n of the paper
        self.nbObservations = np.zeros(nbArms, dtype=int)  #: That's the S_n of the paper

        self.J = -1
        self.lock_channel = False  #: That's the L of the paper
        self.Y = np.zeros(nbArms, dtype=int)  # That's the Yi of the paper
        self.M = np.zeros(nbArms)  # That's the Mj of the paper
        self.index_sort = None

        # Implementation details
        self.t = -1  #: Internal times

    def __str__(self):
        return r"TSN(${} = {:.3g}, {} = {:.3g}$)".format(r"T_{RH}", self.T_RH, r"T_{SH}", self.T_SH)  # Use current estimate

    def startGame(self):
        """ Just reinitialize all the internal memory, and decide how to start (state 1 or 2)."""
        self.t = -1  # -1 because t += 1 is done in self.choice()
        self.cumulatedRewards.fill(0)
        self.nbObservations.fill(0)
        self.J = -1
        self.lock_channel = False
        self.Y.fill(0)
        self.M.fill(0)
        self.index_sort = None
        self.state = State.ChannelCharacterization

    def choice(self):
        """ Choose an arm, as described by the Musical Chair algorithm."""
        self.t += 1
        i = self.last_choice  # by default, stay on the current arm

        if self.state == State.ChannelCharacterization:
            if self.last_was_successful:
                # choose next channel sequentially
                i = (self.last_choice + 1) % self.nbArms
            else:
                # randomly choose channel
                i = np.random.randint(self.nbArms)
        elif self.state == State.TrekkingTSN:
            if self.t == self.T_CC or np.sum(self.Y) == 0:
                i = (self.J - 1) % self.nbArms
            elif self.lock_channel:
                i = self.last_choice
            elif self.Y[self.J] <= self.M[self.J]:
                i = self.last_choice
            else:
                # take next best channel
                i = (self.last_choice - 1) % self.nbArms
                self.J = self.last_choice

            # finally, in Trekking phase, increase Y_J by 1
            self.Y[self.J] += 1

        # Keep in memory the last choice
        self.last_choice = i

        # and use the permutation to aim at the sorted arm
        if self.state == State.TrekkingTSN and self.index_sort is not None:
            return self.index_sort[i]
        else:
            return i

    def getReward(self, arm, reward):
        """ Receive a reward on arm of index 'arm', as described by the Musical Chair algorithm.

        - If not collision, receive a reward after pulling the arm.
        """
        # print("- A TrekkingTSN player receive reward = {} on arm {}, in state {} and time t = {}...".format(reward, arm, self.state, self.t))  # DEBUG
        # If not collision, receive a reward after pulling the arm
        if self.state == State.ChannelCharacterization:
            # Count the observation, update arm cumulated reward
            self.nbObservations[arm] += 1      # One observation of this arm
            r_t = (reward - self.lower) / self.amplitude
            self.cumulatedRewards[arm] += r_t  # More reward
            if r_t > 0:
                self.last_was_successful = True
        # else:
        #     assert self.state == State.TrekkingTSN, "Error: state = {} should be TrekkingTSN here.".format(self.state)  # DEBUG
        #     pass

        # And if t = T_CC, we are do with the initial CC phase
        if self.t >= self.T_CC and self.state == State.ChannelCharacterization:
            self._endCCPhase()

    def _endCCPhase(self):
        """ Small computation needed at the end of the initial CC phase."""
        # print("\n- A TrekkingTSN player has to switch from ChannelCharacterization to TrekkingTSN ...")  # DEBUG
        self.state = State.TrekkingTSN  # Switch ONCE to state 2

        # First, we compute the empirical means mu_i
        # empiricalMeans = (1. + self.cumulatedRewards) / (1. + self.nbObservations)
        empiricalMeans = self.cumulatedRewards / self.nbObservations

        # Finally, sort their index by empirical means, decreasing order
        self.index_sort = np.argsort(-empiricalMeans)

        self.J = self.index_sort[self.last_choice]

        for j in range(self.nbArms):
            self.M[j] = sum(np.ceil(np.log(self.delta / 3.) / (1 - empiricalMeans[self.index_sort[j]])) for i in range(j-1))

    def handleCollision(self, arm, reward=None):
        """ Handle a collision, on arm of index 'arm'.

        - Warning: this method has to be implemented in the collision model, it is NOT implemented in the EvaluatorMultiPlayers.
        """
        # print("- A TrekkingTSN player saw a collision on arm {}, in state {}, and time t = {} ...".format(arm, self.state, self.t))  # DEBUG
        if self.state == State.ChannelCharacterization:
            # count one more collision in this initial phase (no matter the arm)
            pass
        else:
            assert self.state == State.TrekkingTSN, "Error: state = {} should be TrekkingTSN here.".format(self.state)  # DEBUG
            self.lock_channel = True
            if self.last_choice == self.J:
                print("Warning: TrekkingTSN algorithm saw a collision while playing {} and has J = {}, but will lock this channel (forever?), that's not smart... (L = {}).".format(self.last_choice, self.J, self.lock_channel))  # DEBUG
            self.last_choice = self.J
