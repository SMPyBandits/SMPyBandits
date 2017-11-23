# -*- coding: utf-8 -*-
r""" MusicalChair: implementation of the single-player policy from [A Musical Chair approach, Shamir et al., 2015](https://arxiv.org/abs/1512.02866).

- Each player has 3 states, 1st is random exploration, 2nd is musical chair, 3rd is staying sit
- 1st step
    + Every player tries uniformly an arm for :math:`T_0` steps, counting the empirical means of each arm, and the number of observed collisions :math:`C_{T_0}`
    + Finally, :math:`N^* = M` = ``nbPlayers`` is estimated based on nb of collisions :math:`C_{T_0}`, and the :math:`N^*` best arms are computed from their empirical means
- 2nd step:
    + Every player chose an arm uniformly, among the :math:`N^*` best arms, until she does not encounter collision right after choosing it
    + When an arm was chosen by only one player, she decides to sit on this chair (= arm)
- 3rd step:
    + Every player stays sitted on her chair for the rest of the game
    + :math:`\implies` constant regret if :math:`N^*` is well estimated and if the estimated N* best arms were correct
    + :math:`\implies` linear regret otherwise
"""
from __future__ import division, print_function  # Python 2 compatibility, division

__author__ = "Lilian Besson"
__version__ = "0.8"

from enum import Enum  # For the different states
import numpy as np
from .BasePolicy import BasePolicy


# --- Functions to compute the optimal choice of Time0 proposed in [Shamir et al., 2015]

def optimalT0(nbArms=10, epsilon=0.1, delta=0.05):
    r""" Compute the lower-bound suggesting "large-enough" values for :math:`T_0` that should guarantee constant regret with probability at least :math:`1 - \delta`, if the gap :math:`\Delta` is larger than :math:`\epsilon`.

    - Cf. Theorem 1 of [Shamir et al., 2015](https://arxiv.org/abs/1512.02866).

    Examples:

    - For :math:`K=2` arms, and in order to have a constant regret with probability at least :math:`90\%`, if the gap :math:`\Delta` is known to be :math:`\geq 0.05`, then their theoretical analysis suggests to use :math:`T_0 \geq 18459`. That's very huge, for just two arms!

    >>> optimalT0(2, 0.1, 0.05)     # Just 2 arms !
    18459                           # ==> That's a LOT of steps for just 2 arms!

    - For a harder problem with :math:`K=6` arms, for a risk smaller than :math:`1\%` and a gap :math:`\Delta \geq 0.05`, they suggest at least :math:`T_0 \geq 7646924`, i.e., about 7 millions of trials. That is simply too much for any realistic system, and starts to be too large for simulated systems.

    >>> optimalT0(6, 0.01, 0.05)    # Constant regret with >99% proba
    7646924                         # ==> That's a LOT of steps!
    >>> optimalT0(6, 0.001, 0.05)   # Reasonable value of epsilon
    764692376                       # ==> That's a LOT of steps!!!

    - For an even harder problem with :math:`K=17` arms, the values given by their Theorem 1 start to be really unrealistic:

    >>> optimalT0(17, 0.01, 0.05)   # Constant regret with >99% proba
    27331794                        # ==> That's a LOT of steps!
    >>> optimalT0(17, 0.001, 0.05)  # Reasonable value of epsilon
    2733179304                      # ==> That's a LOT of steps!!!
    """
    K = nbArms
    T0_1 = (K / 2.) * np.log(2 * K**2 / delta)
    T0_2 = ((16 * K) / (epsilon**2)) * np.log(4 * K**2 / delta)
    T0_3 = (K**2 * np.log(2 / delta**2)) / 0.02   # delta**2 or delta_2 ? Typing mistake in their paper
    T0 = max(T0_1, T0_2, T0_3)
    return int(np.ceil(T0))


def boundOnFinalRegret(T0, nbPlayers):
    r""" Use the upper-bound on regret when :math:`T_0` and :math:`M` are known.

    - The "constant" regret of course grows linearly with :math:`T_0`, as:

        .. math:: \forall T \geq T_0, \;\; R_T \leq T_0 K + 2 \mathrm{exp}(2) K.

    .. warning:: this bound is not a deterministic result, it is only value with a certain probability (at least :math:`1 - \delta`, if :math:`T_0` is chosen as given by :func:`optimalT0`).


    - Cf. Theorem 1 of [Shamir et al., 2015](https://arxiv.org/abs/1512.02866).
    - Examples:

    >>> boundOnFinalRegret(18459, 2)        # Crazy constant regret!  # doctest: +ELLIPSIS
    36947.5..
    >>> boundOnFinalRegret(7646924, 6)      # Crazy constant regret!!  # doctest: +ELLIPSIS
    45881632.6...
    >>> boundOnFinalRegret(764692376, 6)    # Crazy constant regret!!  # doctest: +ELLIPSIS
    4588154344.6...
    >>> boundOnFinalRegret(27331794, 17)    # Crazy constant regret!!  # doctest: +ELLIPSIS
    464640749.2...
    >>> boundOnFinalRegret(2733179304, 17)  # Crazy constant regret!!  # doctest: +ELLIPSIS
    46464048419.2...
    """
    return T0 * nbPlayers + 2 * np.exp(2) * nbPlayers


#: Different states during the Musical Chair algorithm
State = Enum('State', ['NotStarted', 'InitialPhase', 'MusicalChair', 'Sitted'])


# --- Class MusicalChair

class MusicalChair(BasePolicy):
    """ MusicalChair: implementation of the single-player policy from [A Musical Chair approach, Shamir et al., 2015](https://arxiv.org/abs/1512.02866).
    """

    def __init__(self, nbArms, Time0=0.25, Time1=None, N=None, lower=0., amplitude=1.):  # Named argument to give them in any order
        """
        - nbArms: number of arms,
        - Time0: required, number of step, or portion of the horizon Time1 (optional), for the first step (pure random exploration by each players),
        - N: optional, exact or upper bound on the number of players,
        - Time1: optional, only used to compute Time0 if Time0 is fractional (eg. 0.2).

        Example:

        >>> nbArms, Time0, Time1, N = 17, 0.1, 10000, 6
        >>> player1 = MusicalChair(nbArms, Time0, Time1, N)

        For multi-players use:

        >>> configuration["players"] = Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.25, Time1=HORIZON, N=NB_PLAYERS).children
        """
        super(MusicalChair, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        nbPlayers = N
        assert nbPlayers is None or nbPlayers > 0, "Error, the parameter 'nbPlayers' for MusicalChair class has to be None or > 0."
        self.state = State.NotStarted  #: Current state
        if 0 < Time0 < 1:  # Time0 is a fraction of the horizon Time1
            Time0 = int(Time0 * Time1)  # Lower bound
        elif Time0 >= 1:
            Time0 = int(Time0)
        # Store parameters
        self.Time0 = Time0  #: Parameter T0
        self.nbPlayers = nbPlayers  #: Number of players
        # Internal memory
        self.chair = None  #: Current chair. Not sited yet.
        self.cumulatedRewards = np.zeros(nbArms)  #: That's the s_i(t) of the paper
        self.nbObservations = np.zeros(nbArms, dtype=int)  #: That's the o_i of the paper
        self.A = np.random.permutation(nbArms)  #: A random permutation of arms, it will then be of size nbPlayers!
        self.nbCollision = 0  #: Number of collisions, that's the C_Time0 of the paper
        # Implementation details
        self.t = -1  #: Internal times

    def __str__(self):
        # return r"MusicalChair($N^*={}$, $T_0={}$)".format(self.nbPlayers, self.Time0)  # Use current estimate
        return r"MusicalChair($T_0={}$)".format(self.Time0)  # Use current estimate

    def startGame(self):
        """ Just reinitialize all the internal memory, and decide how to start (state 1 or 2)."""
        self.t = -1  # -1 because t += 1 is done in self.choice()
        self.chair = None  # Not sited yet
        self.cumulatedRewards.fill(0)
        self.nbObservations.fill(0)
        self.A = np.random.permutation(self.nbArms)  # We have to select a random permutation, instead of fill(0), in case the initial phase was too short, the player is not too stupid
        self.nbCollision = 0
        # if nbPlayers is None, start by estimating it to N*, with the initial phase procedure
        if self.nbPlayers is None:
            self.state = State.InitialPhase
        else:  # No need for an initial phase if nbPlayers is known (given)
            self.Time0 = 0
            self.state = State.MusicalChair

    def choice(self):
        """ Chose an arm, as described by the Musical Chair algorithm."""
        self.t += 1
        if self.chair is not None:  # and self.state == State.Sitted:
            # If the player is already sit, nothing to do
            self.state = State.Sitted  # We can stay sitted: no collision right after we sit
            # If we can choose this chair like this, it's because we were already sitted, without seeing a collision
            # print("\n- A MusicalChair player chose arm {} because it's his chair, and time t = {} ...".format(self.chair, self.t))  # DEBUG
            return self.chair
        elif self.state == State.InitialPhase:
            # Play as initial phase: choose a random arm, uniformly among all the K arms
            i = np.random.randint(self.nbArms)
            # print("\n- A MusicalChair player chose a random arm {} among [1,...,{}] as it is in state InitialPhase, and time t = {} ...".format(i, self.nbArms, self.t))  # DEBUG
            return i
        elif self.state == State.MusicalChair:
            # Play as musical chair: choose a random arm, among the M bests
            i = np.random.choice(self.A)  # Random arm among the M bests
            self.chair = i  # Assume that it would be a good chair
            # print("\n- A MusicalChair player chose a random arm i={} of index={} among the {}-best arms in [1,...,{}] as it is in state MusicalChair, and time t = {} ...".format(i, k, self.nbPlayers, self.nbArms, self.t))  # DEBUG
            return i
        else:
            raise ValueError("MusicalChair.choice() should never be in this case. Fix this code, quickly!")

    def getReward(self, arm, reward):
        """ Receive a reward on arm of index 'arm', as described by the Musical Chair algorithm.

        - If not collision, receive a reward after pulling the arm.
        """
        # print("- A MusicalChair player receive reward = {} on arm {}, in state {} and time t = {}...".format(reward, arm, self.state, self.t))  # DEBUG
        # If not collision, receive a reward after pulling the arm
        if self.state == State.InitialPhase:
            # Count the observation, update arm cumulated reward
            self.nbObservations[arm] += 1      # One observation of this arm
            self.cumulatedRewards[arm] += (reward - self.lower) / self.amplitude  # More reward
        # elif self.state in [State.MusicalChair, State.Sitted]:
        #     pass  # Nothing to do in this second phase
        #     # We don't care anymore about rewards in this step

        # And if t = Time0, we are do with the initial phase
        if self.t >= self.Time0 and self.state == State.InitialPhase:
            self._endInitialPhase()

    def _endInitialPhase(self):
        """ Small computation needed at the end of the initial random exploration phase."""
        # print("\n- A MusicalChair player has to switch from InitialPhase to MusicalChair ...")  # DEBUG
        self.state = State.MusicalChair  # Switch ONCE to state 2
        # First, we compute the empirical means mu_i
        empiricalMeans = (1 + self.cumulatedRewards) / (1 + self.nbObservations)
        # Then, we compute the final estimate of N* = nbPlayers
        if self.nbCollision == self.Time0:  # 1st case, we only saw collisions!
            self.nbPlayers = self.nbArms  # Worst case, pessimist estimate of the nb of players
        else:  # 2nd case, we didn't see only collisions
            self.nbPlayers = int(round(1 + np.log((self.Time0 - self.nbCollision) / self.Time0) / np.log(1. - 1. / self.nbArms)))
        # Finally, sort their index by empirical means, decreasing order
        self.A = np.argsort(-empiricalMeans)[:self.nbPlayers]  # FIXED among the best M arms!

    def handleCollision(self, arm, reward=None):
        """ Handle a collision, on arm of index 'arm'.

        - Warning: this method has to be implemented in the collision model, it is NOT implemented in the EvaluatorMultiPlayers.
        """
        # print("- A MusicalChair player saw a collision on arm {}, in state {}, and time t = {} ...".format(arm, self.state, self.t))  # DEBUG
        if self.state == State.InitialPhase:
            # count one more collision in this initial phase (no matter the arm)
            self.nbCollision += 1
        elif self.state == State.MusicalChair:
            assert self.chair is not None, "Error: bug in my code in handleCollision() for MusicalChair class."  # DEBUG
            self.chair = None  # Cannot stay sit here
