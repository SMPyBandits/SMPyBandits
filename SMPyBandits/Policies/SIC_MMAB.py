# -*- coding: utf-8 -*-
r""" SIC_MMAB: implementation of the decentralized multi-player policy from [["SIC-MMAB: Synchronisation Involves Communication in Multiplayer Multi-Armed Bandits", by Etienne Boursier, Vianney Perchet, arXiv 1809.08151, 2018](https://arxiv.org/abs/1809.08151)].

- The algorithm is quite complicated, please see the paper (Algorithm 1, page 6).
- The UCB-H indexes are used, for more details see :class:`Policies.UCBH`.
"""
from __future__ import division, print_function  # Python 2 compatibility, division

__author__ = "Lilian Besson"
__version__ = "0.9"

from enum import Enum  # For the different states
import numpy as np

try:
    from .BasePolicy import BasePolicy
    from .kullback import klucbBern
except (ImportError, SystemError):
    from BasePolicy import BasePolicy
    from kullback import klucbBern

#: Default value for the constant c used in the computation of KL-UCB index.
c = 1.  #: default value, as it was in pymaBandits v1.0
# c = 1.  #: as suggested in the Theorem 1 in https://arxiv.org/pdf/1102.2490.pdf


#: Default value for the tolerance for computing numerical approximations of the kl-UCB indexes.
TOLERANCE = 1e-4


#: Different states during the Musical Chair algorithm
State = Enum('State', [
    'Fixation',
    'Estimation',
    'Exploration',
    'Communication',
    'Exploitation',
])


# --- Class SIC_MMAB

class SIC_MMAB(BasePolicy):
    """ SIC_MMAB: implementation of the decentralized multi-player policy from [["SIC-MMAB: Synchronisation Involves Communication in Multiplayer Multi-Armed Bandits", by Etienne Boursier, Vianney Perchet, arXiv 1809.08151, 2018](https://arxiv.org/abs/1809.08151)].
    """

    def __init__(self, nbArms, horizon,
            lower=0., amplitude=1.,
            alpha=4.0, verbose=False,
        ):  # Named argument to give them in any order
        r"""
        - nbArms: number of arms,
        - horizon: to compute the time :math:`T_0 = \lceil K \log(T) \rceil`,
        - alpha: for the UCB/LCB computations.

        Example:

        >>> nbArms, horizon, N = 17, 10000, 6
        >>> player1 = SIC_MMAB(nbArms, horizon, N)

        For multi-players use:

        >>> configuration["players"] = Selfish(NB_PLAYERS, SIC_MMAB, nbArms, horizon=HORIZON).children
        """
        super(SIC_MMAB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self._nbArms = self.nbArms
        self.phase = State.Fixation  #: Current state
        self.horizon = horizon  #: Horizon T of the experiment.
        self.alpha = float(alpha)  #: Parameter :math:`\alpha` for the UCB/LCB computations.
        self.Time0 = int(np.ceil(self.nbArms * np.e * np.log(horizon)))  #: Parameter :math:`T_0 = \lceil K \log(T) \rceil`.
        # Store parameters
        self.ext_rank = -1  #: External rank, -1 until known
        self.int_rank = 0  #: Internal rank, starts to be 0 then increase when needed
        self.nbPlayers = 1  #: Estimated number of players, starts to be 1
        self.last_action = np.random.randint(self.nbArms)  #: Keep memory of the last played action (starts randomly)
        self.t_phase = 0  #: Number of the phase XXX ?
        self.round_number = 0  #: Number of the round XXX ?
        self.last_phase_stats = np.zeros(self.nbArms)
        self.active_arms = np.arange(0, self.nbArms)  #: Set of active arms (kept as a numpy array)
        self.verbose = verbose

    def __str__(self):
        return r"SIC-MMAB(UCB-H, $T_0={}$)".format(self.Time0)

    def startGame(self):
        """ Just reinitialize all the internal memory, and decide how to start (state 1 or 2)."""
        self.phase = State.Fixation  #: Current state
        self.rewards.fill(0)
        self.pulls.fill(0)
        # Store parameters
        self.ext_rank = -1
        self.int_rank = 0
        self.nbPlayers = 1
        self.t_phase = 0
        self.round_number = 0
        self.last_phase_stats.fill(0)
        self.active_arms = np.arange(0, self.nbArms)

    def compute_ucb_lcb(self):
        r""" Compute the Upper-Confidence Bound and Lower-Confidence Bound for active arms, at the current time step.

        - By default, the SIC-MMAB algorithm uses the UCB-H confidence bounds:

        .. math::

            \mathrm{UCB}_k(t) &= \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{\alpha \log(T)}{2 N_k(t)}},\\
            \mathrm{LCB}_k(t) &= \frac{X_k(t)}{N_k(t)} - \sqrt{\frac{\alpha \log(T)}{2 N_k(t)}}.

        - Reference: [Audibert et al. 09].

        - Other possibilities include UCB (see :class:`SIC_MMAB_UCB`) and klUCB (see :class:`SIC_MMAB_klUCB`).
        """
        means = self.rewards[self.active_arms] / self.pulls[self.active_arms]
        bias = np.sqrt(self.alpha * np.log(self.horizon) / (2 * self.pulls[self.active_arms]))
        upper_confidence_bound = means + bias
        lower_confidence_bound = means - bias
        return upper_confidence_bound, lower_confidence_bound

    def choice(self):
        """ Choose an arm, as described by the SIC-MMAB algorithm."""
        # 1) fixation phase
        if self.phase == State.Fixation:
            if self.ext_rank == -1:
                # still trying to fix to an arm
                return np.random.randint(self.nbArms)
            # sequential hopping
            return (self.last_action + 1) % self.nbArms

        # 2) estimation phase
        elif self.phase == State.Estimation:
            if self.t <= self.Time0 + 2*self.ext_rank:
                # waiting its turn to sequential hop
                return self.ext_rank
            # sequential hopping
            return (self.last_action + 1) % self.nbArms

        # 3) exploration phase
        elif self.phase == State.Exploration:
            if self.last_action not in self.active_arms:
                print("Warning: a SIC_MMAB player should never be in exploration phase with last_action = {} not in active_arms = {}...".format(self.last_action, self.active_arms))  # DEBUG
                return np.random.choice(self.active_arms)
            last_index = np.where(self.active_arms == self.last_action)[0][0]
            return self.active_arms[(last_index + 1) % self.nbArms]

        # 4) communication phase
        elif self.phase == State.Communication:
            # your turn to communicate
            if (
                self.t_phase < (self.int_rank + 1) * (self.nbPlayers - 1) * self.nbArms * (self.round_number + 2)
                and (self.t_phase >= (self.int_rank) * (self.nbPlayers - 1) * self.nbArms * (self.round_number + 2))
            ):
                # determine the number of the bit to send, the channel and the player

                # the actual time step in the communication phase (while giving info)
                t0 = self.t_phase % ((self.nbPlayers - 1) * self.nbArms * (self.round_number + 2))
                # the number of the bit to send
                bitToSend = int(t0 % (self.round_number + 2))
                # the index of the arm to send
                k0 = int(((t0 - bitToSend) / (self.round_number + 2)) % self.nbArms)
                # the arm to send
                k = self.active_arms[k0]
                # has to send the bit
                if ((int(self.last_phase_stats[k]) >> bitToSend) % 2):
                    playerToSendTo = (t0 - bitToSend - (self.round_number + 2) * k0) / ((self.round_number + 2) * self.nbArms)
                    # the player to send
                    playerToSendTo = int((playerToSendTo + (playerToSendTo >= self.int_rank)) % self.nbArms)
                    if self.verbose:
                        print('Communicate bit {} about arm {}, at player on arm {}, by player {} at timestep {}'.format(bitToSend, k, self.active_arms[playerToSendTo], self.ext_rank, self.t_phase))  # DEBUG
                    return self.active_arms[playerToSendTo]

            return self.active_arms[self.int_rank]

        # 5) exploitation phase
        elif self.phase == State.Exploitation:
            return self.last_action
        else:
            raise ValueError("SIC_MMAB.choice() should never be in this case. Fix this code, quickly!")

    def getReward(self, arm, reward, collision=False):
        """ Receive a reward on arm of index 'arm', as described by the SIC-MMAB algorithm.

        - If not collision, receive a reward after pulling the arm.
        """
        # print("A SIC_MMAB player got a reward = {} on arm {} at time {}.".format(reward, arm, self.t))  # DEBUG
        self.last_action = arm
        assert reward == 0 or reward == 1, "Error: SIC-MMAB works only for binary rewards!"  # DEBUG

        # 1) fixation phase
        if self.phase == State.Fixation:
            if self.ext_rank == -1:
                if not collision:
                    # successfully fixed, decide the external rank
                    self.ext_rank = (self.Time0 + arm - self.t) % self.nbArms

            # phase change
            if self.t == self.Time0:
                self.phase = State.Estimation  # goes to estimation of M
                assert self.ext_rank >= 0, "Error: in this phase change, a SIC_MMAB player cannot have ext_rank = -1."  # DEBUG
                self.last_action = self.ext_rank

        # 3) estimation phase
        elif self.phase == State.Estimation:
            if collision:  # collision with a player
                assert self.ext_rank >= 0, "Error: in this phase change, a SIC_MMAB player cannot have ext_rank = -1."  # DEBUG
                if self.t <= (self.Time0 + 2*self.ext_rank):
                    # it also increases the internal rank at the same time
                    self.int_rank += 1
                self.nbPlayers += 1

            if self.t == self.Time0 + 2*self.nbArms:
                self.phase = State.Exploration
                self.t_phase = 0
                self.round_number = int(np.ceil(np.log2(self.nbPlayers)))

        # 4) exploration phase
        elif self.phase == State.Exploration:
            self.last_phase_stats[arm] += reward
            self.rewards[arm] += reward
            self.t_phase += 1

            if self.t_phase == (2<<self.round_number) * self.nbArms:
                # end of exploration round
                self.phase = State.Communication
                self.t_phase = 0

        # 5) communication phase
        elif self.phase == State.Communication:
            # reception case
            if (
                self.t_phase >= (self.int_rank + 1) * (self.nbPlayers - 1) * self.nbArms * (self.round_number + 2)
                or (self.t_phase < (self.int_rank) * (self.nbPlayers - 1) * self.nbArms * (self.round_number + 2))
            ):
                if collision:
                    t0 = self.t_phase % ((self.nbPlayers - 1) * self.nbArms * (self.round_number + 2))
                    # the actual time step in the communication phase (while giving info)
                    actual_time = int(t0 % (self.round_number + 2))
                    # the number of the bit to send

                    # the number of the channel to send
                    k0 = int(((t0 - actual_time) / (self.round_number + 2)) % self.nbArms)
                    # the channel to send
                    k = self.active_arms[k0]

                    # Extract ONE bit from actual_time
                    extracted_bit_from_actual_time = ((2 << actual_time) >> 1)
                    self.rewards[k] += extracted_bit_from_actual_time

            self.t_phase += 1

            if (
                self.t_phase == self.nbPlayers * (self.nbPlayers - 1) * self.nbArms * (self.round_number + 2)
                or self.nbPlayers == 1
            ):
                # end of the communication phase
                # all the updates to do
                for k in self.active_arms:
                    self.pulls[k] += (2 << self.round_number) * self.nbPlayers

                upper_confidence_bound, lower_confidence_bound = self.compute_ucb_lcb()

                reject = []
                accept = []

                for i, k in enumerate(self.active_arms):
                    better = np.sum(lower_confidence_bound > upper_confidence_bound[i])
                    worse = np.sum(upper_confidence_bound < lower_confidence_bound[i])
                    if better >= self.nbPlayers:
                        reject.append(k)
                        self.active_arms = np.setdiff1d(self.active_arms, k)
                        if self.verbose:
                            print("Player {} rejected arm {} at round {}".format(self.ext_rank, k, self.round_number))  # DEBUG
                    if worse >= (self.nbArms - self.nbPlayers):
                        accept.append(k)
                        self.active_arms = np.setdiff1d(self.active_arms, k)
                        if self.verbose:
                            print("Player {} accepted arm {} at round {}".format(self.ext_rank, k, self.round_number))  # DEBUG

                self.nbPlayers -= len(accept)
                self.nbArms -= (len(accept) + len(reject))

                if len(accept) > self.int_rank:
                    self.phase = State.Exploitation
                    if self.verbose:
                        print("Player {} starts exploiting arm {}".format(self.ext_rank, accept[self.int_rank]))
                    self.last_action = accept[self.int_rank]
                else:
                    self.phase = State.Exploration
                    self.int_rank -= len(accept)
                    self.last_action = self.active_arms[self.int_rank]
                    self.round_number += 1
                    self.last_phase_stats.fill(0)
                    self.t_phase = 0
        # End of all the cases
        self.t += 1

    def handleCollision(self, arm, reward=None):
        """ Handle a collision, on arm of index 'arm'. """
        assert reward is not None, "Error: a SIC_MMAB player got a collision on arm {} at time {} with reward = None but it should also see the reward.".format(arm, self.t)  # DEBUG
        if self.verbose:
            print("A SIC_MMAB player got a collision on arm {} at time {} with reward = {}.".format(arm, self.t, reward))  # DEBUG
        return self.getReward(arm, reward, collision=True)



# --- Class SIC_MMAB_UCB

class SIC_MMAB_UCB(SIC_MMAB):
    """ SIC_MMAB_UCB: SIC-MMAB with the simple UCB-1 confidence bounds."""

    def __str__(self):
        return r"SIC-MMAB(UCB, $T_0={}$)".format(self.Time0)

    def compute_ucb_lcb(self):
        r""" Compute the Upper-Confidence Bound and Lower-Confidence Bound for active arms, at the current time step.

        - :class:`SIC_MMAB_UCB` uses the simple UCB-1 confidence bounds:

        .. math::

            \mathrm{UCB}_k(t) &= \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{\alpha \log(t)}{2 N_k(t)}},\\
            \mathrm{LCB}_k(t) &= \frac{X_k(t)}{N_k(t)} - \sqrt{\frac{\alpha \log(t)}{2 N_k(t)}}.

        - Reference: [Auer et al. 02].

        - Other possibilities include UCB-H (the default, see :class:`SIC_MMAB`) and klUCB (see :class:`SIC_MMAB_klUCB`).
        """
        means = self.rewards[self.active_arms] / self.pulls[self.active_arms]
        bias = np.sqrt(self.alpha * np.log(self.t) / (2 * self.pulls[self.active_arms]))
        upper_confidence_bound = means + bias
        lower_confidence_bound = means - bias
        return upper_confidence_bound, lower_confidence_bound


# --- Class SIC_MMAB_klUCB

class SIC_MMAB_klUCB(SIC_MMAB):
    """ SIC_MMAB_klUCB: SIC-MMAB with the kl-UCB confidence bounds."""

    def __init__(self, nbArms, horizon,
            lower=0., amplitude=1.,
            alpha=4.0, verbose=False,
            tolerance=TOLERANCE, klucb=klucbBern, c=c,
        ):  # Named argument to give them in any order
        super(SIC_MMAB_klUCB, self).__init__(nbArms, horizon, lower=lower, amplitude=amplitude, alpha=alpha, verbose=verbose)
        self.c = c  #: Parameter c
        self.klucb = np.vectorize(klucb)  #: kl function to use
        self.klucb.__name__ = klucb.__name__
        self.tolerance = tolerance  #: Numerical tolerance

    def __str__(self):
        name = self.klucb.__name__[5:]
        if name == "Bern": name = ""
        complement = "{}{}".format(name, "" if self.c == 1 else r"$c={:.3g}$".format(self.c))
        if complement != "": complement = "({})".format(complement)
        name_of_kl = "kl-UCB{}".format(complement)
        return r"SIC-MMAB({}, $T_0={}$)".format(name_of_kl, self.Time0)

    def compute_ucb_lcb(self):
        r""" Compute the Upper-Confidence Bound and Lower-Confidence Bound for active arms, at the current time step.

        - :class:`SIC_MMAB_klUCB` uses the simple kl-UCB confidence bounds:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            \mathrm{UCB}_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c \log(t)}{N_k(t)} \right\},\\
            \mathrm{Biais}_k(t) &= \mathrm{UCB}_k(t) - \hat{\mu}_k(t),\\
            \mathrm{LCB}_k(t) &= \hat{\mu}_k(t) - \mathrm{Biais}_k(t).

        - If rewards are in :math:`[a, b]` (default to :math:`[0, 1]`) and :math:`\mathrm{kl}(x, y)` is the Kullback-Leibler divergence between two distributions of means x and y (see :mod:`Arms.kullback`),
        and c is the parameter (default to 1).

        - Reference: [Garivier & Cappé - COLT, 2011](https://arxiv.org/pdf/1102.2490.pdf).

        - Other possibilities include UCB-H (the default, see :class:`SIC_MMAB`) and klUCB (see :class:`SIC_MMAB_klUCB`).
        """
        rewards, pulls = self.rewards[self.active_arms], self.pulls[self.active_arms]
        means = rewards / pulls
        upper_confidence_bound = self.klucb(means, self.c * np.log(self.t) / pulls, self.tolerance)
        upper_confidence_bound[pulls < 1] = float('+inf')
        bias = upper_confidence_bound - means
        lower_confidence_bound = means - bias
        return upper_confidence_bound, lower_confidence_bound