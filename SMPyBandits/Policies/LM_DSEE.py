# -*- coding: utf-8 -*-
r""" The LM-DSEE policy for non-stationary bandits, from [["On Abruptly-Changing and Slowly-Varying Multiarmed Bandit Problems", by Lai Wei, Vaibhav Srivastava, 2018, arXiv:1802.08380]](https://arxiv.org/pdf/1802.08380)

- It uses an additional :math:`\mathcal{O}(\tau_\max)` memory for a game of maximum stationary length :math:`\tau_\max`.

.. warning:: This implementation is still experimental!
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


from enum import Enum  # For the different states
import numpy as np

try:
    from .BasePolicy import BasePolicy
except ImportError:
    from BasePolicy import BasePolicy


#: Different states during the LM-DSEE algorithm
State = Enum('State', ['Exploration', 'Exploitation'])

# --- Utility function

VERBOSE = True
#: Whether to be verbose when doing the search for valid parameter :math:`\ell`.
VERBOSE = False


def parameter_ell(a, N, b, gamma,
        verbose=VERBOSE, max_value_on_l=int(1e6)
    ):
    r""" Look for the smallest value of the parameter :math:`\ell` that satisfies the following equations:

    .. math:

        \mathrm{Bound}(l) &:= \frac{K}{a} \lceil \gamma \log(l b) \rceil,\\
        \mathrm{Bound}(l) &> 0,
        l &\leq \mathrm{Bound}(l).
    """
    if verbose: print("a = {}, N = {}, b = {}, gamma = {}".format(a, N, b, gamma))
    def bound(ell):
        return (N/a) * np.ceil(gamma * np.log(ell * b))
    ell = 1
    bound_ell = bound(ell)
    if verbose: print("ell = {} gives bound = {}".format(ell, bound_ell))
    while ell < max_value_on_l and not(ell >= bound_ell > 0):
        if verbose: print("ell = {} gives bound = {}".format(ell, bound_ell))
        ell += 1
        bound_ell = bound(ell)
    return ell


# --- Class


class LM_DSEE(BasePolicy):
    r""" The LM-DSEE policy for non-stationary bandits, from [["On Abruptly-Changing and Slowly-Varying Multiarmed Bandit Problems", by Lai Wei, Vaibhav Srivastava, 2018, arXiv:1802.08380]](https://arxiv.org/pdf/1802.08380)
    """

    def __init__(self, nbArms,
            nu=0.5, DeltaMin=0.5, a=1, b=0.25,
            *args, **kwargs
        ):
        super(LM_DSEE, self).__init__(nbArms, *args, **kwargs)
        # Parameters
        assert a > 0, "Error: for a LM_DSEE policy, the parameter 'a' should be > 0 but was = {}".format(a)  # DEBUG
        self.a = a  #: Parameter :math:`a` for the LM-DSEE algorithm.
        assert 0 < b <= 1, "Error: for a LM_DSEE policy, the parameter 'b' should be in (0, 1] but was = {}".format(b)  # DEBUG
        self.b = b  #: Parameter :math:`b` for the LM-DSEE algorithm.
        assert 0 < DeltaMin < 1, "Error: for a LM_DSEE policy, the parameter 'DeltaMin' should be in (0,1) but was = {}".format(DeltaMin)  # DEBUG
        gamma = 2 / DeltaMin**2
        self.l = parameter_ell(a, nbArms, b, gamma)  #: Parameter :math:`\ell` for the LM-DSEE algorithm, as computed by the function :func:`parameter_ell`.
        self.gamma = gamma  #: Parameter :math:`\gamma` for the LM-DSEE algorithm.
        assert 0 <= nu < 1, "Error: for a LM_DSEE policy, the parameter 'nu' should be in [0,1) but was = {}".format(nu)  # DEBUG
        rho = (1 - nu) / (1.0 + nu)
        self.rho = rho  #: Parameter :math:`\rho = \frac{1-\nu}{1+\nu}` for the LM-DSEE algorithm.
        # Internal memory
        self.phase = State.Exploration  #: Current phase, exploration or exploitation.
        self.current_exploration_arm = None  #: Currently explored arm.
        self.current_exploitation_arm = None  #: Currently exploited arm.
        self.batch_number = 1  #: Number of batch
        self.length_of_current_phase = None  #: Length of the current phase, either computed from :func:`length_exploration_phase` or func:`length_exploitation_phase`.
        self.step_of_current_phase = 0  #: Timer inside the current phase.
        self.all_rewards = [[] for _ in range(self.nbArms)]  #: Memory of all the rewards. A list per arm. Growing list until restart of that arm?

    def __str__(self):
        return r"LM-DSEE($\gamma={:.3g}$, $\rho={:.3g}$, $\ell={:.3g}$, $a={:.3g}$, $b={:.3g}$)".format(self.gamma, self.rho, self.l, self.a, self.b)

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        super(LM_DSEE, self).startGame()
        self.current_exploration_arm = None
        self.current_exploitation_arm = None
        self.batch_number = 1
        self.length_of_current_phase = None
        self.step_of_current_phase = 0
        self.all_rewards = [[] for _ in range(self.nbArms)]

    def length_exploration_phase(self, verbose=VERBOSE):
        r""" Compute the value of the current exploration phase:

        .. math:: L_1(k) = L(k) = \lceil \gamma \log(k^{\rho} l b)\rceil.

        .. warning:: I think there is a typo in the paper, as their formula are weird (like :math:`al` is defined from :math:`a`). See :func:`parameter_ell`.
        """
        value_Lk = self.gamma * np.log((self.batch_number**self.rho) * self.l * self.b)
        length = max(1, int(np.ceil(value_Lk)))
        if verbose: print("Length of exploration phase: computed to be = {} for batch number = {}...".format(length, self.batch_number))  # DEBUG
        return length

    def length_exploitation_phase(self, verbose=VERBOSE):
        r""" Compute the value of the current exploitation phase:

        .. math:: L_2(k) = \lceil a k^{\rho} l \rceil - K L_1(k).

        .. warning:: I think there is a typo in the paper, as their formula are weird (like :math:`al` is defined from :math:`a`). See :func:`parameter_ell`.
        """
        large_value = int(np.ceil(self.a * (self.batch_number**self.rho) * self.l))
        Lk = self.length_exploration_phase(verbose=False)
        length = max(1, large_value - self.nbArms * Lk)
        if verbose: print("Length of exploitation phase: computed to be = {} for batch number = {}...".format(length, self.batch_number))  # DEBUG
        return length

    def getReward(self, arm, reward):
        """ Get a reward from an arm."""
        super(LM_DSEE, self).getReward(arm, reward)
        reward = (reward - self.lower) / self.amplitude
        self.all_rewards[arm].append(reward)

    def choice(self):
        """ Choose an arm following the different phase of growing lenghts according to the LM-DSEE algorithm."""
        # print("For a {} policy: t = {}, current_exploration_arm = {}, current_exploitation_arm = {}, batch_number = {}, length_of_current_phase = {}, step_of_current_phase = {}".format(self, self.t, self.current_exploration_arm, self.current_exploitation_arm, self.batch_number, self.length_of_current_phase, self.step_of_current_phase))  # DEBUG
        # 1) exploration
        if self.phase == State.Exploration:
            # beginning of exploration phase
            if self.current_exploration_arm is None:
                self.current_exploration_arm = 0
            # if length of current exploration phase not computed, do it
            if self.length_of_current_phase is None:
                self.length_of_current_phase = self.length_exploration_phase()
            # if in a phase, do it
            if self.step_of_current_phase < self.length_of_current_phase:
                self.step_of_current_phase += 1
            else:  # done for this arm
                self.current_exploration_arm += 1  # go for next arm
                # if done for all the arms, go to exploitation
                if self.current_exploration_arm >= self.nbArms:
                    self.length_of_current_phase = None  # flag to start the next one
                    self.phase = State.Exploitation
                    self.step_of_current_phase = 0
                    self.current_exploration_arm = 0
                    # note that this last update might force to sample the arm 0 instead of arm K-1, once in a while...
            return self.current_exploration_arm
        # ---
        # 2) exploitation
        # ---
        elif self.phase == State.Exploitation:
            # beginning of exploitation phase
            if self.length_of_current_phase is None:
                self.length_of_current_phase = self.length_exploitation_phase()
            if self.current_exploitation_arm is None:
                # compute exploited arm
                mean_rewards = [np.mean(rewards_of_arm_k) for rewards_of_arm_k in self.all_rewards]
                pulls = [len(rewards_of_arm_k) for rewards_of_arm_k in self.all_rewards]
                j_epch_k = np.argmax(mean_rewards)
                print("A {} player at time {} and batch number {} observed the mean rewards = {} (for pulls {}) and will play {} for this exploitation phase.".format(self, self.t, self.batch_number, mean_rewards, pulls, j_epch_k))  # DEBUG
                self.current_exploitation_arm = j_epch_k
                # erase current memory
                self.all_rewards = [[] for _ in range(self.nbArms)]
            # if in a phase, do it
            if self.step_of_current_phase < self.length_of_current_phase:
                self.step_of_current_phase += 1
            # otherwise, reinitialize
            else:
                self.phase = State.Exploration
                self.length_of_current_phase = None  # flag to start the next one
                self.step_of_current_phase = 0
                self.current_exploration_arm = 0
                self.batch_number += 1
            return self.current_exploitation_arm
        else:
            raise ValueError("Error: LM_DSEE should only be in phase Exploration or Exploitation.")
