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


#: Different states during the Musical Chair algorithm
State = Enum('State', ['Exploration', 'Exploitation'])


# --- Class

class LM_DSEE(BasePolicy):
    r""" The LM-DSEE policy for non-stationary bandits, from [["On Abruptly-Changing and Slowly-Varying Multiarmed Bandit Problems", by Lai Wei, Vaibhav Srivastava, 2018, arXiv:1802.08380]](https://arxiv.org/pdf/1802.08380)
    """

    def __init__(self, nbArms,
            nu=0.05, DeltaMin=0.5, a=1, b=2,
            lower=0., amplitude=1., *args, **kwargs
        ):
        super(LM_DSEE, self).__init__(nbArms, lower=lower, amplitude=amplitude, *args, **kwargs)
        # Parameters
        self.a = a  #: Parameter :math:`a` for the LM-DSEE algorithm.
        self.b = b  #: Parameter :math:`b` for the LM-DSEE algorithm.
        assert 0 < DeltaMin < 1, "Error: for a LM_DSEE policy, the parameter 'DeltaMin' should be in (0,1) but was = {}".format(DeltaMin)  # DEBUG
        gamma = 2 / DeltaMin**2
        self.l = nbArms * np.ceil(gamma * np.log(b)) / a  #: Parameter :math:`\ell` for the LM-DSEE algorithm.
        self.gamma = gamma  #: Parameter :math:`\gamma` for the LM-DSEE algorithm.
        assert 0 < nu < 1, "Error: for a LM_DSEE policy, the parameter 'nu' should be in (0,1) but was = {}".format(nu)  # DEBUG
        rho = (1 - nu) / (1 + nu)
        self.rho = rho  #: Parameter :math:`\rho` for the LM-DSEE algorithm.
        # Internal memory
        self.t = 0  #: Internal timer
        self.phase = State.Exploration  #: Current phase, exploration or exploitation
        self.current_exploration_arm = None  #: Currently explored arm
        self.current_exploitation_arm = None  #: Currently exploited arm
        self.batch_number = 1  #: Number of batch
        self.length_of_current_phase = None  #: Length of the current phase, either computed from :func:`length_exploration_phase` or func:length_exploitation_phase``
        self.step_of_current_phase = 0  #: Timer inside the current phase
        self.all_rewards = [[] for _ in range(self.nbArms)]

    def __str__(self):
        return r"LM-DSEE($\gamma={:.3g}$, $\rho={:.3g}$, $l={:.3g}$, $b={:.3g}$)".format(self.gamma, self.rho, self.l, self.b)

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        super(LM_DSEE, self).startGame()
        self.t = 0
        self.current_exploration_arm = None
        self.current_exploitation_arm = None
        self.batch_number = 1
        self.length_of_current_phase = None
        self.step_of_current_phase = 0
        self.all_rewards = [[] for _ in range(self.nbArms)]

    def length_exploration_phase(self):
        r""" Compute the value of the current exploration phase:

        .. math:: L(k) = \lceil \gamma \log(k^{\rho} l b)\rceil.
        """
        value_Lk = self.gamma * np.log(self.batch_number**self.rho * self.l * self.b)
        return int(np.ceil(value_Lk))

    def length_exploitation_phase(self):
        r""" Compute the value of the current exploitation phase:

        .. math:: \lceil a k^{\rho} l\rceil - K L(k).
        """
        large_value = int(np.ceil(self.a * self.batch_number**self.rho * self.l))
        Lk = self.length_exploration_phase()
        return large_value - self.nbArms * Lk

    def getReward(self, arm, reward):
        """ Get a reward from an arm."""
        super(LM_DSEE, self).getReward(arm, reward)
        # 1) exploration
        if self.phase == State.Exploration:
            reward = (reward - self.lower) / self.amplitude
            self.all_rewards[arm].append(reward)

    def choice(self):
        """ Chose an arm following the different phase of growing lenghts according to the LM-DSEE algorithm."""
        # print("For a {} policy: t = {}, current_exploration_arm = {}, current_exploitation_arm = {}, batch_number = {}, length_of_current_phase = {}, step_of_current_phase = {}".format(self, self.t, self.current_exploration_arm, self.current_exploitation_arm, self.batch_number, self.length_of_current_phase, self.step_of_current_phase))  # DEBUG
        # 1) exploration
        if self.phase == State.Exploration:
            # beginning of explore phase
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
            return self.current_exploration_arm
        # ---
        # 2) exploitation
        # ---
        elif self.phase == State.Exploitation:
            if self.length_of_current_phase is None:
                # start exploitation
                self.length_of_current_phase = self.length_exploitation_phase()
            if self.current_exploitation_arm is None:
                # compute exploited arm
                mean_rewards = [np.mean(rewards_of_arm_k) for rewards_of_arm_k in self.all_rewards]
                j_epch_k = np.argmax(mean_rewards)
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
                self.batch_number += 1
            return self.current_exploitation_arm
        else:
            raise ValueError("Error: LM_DSEE should only be in phase Exploration or Exploitation.")


