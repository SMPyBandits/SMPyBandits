# -*- coding: utf-8 -*-
""" Optimal Sampling for Structured Bandits (OSSB) algorithm.

- Reference: [[Minimal Exploration in Structured Stochastic Bandits, Combes et al, arXiv:1711.00400 [stat.ML]]](https://arxiv.org/abs/1711.00400)
- See also: https://github.com/SMPyBandits/SMPyBandits/issues/101

.. warning:: This is the simplified OSSB algorithm for classical bandits. It can be applied to more general bandit problems, see the original paper.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


from enum import Enum  # For the different phases
import numpy as np
from .BasePolicy import BasePolicy


from .kullback import klBern
klBern_vect = np.vectorize(klBern)


#: Different phases during the OSSB algorithm
Phase = Enum('Phase', ['initialisation', 'exploitation', 'estimation', 'exploration'])


#: Default value for the :math:`\varepsilon` parameter, 0.0 is a safe default.
EPSILON = 0.0


#: Default value for the :math:`\gamma` parameter, 0.0 is a safe default.
GAMMA = 0.0


def solve_optimization_problem(thetas):
    r""" Solve the optimization problem (2)-(3) as defined in the paper.

    - No need to solve anything, as they give the solution for classical bandits.
    """
    # values = np.zeros_like(thetas)
    # theta_max = np.max(thetas)
    # for i, theta in enumerate(thetas):
    #     if theta < theta_max:
    #         values[i] = 1 / klBern(theta, theta_max)
    # return values
    return 1. / klBern_vect(thetas, np.max(thetas))


class OSSB(BasePolicy):
    r""" Optimal Sampling for Structured Bandits (OSSB) algorithm.

    - Reference: [[Minimal Exploration in Structured Stochastic Bandits, Combes et al, arXiv:1711.00400 [stat.ML]]](https://arxiv.org/abs/1711.00400)
    """

    def __init__(self, nbArms, epsilon=EPSILON, gamma=GAMMA,
                 lower=0., amplitude=1.):
        super(OSSB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # Arguments
        assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for 'OSSB' class has to be 0 <= <= 1."  # DEBUG
        self.epsilon = epsilon
        assert gamma >= 0, "Error: the 'gamma' parameter for 'OSSB' class has to be >= 0."  # DEBUG
        self.gamma = gamma
        # Internal memory
        self.counter_s_no_exploitation_phase = 0
        self.phase = None

    def __str__(self):
        """ -> str"""
        return r"OSSB($\varepsilon={:.3g}$, $\gamma={:.3g}$)".format(self.epsilon, self.gamma)

    # --- Start game, and receive rewards

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        super(OSSB, self).startGame()
        self.counter_s_no_exploitation_phase = 0
        self.phase = Phase.initialisation

    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
        super(OSSB, self).getReward(arm, reward)

    # --- Basic choice() and handleCollision() method

    def choice(self):
        """ Applies the OSSB procedure, it's quite complicated so see the original paper."""
        means = (self.rewards / self.pulls)
        if np.any(self.pulls < 1):
            return np.random.choice(np.nonzero(self.pulls < 1)[0])

        values_c_x_mt = solve_optimization_problem(means)

        if np.all(self.pulls >= (1 + self.gamma) * np.log(self.t) * values_c_x_mt):
            self.phase = Phase.exploitation
            # self.counter_s_no_exploitation_phase += 0  # useless
            return np.random.choice(np.nonzero(means == np.max(means))[0])
        else:
            self.counter_s_no_exploitation_phase += 1
            # we don't just take argmin because of possible non-uniqueness
            least_explored = np.random.choice(np.nonzero(self.pulls == np.min(self.pulls))[0])
            ratios = self.pulls / values_c_x_mt
            least_probable = np.random.choice(np.nonzero(ratios == np.min(ratios))[0])

            if self.pulls[least_explored] <= self.epsilon * self.counter_s_no_exploitation_phase:
                self.phase = Phase.estimation
                return least_explored
            else:
                self.phase = Phase.exploration
                return least_probable

    # --- Others choice...() methods, partly implemented
    # FIXME write choiceWithRank, choiceFromSubSet, choiceMultiple also

    def handleCollision(self, arm, reward=None):
        """ Nothing special to do."""
        pass

