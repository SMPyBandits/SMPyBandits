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


from .kullback import klBern, klGauss
klBern_vect = np.vectorize(klBern)

# WARNING using np.vectorize gave weird result on klGauss
# klGauss_vect = np.vectorize(klGauss, excluded="y")
def klGauss_vect(xs, y, sig2x=0.25):
    return np.array([klGauss(x, y, sig2x) for x in xs])


#: Different phases during the OSSB algorithm
Phase = Enum('Phase', ['initialisation', 'exploitation', 'estimation', 'exploration'])


#: Default value for the :math:`\varepsilon` parameter, 0.0 is a safe default.
EPSILON = 0.0


#: Default value for the :math:`\gamma` parameter, 0.0 is a safe default.
GAMMA = 0.0


def solve_optimization_problem__classic(thetas):
    r""" Solve the optimization problem (2)-(3) as defined in the paper, for classical stochastic bandits.

    - No need to solve anything, as they give the solution for classical bandits.
    """
    # values = np.zeros_like(thetas)
    # theta_max = np.max(thetas)
    # for i, theta in enumerate(thetas):
    #     if theta < theta_max:
    #         values[i] = 1 / klBern(theta, theta_max)
    # return values
    return 1. / klBern_vect(thetas, np.max(thetas))


def solve_optimization_problem__gaussian(thetas):
    r""" Solve the optimization problem (2)-(3) as defined in the paper, for classical stochastic bandits.

    - No need to solve anything, as they give the solution for classical bandits.
    """
    # values = np.zeros_like(thetas)
    # theta_max = np.max(thetas)
    # for i, theta in enumerate(thetas):
    #     if theta < theta_max:
    #         values[i] = 1 / klGauss(theta, theta_max)
    # return values
    return 1. / klGauss_vect(thetas, np.max(thetas))


def solve_optimization_problem__sparse_bandits(thetas, sparsity=None):
    r""" Solve the optimization problem (2)-(3) as defined in the paper, for sparse stochastic bandits.

    - I recomputed suboptimal solution to the optimization problem, and found the same as in [["Sparse Stochastic Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)].
    """
    # print("Calling 'solve_optimization_problem__sparse_bandits' with thetas = {} and sparsity = {}...".format(thetas, sparsity))  # DEBUG

    d = len(thetas)
    if sparsity is None:
        sparsity = d
    permutation = np.argsort(thetas)[::-1]  # sort in decreasing order!
    # sorted_thetas = np.sort(thetas)
    sorted_thetas = thetas[permutation]
    best_theta = sorted_thetas[0]
    gaps = sorted_thetas - best_theta
    strong_sparsity = lambda k: (d - sparsity)/float(best_theta) - sum(gaps[i]/sorted_thetas[i]**2 for i in range(k, sparsity) if gaps[i] > 0)
    ci = np.zeros(d)

    # # DEBUG
    # print("    We have d =", d, "sparsity =", sparsity, "permutation =", permutation)  # DEBUG
    # print("    and sorted_thetas =", sorted_thetas, "with best_thetas =", sorted_thetas)  # DEBUG
    # print("    gaps =", gaps)  # DEBUG
    # for i in range(d):
    #     print("       for i =", i, "strong_sparsity(i) =", strong_sparsity(i))  # DEBUG
    # # DEBUG

    if strong_sparsity(0) > 0:
        # OK we have strong sparsity
        # print("Info: OK we have strong sparsity! With d = {} arms and s = {}, µ1 = {}, and (d-s)/µ1 - sum(Delta_i/µi²) = {:.3g} > 0...".format(d, sparsity, best_theta, strong_sparsity(0)))  # DEBUG

        for i in range(sparsity):
            if gaps[i] > 0:
                ci[permutation[i]] = 0.5 / min(gaps[i], sorted_thetas[i])
    else:
        # we only have weak sparsity... search for the k!
        k = None
        for possible_k in range(1, sparsity):
            if strong_sparsity(possible_k) < 0:
                k = possible_k
                break  # no need to continue the loop
        assert k is not None, "Error: there must exist a k in [1, s] such that (d-s)/µ1 - sum(Delta_i/µi², i=k...s) < 0..."  # DEBUG
        # print("Warning: we only have weak sparsity! With d = {} arms and s = {}, µ1 = {}, and (d-s)/µ1 - sum(Delta_i/µi², i=k={}...s) = {:.3g} < 0...".format(d, sparsity, best_theta, k, strong_sparsity(k)))  # DEBUG

        for i in range(k):
            if gaps[i] > 0:
                ci[permutation[i]] = 0.5 / min(gaps[i], sorted_thetas[i])
        for i in range(k, sparsity):
            # assert gaps[i] > 0, "Error: for i = {} >= k = {}, the gap should be > 0, but is {:.3g}...".format(i, k, gaps[i])  # XXX should be useless
            ci[permutation[i]] = 0.5 * (sorted_thetas[k] / (sorted_thetas[i] * gaps[i])) ** 2
        for i in range(sparsity, d):
            # assert gaps[i] > 0, "Error: for i = {} >= s = {}, the gap should be > 0, but is {:.3g}...".format(i, sparsity, gaps[i])  # XXX should be useless
            ci[permutation[i]] = 0.5 * (1 - (sorted_thetas[k] / gaps[k]) ** 2) / (gaps[i] * best_theta)
    # return the argmax ci of the optimization problem
    return ci


class OSSB(BasePolicy):
    r""" Optimal Sampling for Structured Bandits (OSSB) algorithm.

    - ``solve_optimization_problem`` can be ``"classic"`` or ``"bernoulli"`` for classic stochastic bandit with no structure, ``"gaussian"`` for classic bandit for Gaussian arms, or ``"sparse"`` for sparse stochastic bandit (give the sparsity ``s`` in a ``kwargs``).
    - Reference: [[Minimal Exploration in Structured Stochastic Bandits, Combes et al, arXiv:1711.00400 [stat.ML]]](https://arxiv.org/abs/1711.00400)
    """

    def __init__(self, nbArms, epsilon=EPSILON, gamma=GAMMA,
                 solve_optimization_problem="classic",
                 lower=0., amplitude=1., **kwargs):
        super(OSSB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # Arguments
        assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for 'OSSB' class has to be 0 <= . <= 1 but was {:.3g}.".format(epsilon)  # DEBUG
        self.epsilon = epsilon  #: Parameter :math:`\varepsilon` for the OSSB algorithm. Can be = 0.
        assert gamma >= 0, "Error: the 'gamma' parameter for 'OSSB' class has to be >= 0. but was {:.3g}.".format(gamma)  # DEBUG
        self.gamma = gamma  #: Parameter :math:`\gamma` for the OSSB algorithm. Can be = 0.
        # Solver for the optimization problem.
        self._solve_optimization_problem = solve_optimization_problem__classic  # Keep the function to use to solve the optimization problem
        self._info_on_solver = ", Bernoulli"  # small delta string

        # WARNING the option is a string to keep the configuration hashable and pickable
        if solve_optimization_problem == "sparse":
            self._info_on_solver = ", sparse Gaussian"
            self._solve_optimization_problem = solve_optimization_problem__sparse_bandits
        elif solve_optimization_problem == "gaussian":
            self._info_on_solver = ", Gaussian"
            self._solve_optimization_problem = solve_optimization_problem__gaussian
        self._kwargs = kwargs  # Keep in memory the other arguments, to give to self._solve_optimization_problem
        # Internal memory
        self.counter_s_no_exploitation_phase = 0  #: counter of number of exploitation phase
        # self.phase = None  #: categorical variable for the phase

    def __str__(self):
        """ -> str"""
        return r"OSSB($\varepsilon={:.3g}$, $\gamma={:.3g}${})".format(self.epsilon, self.gamma, self._info_on_solver)

    # --- Start game, and receive rewards

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        super(OSSB, self).startGame()
        self.counter_s_no_exploitation_phase = 0
        # self.phase = Phase.initialisation

    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
        super(OSSB, self).getReward(arm, reward)

    # --- Basic choice() and handleCollision() method

    def choice(self):
        """ Applies the OSSB procedure, it's quite complicated so see the original paper."""
        means = (self.rewards / self.pulls)
        if np.any(self.pulls < 1):
            return np.random.choice(np.nonzero(self.pulls < 1)[0])

        values_c_x_mt = self._solve_optimization_problem(means, **self._kwargs)

        if np.all(self.pulls >= (1. + self.gamma) * np.log(self.t) * values_c_x_mt):
            # self.phase = Phase.exploitation
            # self.counter_s_no_exploitation_phase += 0  # useless
            return np.random.choice(np.nonzero(means == np.max(means))[0])
        else:
            self.counter_s_no_exploitation_phase += 1
            # we don't just take argmin because of possible non-uniqueness
            least_explored = np.random.choice(np.nonzero(self.pulls == np.min(self.pulls))[0])
            ratios = self.pulls / values_c_x_mt
            least_probable = np.random.choice(np.nonzero(ratios == np.min(ratios))[0])

            if self.pulls[least_explored] <= self.epsilon * self.counter_s_no_exploitation_phase:
                # self.phase = Phase.estimation
                return least_explored
            else:
                # self.phase = Phase.exploration
                return least_probable

    # --- Others choice...() methods, partly implemented
    # FIXME write choiceWithRank, choiceFromSubSet, choiceMultiple also

    def handleCollision(self, arm, reward=None):
        """ Nothing special to do."""
        pass


class SparseOSSB(OSSB):
    r""" Optimal Sampling for Structured Bandits (OSSB) algorithm, for Sparse Stochastic Bandits. """

    def __init__(self, nbArms, epsilon=EPSILON, gamma=GAMMA, sparsity=None,
                 lower=0., amplitude=1., **kwargs):
        if sparsity is None or sparsity == nbArms:
            sparsity = nbArms
            print("Warning: regular OSSB should be used instead of SparseOSSB if 'sparsity' = 'nbArms' = {} ...".format(nbArms))  # DEBUG
        kwargs.update({'sparsity': sparsity})
        super(SparseOSSB, self).__init__(nbArms, epsilon=epsilon, gamma=gamma, solve_optimization_problem="sparse", lower=lower, amplitude=amplitude, **kwargs)
