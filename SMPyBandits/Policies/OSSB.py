# -*- coding: utf-8 -*-
""" Optimal Sampling for Structured Bandits (OSSB) algorithm.

- Reference: [[Minimal Exploration in Structured Stochastic Bandits, Combes et al, arXiv:1711.00400 [stat.ML]]](https://arxiv.org/abs/1711.00400)
- See also: https://github.com/SMPyBandits/SMPyBandits/issues/101

.. warning:: This is the simplified OSSB algorithm for classical bandits. It can be applied to more general bandit problems, see the original paper.

- The :class:`OSSB` is for Bernoulli stochastic bandits, and :class:`GaussianOSSB` is for Gaussian stochastic bandits, with a direct application of the result from their paper.
- The :class:`SparseOSSB` is for sparse Gaussian (or sub-Gaussian) stochastic bandits, of known variance.

- I also added support for non-constant :math:`\varepsilon` and :math:`\gamma` rates, as suggested in a talk given by Combes, 24th of May 2018, Rotterdam (Workshop, "Learning while Earning"). See :class:`OSSB_DecreasingRate` and :class:`OSSB_AutoDecreasingRate`.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


from enum import Enum  # For the different phases
import numpy as np

try:
    from .BasePolicy import BasePolicy
    from .kullback import klBern, klGauss
except ImportError:
    from BasePolicy import BasePolicy
    from kullback import klBern, klGauss
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


def solve_optimization_problem__gaussian(thetas, sig2x=0.25):
    r""" Solve the optimization problem (2)-(3) as defined in the paper, for Gaussian classical stochastic bandits.

    - No need to solve anything, as they give the solution for Gaussian classical bandits.
    """
    # values = np.zeros_like(thetas)
    # theta_max = np.max(thetas)
    # for i, theta in enumerate(thetas):
    #     if theta < theta_max:
    #         values[i] = 1 / klGauss(theta, theta_max)
    # return values
    return 1. / klGauss_vect(thetas, np.max(thetas), sig2x=sig2x)
    # # DEBUG
    # values = 1. / klGauss_vect(thetas, np.max(thetas), sig2x=sig2x)
    # print("solve_optimization_problem__gaussian({}, sig2x={}) gives {}...".format(thetas, sig2x, values))  # DEBUG
    # return values


def solve_optimization_problem__sparse_bandits(thetas, sparsity=None, only_strong_or_weak=False):
    r""" Solve the optimization problem (2)-(3) as defined in the paper, for sparse stochastic bandits.

    - I recomputed suboptimal solution to the optimization problem, and found the same as in [["Sparse Stochastic Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)].

    - If only_strong_or_weak is ``True``, the solution :math:`c_i` are not returned, but instead ``strong_or_weak, k`` is returned (to know if the problem is strongly sparse or not, and if not, the k that satisfy the required constraint).
    """
    # print("Calling 'solve_optimization_problem__sparse_bandits' with thetas = {} and sparsity = {}...".format(thetas, sparsity))  # DEBUG

    thetas = np.array(thetas)  # copy and force to be an array
    d = len(thetas)
    if sparsity is None:
        sparsity = d
    permutation = np.argsort(thetas)[::-1]  # sort in decreasing order!
    anti_permutation = [-1] * d
    for i in range(d):
        anti_permutation[permutation[i]] = i
    # sorted_thetas = np.sort(thetas)
    sorted_thetas = thetas[permutation]
    # assert list(np.sort(sorted_thetas)[::-1]) == list(sorted_thetas), "Error in the sorting of list thetas."  # DEBUG

    best_theta = sorted_thetas[0]
    gaps = best_theta - sorted_thetas
    # assert np.all(gaps >= 0), "Error in the computation of gaps = {}, they should be > 0.".format(gaps)  # DEBUG

    def strong_sparsity(k):
        left_term = (d - sparsity) / float(best_theta) if best_theta != 0 else 0
        right_term = 0
        for i in range(k, sparsity):
            right_term += gaps[i] / (sorted_thetas[i]**2) if sorted_thetas[i] != 0 else 0
        return left_term - right_term

    ci = np.zeros(d)

    # # DEBUG
    # print("    We have d =", d, "sparsity =", sparsity, "permutation =", permutation)  # DEBUG
    # print("    and sorted_thetas =", sorted_thetas, "with best_theta =", best_theta)  # DEBUG
    # print("    gaps =", gaps)  # DEBUG
    # # DEBUG

    if strong_sparsity(0) > 0:
        # print("       for k =", 0, "strong_sparsity(0) =", strong_sparsity(0))  # DEBUG
        # OK we have strong sparsity

        if only_strong_or_weak:
            print("Info: OK we have strong sparsity! With d = {} arms and s = {}, µ1 = {}, and (d-s)/µ1 - sum(Delta_i/µi²) = {:.3g} > 0...".format(d, sparsity, best_theta, strong_sparsity(0)))  # DEBUG
            return True, 0

        for i in range(1, sparsity):
            if gaps[i] > 0:
                ci[anti_permutation[i]] = 0.5 / min(gaps[i], sorted_thetas[i])
                # print("       for i =", i, "ci[", anti_permutation[i], "] =", ci[anti_permutation[i]])  # DEBUG
    else:
        # we only have weak sparsity... search for the good k
        k = None
        for possible_k in range(1, sparsity - 1):
            # print("       for k =", possible_k, "strong_sparsity(k) =", strong_sparsity(possible_k))  # DEBUG
            if strong_sparsity(possible_k) <= 0:
                k = possible_k
                break  # no need to continue the loop
        assert k is not None, "Error: there must exist a k in [1, s] such that (d-s)/µ1 - sum(Delta_i/µi², i=k...s) < 0..."  # DEBUG

        if only_strong_or_weak:
            print("Warning: we only have weak sparsity! With d = {} arms and s = {}, µ1 = {}, and (d-s)/µ1 - sum(Delta_i/µi², i=k={}...s) = {:.3g} < 0...".format(d, sparsity, best_theta, k, strong_sparsity(k)))  # DEBUG
            return False, k

        for i in range(1, k):
            if gaps[i] > 0:
                ci[anti_permutation[i]] = 0.5 / min(gaps[i], sorted_thetas[i])
        for i in range(k, sparsity):
            ci[anti_permutation[i]] = 0.5 * (sorted_thetas[k] / (sorted_thetas[i] * gaps[i])) ** 2
        for i in range(sparsity, d):
            ci[anti_permutation[i]] = 0.5 * (1 - (sorted_thetas[k] / gaps[k]) ** 2) / (gaps[i] * best_theta)

    # return the argmax ci of the optimization problem
    ci = np.maximum(0, ci)
    # print("So we have ci =", ci)  # DEBUG
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
        self._info_on_solver = ", Bern"  # small delta string

        # WARNING the option is a string to keep the configuration hashable and pickable
        if solve_optimization_problem == "sparse":
            # self._info_on_solver = ", sparse Gauss"  # XXX
            self._info_on_solver = ", sGauss"
            self._solve_optimization_problem = solve_optimization_problem__sparse_bandits
        elif solve_optimization_problem == "gaussian":
            self._info_on_solver = ", Gauss"
            self._solve_optimization_problem = solve_optimization_problem__gaussian
        self._kwargs = kwargs  # Keep in memory the other arguments, to give to self._solve_optimization_problem
        # Internal memory
        self.counter_s_no_exploitation_phase = 0  #: counter of number of exploitation phase
        self.phase = None  #: categorical variable for the phase

    def __str__(self):
        """ -> str"""
        return r"OSSB($\varepsilon={:.3g}$, $\gamma={:.3g}${})".format(self.epsilon, self.gamma, self._info_on_solver)

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
            # print("[initial phase] force exploration of an arm that was never pulled...")  # DEBUG
            return np.random.choice(np.nonzero(self.pulls < 1)[0])

        values_c_x_mt = self._solve_optimization_problem(means, **self._kwargs)

        if np.all(self.pulls >= (1. + self.gamma) * np.log(self.t) * values_c_x_mt):
            self.phase = Phase.exploitation
            # self.counter_s_no_exploitation_phase += 0  # useless
            chosen_arm = np.random.choice(np.nonzero(means == np.max(means))[0])
            # print("[exploitation phase] Choosing at random in the set of best arms {} at time t = {} : choice = {} ...".format(np.nonzero(means == np.max(means))[0], self.t, chosen_arm))  # DEBUG
            return chosen_arm
        else:
            self.counter_s_no_exploitation_phase += 1
            # we don't just take argmin because of possible non-uniqueness
            least_explored = np.random.choice(np.nonzero(self.pulls == np.min(self.pulls))[0])
            ratios = self.pulls / values_c_x_mt
            min_ratios_non_inf = np.nanmin(ratios[~np.isinf(ratios)])
            if np.isnan(min_ratios_non_inf):
                least_probable = np.random.choice(self.nbArms)
            else:
                least_probable = np.random.choice(np.nonzero(ratios == min_ratios_non_inf)[0])
            # print("Using ratio of pulls / values_c_x_mt = {}, and least probable arm(s) are {}...".format(ratios, least_probable))  # DEBUG

            if self.pulls[least_explored] <= self.epsilon * self.counter_s_no_exploitation_phase:
                self.phase = Phase.estimation
                # print("[estimation phase] Choosing the arm the least explored at time t = {} : choice = {} ...".format(self.t, least_explored))  # DEBUG
                return least_explored
            else:
                self.phase = Phase.exploration
                # print("[exploration phase] Choosing the arm the least probable at time t = {} : choice = {} ...".format(self.t, least_explored))  # DEBUG
                return least_probable

    # --- Others choice...() methods, partly implemented
    # TODO write choiceWithRank, choiceFromSubSet, choiceMultiple also

    def handleCollision(self, arm, reward=None):
        """ Nothing special to do."""
        pass


class GaussianOSSB(OSSB):
    r""" Optimal Sampling for Structured Bandits (OSSB) algorithm, for Gaussian Stochastic Bandits. """

    def __init__(self, nbArms, epsilon=EPSILON, gamma=GAMMA, variance=0.25,
                 lower=0., amplitude=1., **kwargs):
        kwargs.update({'sig2x': variance})
        super(GaussianOSSB, self).__init__(nbArms, epsilon=epsilon, gamma=gamma, solve_optimization_problem="gaussian", lower=lower, amplitude=amplitude, **kwargs)


class SparseOSSB(OSSB):
    r""" Optimal Sampling for Structured Bandits (OSSB) algorithm, for Sparse Stochastic Bandits. """

    def __init__(self, nbArms, epsilon=EPSILON, gamma=GAMMA, sparsity=None,
                 lower=0., amplitude=1., **kwargs):
        if sparsity is None or sparsity == nbArms:
            sparsity = nbArms
            print("Warning: regular OSSB should be used instead of SparseOSSB if 'sparsity' = 'nbArms' = {} ...".format(nbArms))  # DEBUG
        kwargs.update({'sparsity': sparsity})
        super(SparseOSSB, self).__init__(nbArms, epsilon=epsilon, gamma=gamma, solve_optimization_problem="sparse", lower=lower, amplitude=amplitude, **kwargs)
        self._info_on_solver += ", $s={}$".format(sparsity)


#: Default value for the constant for the decreasing rate
DECREASINGRATE = 1e-6


class OSSB_DecreasingRate(OSSB):
    r""" Optimal Sampling for Structured Bandits (OSSB) algorithm, with decreasing rates for both :math:`\varepsilon` and :math:`\gamma`.

    .. warning:: This is purely experimental, the paper does not talk about how to chose decreasing rates. It is inspired by the rates for Exp3 algorithm, cf [Bubeck & Cesa-Bianchi, 2012](http://sbubeck.com/SurveyBCB12.pdf).
    """

    def __init__(self, nbArms, epsilon=EPSILON, gamma=GAMMA, decreasingRate=DECREASINGRATE,
                 lower=0., amplitude=1., **kwargs):
        super(OSSB_DecreasingRate, self).__init__(nbArms, epsilon=epsilon, gamma=gamma, lower=lower, amplitude=amplitude, **kwargs)
        # new parameters
        self._decreasingRate = decreasingRate
        self._epsilon = epsilon
        self._gamma = gamma

    def __str__(self):
        """ -> str"""
        return r"OSSB(decreasing $\varepsilon(t),\gamma(t)${})".format(self._info_on_solver)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def epsilon(self):
        r"""Decreasing :math:`\varepsilon(t) = \min(1, \varepsilon_0 \exp(- t \tau))`."""
        return min(1, self._epsilon * np.exp(- self.t * self._decreasingRate))

    # https://docs.python.org/3/library/functions.html#property
    @epsilon.setter
    def epsilon(self, value):
        pass

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def gamma(self):
        r"""Decreasing :math:`\gamma(t) = \min(1, \gamma_0 \exp(- t \tau))`."""
        return min(1, self._gamma * np.exp(- self.t * self._decreasingRate))

    # https://docs.python.org/3/library/functions.html#property
    @gamma.setter
    def gamma(self, value):
        pass


class OSSB_AutoDecreasingRate(OSSB):
    r""" Optimal Sampling for Structured Bandits (OSSB) algorithm, with automatically-tuned decreasing rates for both :math:`\varepsilon` and :math:`\gamma`.

    .. warning:: This is purely experimental, the paper does not talk about how to chose decreasing rates. It is inspired by the rates for Exp3++ algorithm, [[One practical algorithm for both stochastic and adversarial bandits, S.Seldin & A.Slivkins, ICML, 2014](http://www.jmlr.org/proceedings/papers/v32/seldinb14-supp.pdf)].
    """

    def __init__(self, nbArms,
                 lower=0., amplitude=1., **kwargs):
        super(OSSB_AutoDecreasingRate, self).__init__(nbArms, epsilon=EPSILON, gamma=GAMMA, lower=lower, amplitude=amplitude, **kwargs)

    def __str__(self):
        """ -> str"""
        return r"OSSB(auto decreasing $\varepsilon(t),\gamma(t)${})".format(self._info_on_solver)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def epsilon(self):
        r"""Decreasing :math:`\varepsilon(t) = \frac{1}{2} \sqrt{\frac{\log(K)}{t K}}`."""
        return 0.5 * np.sqrt(np.log(self.nbArms) / (max(2, self.t) * self.nbArms))

    # https://docs.python.org/3/library/functions.html#property
    @epsilon.setter
    def epsilon(self, value):
        pass

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def gamma(self):
        r"""Decreasing :math:`\gamma(t) = \frac{1}{2} \sqrt{\frac{\log(K)}{t K}}`."""
        return 0.5 * np.sqrt(np.log(self.nbArms) / (max(2, self.t) * self.nbArms))

    # https://docs.python.org/3/library/functions.html#property
    @gamma.setter
    def gamma(self, value):
        pass

