# -*- coding: utf-8 -*-
""" Generic randomized index policy.

- Reference: [["On the Optimality of Perturbations in Stochastic and Adversarial Multi-armed Bandit Problems", by Baekjin Kim, Ambuj Tewari, arXiv:1902.00610]](https://arxiv.org/pdf/1902.00610.pdf)
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .IndexPolicy import IndexPolicy
except (ImportError, SystemError):
    from IndexPolicy import IndexPolicy


#: True to debug information about the perturbations
VERBOSE = True
VERBOSE = False


def uniform_perturbation(size=1, low=-1.0, high=1.0):
    r""" Uniform random perturbation, not from :math:`[0, 1]` but from :math:`[-1, 1]`, that is :math:`\mathcal{U}niform([-1, 1])`.

    - Reference: see Corollary 6 from [["On the Optimality of Perturbations in Stochastic and Adversarial Multi-armed Bandit Problems", by Baekjin Kim, Ambuj Tewari, arXiv:1902.00610]](https://arxiv.org/pdf/1902.00610.pdf)
    """
    return np.random.uniform(low=low, high=high, size=size)


def normal_perturbation(size=1, loc=0.0, scale=0.25):
    r""" Normal (Gaussian) random perturbation, with mean ``loc=0`` and scale (sigma2) ``scale=0.25`` (by default), that is :math:`\mathcal{N}ormal(loc, scale)`.

    - Reference: see Corollary 6 from [["On the Optimality of Perturbations in Stochastic and Adversarial Multi-armed Bandit Problems", by Baekjin Kim, Ambuj Tewari, arXiv:1902.00610]](https://arxiv.org/pdf/1902.00610.pdf)
    """
    return np.random.normal(loc=loc, scale=scale, size=size)

gaussian_perturbation = normal_perturbation


def exponential_perturbation(size=1, scale=0.25):
    r""" Exponential random perturbation, with parameter (:math:`\lambda`) ``scale=0.25`` (by default), that is :math:`\mathcal{E}xponential(\lambda)`.

    - Reference: see Corollary 7 from [["On the Optimality of Perturbations in Stochastic and Adversarial Multi-armed Bandit Problems", by Baekjin Kim, Ambuj Tewari, arXiv:1902.00610]](https://arxiv.org/pdf/1902.00610.pdf)
    """
    return np.random.exponential(scale=scale, size=size)


def gumbel_perturbation(size=1, loc=0.0, scale=0.25):
    r""" Gumbel random perturbation, with mean ``loc=0`` and scale ``scale=0.25`` (by default), that is :math:`\mathcal{G}umbel(loc, scale)`.

    - Reference: see Corollary 7 from [["On the Optimality of Perturbations in Stochastic and Adversarial Multi-armed Bandit Problems", by Baekjin Kim, Ambuj Tewari, arXiv:1902.00610]](https://arxiv.org/pdf/1902.00610.pdf)
    """
    return np.random.gumbel(loc=loc, scale=scale, size=size)


#: Map perturbation names (like ``"uniform"``) to perturbation functions (like :func:`uniform_perturbation`).
map_perturbation_str_to_function = {
    "uniform": uniform_perturbation,
    "normal": normal_perturbation,
    "gaussian": gaussian_perturbation,
    "exponential": exponential_perturbation,
    "gumbel": gumbel_perturbation,
}


class RandomizedIndexPolicy(IndexPolicy):
    """ Class that implements a generic randomized index policy."""

    def __init__(self, nbArms, perturbation="uniform", lower=0., amplitude=1., *args, **kwargs):
        """ New generic index policy.

        - nbArms: the number of arms,
        - perturbation: ["uniform", "normal", "exponential", "gaussian"] or a function like :func:`numpy.random.uniform`,
        - lower, amplitude: lower value and known amplitude of the rewards.
        """
        super(RandomizedIndexPolicy, self).__init__(nbArms, lower=lower, amplitude=amplitude, *args, **kwargs)
        if isinstance(perturbation, str):
            perturbation_name = perturbation
            perturbation = map_perturbation_str_to_function.get(perturbation_name, uniform_perturbation)
        else:
            perturbation_name = perturbation.__name__
        self.perturbation_name = perturbation_name  #: Name of the function to generate the random perturbation.
        self.perturbation = perturbation  #: Function to generate the random perturbation.

    def __str__(self):
        """ -> str"""
        return "{}({})".format(self.__class__.__name__, self.perturbation_name)

    # --- Basic choice() method

    def computeIndex(self, arm):
        r""" In a randomized index policy, with distribution :math:`\mathrm{Distribution}` generating perturbations :math:`Z_k(t)`, with index :math:`I_k(t)` and mean :math:`\hat{\mu}_k(t)` for each arm :math:`k`, it chooses an arm with maximal perturbated index (uniformly at random):

        .. math::
            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            Z_k(t) &\sim \mathrm{Distribution}, \\
            \mathrm{UCB}_k(t) &= I_k(t) - \hat{\mu}_k(t),\\
            A(t) &\sim U(\arg\max_{1 \leq k \leq K} \hat{\mu}_k(t) + \mathrm{UCB}_k(t) \cdot Z_k(t)).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        index = super(RandomizedIndexPolicy, self).computeIndex(arm)
        mean = self.rewards[arm] / self.pulls[arm]
        ucb = index - mean
        random_perturbation = self.perturbation()
        perturbated_index = mean + ucb * random_perturbation
        if VERBOSE:
            print("  - at time t = {}, policy {} would have used index = {} and mean = {}, but using its perturbation distribution ({}), it sampled a perturbation = {}, and the perturbated index was {} instead...".format(self.t, self, index, mean, self.perturbation_name, random_perturbation, perturbated_index))  # DEBUG
        self.index = perturbated_index

    def computeAllIndex(self):
        r""" In a randomized index policy, with distribution :math:`\mathrm{Distribution}` generating perturbations :math:`Z_k(t)`, with index :math:`I_k(t)` and mean :math:`\hat{\mu}_k(t)` for each arm :math:`k`, it chooses an arm with maximal perturbated index (uniformly at random):

        .. math::
            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            Z_k(t) &\sim \mathrm{Distribution}, \\
            \mathrm{UCB}_k(t) &= I_k(t) - \hat{\mu}_k(t),\\
            A(t) &\sim U(\arg\max_{1 \leq k \leq K} \hat{\mu}_k(t) + \mathrm{UCB}_k(t) \cdot Z_k(t)).
        """
        super(RandomizedIndexPolicy, self).computeAllIndex()
        index = self.index
        means = self.rewards / self.pulls
        ucb = index - means
        random_perturbations = self.perturbation(size=self.nbArms)
        for arm in range(self.nbArms):
            perturbated_index = means[arm] + ucb[arm] * random_perturbations[arm]
            self.index[arm] = perturbated_index
            if self.pulls[arm] < 1:
                self.index[arm] = float('+inf')
        if VERBOSE:
            print("  - at time t = {}, policy {} would have used indexes = {} and means = {}, but using its perturbation distribution ({}), it sampled perturbations = {}, and the perturbated indexes was {} instead...".format(self.t, self, index, means, self.perturbation_name, random_perturbations, self.index))  # DEBUG
