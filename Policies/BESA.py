# -*- coding: utf-8 -*-
""" The Best Empirical Sampled Average (BESA) algorithm.

- Reference: [[Sub-Sampling For Multi Armed Bandits, Baransi et al., 2014]](https://hal.archives-ouvertes.fr/hal-01025651)
- See also: https://github.com/Naereen/AlgoBandits/issues/103

.. warning:: TODO Still a work in progress, I need to conclude this.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


import numpy as np
from .BasePolicy import BasePolicy


def subsample_deterministic(n, m):
    r"""Returns :math:`\{1,dots,n\}` if :math:`n < m` or :math:`\{1,\dots,m\}` if :math:`n \geq m` (*ie*, it is :math:`\{1,\dots,\min(n,m)\}`)."""
    return np.arange(min(n, m))


def subsample_uniform(n, m):
    r"""Returns :math:`\{1,dots,n\}` if :math:`n < m` or :math:`\{1,\dots,m\}` if :math:`n \geq m` (*ie*, it is :math:`\{1,\dots,\min(n,m)\}`)."""
    nb_samples = min(n, m)
    size_of_set_to_sample_from = max(n, m)
    return np.random.choice(nb_samples, size=size_of_set_to_sample_from, replace=False)


def besa_two_actions(rewards, pulls, a, b, random_subsample=False):
    """ Core algorithm for the BESA selection, for two actions a and b:

    - N = min(Na, Nb),
    - Sub-sample N values from rewards of arm a, and N values from rewards of arm b,
    - Compute mean of both samples of size N, call them m_a, m_b,
    - If m_a > m_b, choose a,
    - Else if m_a < m_b, choose b,
    - And in case of a tie, break by choosing i such that Ni is minimal (or random [a, b] if Na=Nb).
    """
    assert a != b, "Error: now need to call 'besa_two_actions' if a = = {} = b = {}...".format(a, b)  # DEBUG
    Na, Nb = pulls[a], pulls[b]
    N = min(Na, Nb)
    # print("N =", N)  # DEBUG
    if random_subsample:
        Ia = subsample_uniform(N, Na)
        Ib = subsample_uniform(N, Nb)
    else:
        Ia = subsample_deterministic(N, Na)
        Ib = subsample_deterministic(N, Nb)
    # print("Ia =", repr(Ia))  # DEBUG
    # print("Ib =", repr(Ib))  # DEBUG
    sub_mean_a = np.mean(rewards[a, Ia])
    # print("sub_mean_a =", sub_mean_a)  # DEBUG
    sub_mean_b = np.mean(rewards[b, Ib])
    # print("sub_mean_b =", sub_mean_b)  # DEBUG
    # XXX I tested and these manual branching steps are the most efficient solution
    # it is faster than using np.argmax()
    if sub_mean_a > sub_mean_b:
        return a
    elif sub_mean_a < sub_mean_b:
        return b
    else:
        if Na < Nb:
            return a
        elif Na > Nb:
            return b
        else:  # if no way of breaking the tie, choose uniformly at random
            return np.random.choice([a, b])


def besa_K_actions(rewards, pulls, left, right, random_subsample=False, depth=0):
    r"""BESA recursive selection algorithm for an action set of size :math:`\mathcal{K} \geq 1`."""
    assert left <= right, "Error: in 'besa_K_actions' function, left = {} was not <= right = {}...".format(left, right)  # DEBUG
    # print("In 'besa_K_actions', left = {} and right = {} for this call.".format(left, right))  # DEBUG
    if left == right:
        chosen_arm = left
    elif right == left + 1:
        chosen_arm = besa_two_actions(rewards, pulls, left, right, random_subsample=random_subsample)
    else:
        pivot = (right + left) // 2
        # print("Using pivot = {}, left = {} and right = {}...".format(pivot, left, right))  # DEBUG
        chosen_left = besa_K_actions(rewards, pulls, left, pivot, random_subsample=random_subsample, depth=depth+1)
        chosen_right = besa_K_actions(rewards, pulls, pivot + 1, right, random_subsample=random_subsample, depth=depth+1)
        # print("The two recursive calls gave chosen_left = {}, chosen_right = {}...".format(chosen_left, chosen_right))  # DEBUG
        chosen_arm = besa_two_actions(rewards, pulls, chosen_left, chosen_right, random_subsample=random_subsample)
    print("{}In 'besa_K_actions', left = {} and right = {} gave chosen_arm = {}.".format("\t" * depth, left, right, chosen_arm))  # DEBUG
    return chosen_arm


class BESA(BasePolicy):
    r""" The Best Empirical Sampled Average (BESA) algorithm.

    - Reference: [[Sub-Sampling For Multi Armed Bandits, Baransi et al., 2014]](https://arxiv.org/abs/1711.00400)
    """

    def __init__(self, nbArms, horizon, random_subsample=False,
                 lower=0., amplitude=1.):
        super(BESA, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # Arguments
        self.horizon = horizon  #: Just to know the memory to allocate for rewards
        self.random_subsample = random_subsample  #: Whether to use a deterministic or random sub-sampling procedure.
        # Internal memory
        assert nbArms > 1, "Error: BESA algorithm can only work for more than 2 arms."
        self._left = 0
        self._right = nbArms - 1
        self.all_rewards = np.zeros((nbArms, horizon))  #: Keep **all** rewards of each arms

    def __str__(self):
        """ -> str"""
        return "BESA{}".format("(random subsample)" if self.random_subsample else "")

    def getReward(self, arm, reward):
        """ Count the reward in the global history.

        .. note:: There is no need to renormalize the reward in [0,1], that's one of the strong point of the BESA algorithm."""
        self.all_rewards[arm, self.t] = reward
        super(BESA, self).getReward(arm, reward)

    # --- Basic choice() and handleCollision() method

    def choice(self):
        """ Applies the BESA procedure, it's quite easy."""
        if not np.all(self.pulls >= 1):
            return np.random.randint(self.nbArms)
        else:
            return besa_K_actions(self.all_rewards, self.pulls, self._left, self._right, random_subsample=self.random_subsample, depth=0)

    # --- Others choice...() methods, partly implemented

    def estimatedOrder(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means.

        - For a base policy, it is completely random.
        """
        means = self.rewards / self.pulls
        means[self.pulls < 1] = float('+inf')
        return np.argsort(means)

    def handleCollision(self, arm, reward=None):
        """ Nothing special to do."""
        pass

