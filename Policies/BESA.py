# -*- coding: utf-8 -*-
""" The Best Empirical Sampled Average (BESA) algorithm.

- Reference: [[Sub-Sampling For Multi Armed Bandits, Baransi et al., 2014]](https://hal.archives-ouvertes.fr/hal-01025651)
- See also: https://github.com/Naereen/AlgoBandits/issues/103

.. warning:: This algorithm works very well but is quite peculiar. It sounds "too easy", so take a look to the article before wondering why it should work.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


import numpy as np
from .BasePolicy import BasePolicy


# --- Utility functions


def subsample_deterministic(n, m):
    r"""Returns :math:`\{1,dots,n\}` if :math:`n < m` or :math:`\{1,\dots,m\}` if :math:`n \geq m` (*ie*, it is :math:`\{1,\dots,\min(n,m)\}`).

    .. warning:: The BESA algorithm is efficient only with the random sub-sampling, don't use this one except for comparing.
    """
    return np.arange(min(n, m) + 1)


def subsample_uniform(n, m):
    r"""Returns a uniform sub-set of size :math:`n`, from :math:`\{1,dots, m\}`.

    .. note:: The BESA algorithm is efficient only with the random sub-sampling.
    """
    return np.random.choice(m + 1, size=n, replace=False)


# --- BESA core function, base case and recursive case


#: Numerical tolerance when comparing two means. Should not be zero!
TOLERANCE = 1e-4


def besa_two_actions(rewards, pulls, a, b, subsample_function=subsample_uniform):
    """ Core algorithm for the BESA selection, for two actions a and b:

    - N = min(Na, Nb),
    - Sub-sample N values from rewards of arm a, and N values from rewards of arm b,
    - Compute mean of both samples of size N, call them m_a, m_b,
    - If m_a > m_b, choose a,
    - Else if m_a < m_b, choose b,
    - And in case of a tie, break by choosing i such that Ni is minimal (or random [a, b] if Na=Nb).
    """
    # assert a != b, "Error: now need to call 'besa_two_actions' if a = = {} = b = {}...".format(a, b)  # DEBUG
    Na, Nb = pulls[a], pulls[b]
    N = min(Na, Nb)
    Ia = subsample_function(N, Na)
    # assert all(0 <= i <= Na for i in Ia), "Error: indexes in Ia should be between 0 and Na = {}".format(Na)  # DEBUG
    Ib = subsample_function(N, Nb)
    # assert all(0 <= i <= Nb for i in Ib), "Error: indexes in Ib should be between 0 and Nb = {}".format(Nb)  # DEBUG
    # assert len(Ia) == len(Ib) == N, "Error in subsample_function, Ia of size = {} and Ib of size = {} should have size N = {} ...".format(len(Ia), len(Ib), N)  # DEBUG
    sub_mean_a = np.mean(rewards[a, Ia])
    sub_mean_b = np.mean(rewards[b, Ib])
    # XXX I tested and these manual branching steps are the most efficient solution it is faster than using np.argmax()
    if sub_mean_a > (sub_mean_b + TOLERANCE):
        return a
    elif sub_mean_a < (sub_mean_b - TOLERANCE):
        return b
    else:  # 0 <= abs(sub_mean_a - sub_mean_b) <= TOLERANCE
        # WARNING warning about the numerical errors with float number...
        if Na < Nb:
            return a
        elif Na > Nb:
            return b
        else:  # if no way of breaking the tie, choose uniformly at random
            # FIXME this happens a lot! It's weird!
            return np.random.choice([a, b])
            # chosen_arm = np.random.choice([a, b])
            # print("Warning: arms a = {} and b = {} had same sub-samples means = {:.3g} = {:.3g} and nb selections = {} = {}... so choosing uniformly at random {}!".format(a, b, sub_mean_a, sub_mean_b, Na, Nb, chosen_arm))  # WARNING
            # return chosen_arm


def besa_K_actions(rewards, pulls, left, right, subsample_function=subsample_uniform, depth=0):
    r""" BESA recursive selection algorithm for an action set of size :math:`\mathcal{K} \geq 1`.

    - I prefer to implement for a discrete action set :math:`\{\text{left}, \dots, \text{right}\}` (end *included*) instead of a generic ``actions`` vector, to speed up the code, but it is less readable.
    - The depth argument is just for pretty printing debugging information (useless).
    """
    # assert left <= right, "Error: in 'besa_K_actions' function, left = {} was not <= right = {}...".format(left, right)  # DEBUG
    # print("In 'besa_K_actions', left = {} and right = {} for this call.".format(left, right))  # DEBUG
    if left == right:
        chosen_arm = left
    elif right == left + 1:
        chosen_arm = besa_two_actions(rewards, pulls, left, right, subsample_function=subsample_function)
    else:
        pivot = (left + right) // 2
        # print("Using pivot = {}, left = {} and right = {}...".format(pivot, left, right))  # DEBUG
        chosen_left = besa_K_actions(rewards, pulls, left, pivot, subsample_function=subsample_function, depth=depth+1)
        # assert left <= chosen_left <= pivot, "Error: the output chosen_left = {} from tournament from left = {} to pivot = {} should be between the two...".format(chosen_left, left, pivot)  # DEBUG
        chosen_right = besa_K_actions(rewards, pulls, pivot + 1, right, subsample_function=subsample_function, depth=depth+1)
        # assert pivot + 1 <= chosen_right <= right, "Error: the output chosen_right = {} from tournament from pivot + 1 = {} to right = {} should be between the two...".format(chosen_right, pivot + 1, right)  # DEBUG
        # print("The two recursive calls gave chosen_left = {}, chosen_right = {}...".format(chosen_left, chosen_right))  # DEBUG
        chosen_arm = besa_two_actions(rewards, pulls, chosen_left, chosen_right, subsample_function=subsample_function)
    # print("{}In 'besa_K_actions', left = {} and right = {} gave chosen_arm = {}.".format("\t" * depth, left, right, chosen_arm))  # DEBUG
    return chosen_arm


# --- The BESA policy


class BESA(BasePolicy):
    r""" The Best Empirical Sampled Average (BESA) algorithm.

    - Reference: [[Sub-Sampling For Multi Armed Bandits, Baransi et al., 2014]](https://arxiv.org/abs/1711.00400)
    """

    def __init__(self, nbArms, horizon, random_subsample=True, lower=0., amplitude=1.):
        super(BESA, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # --- Arguments
        # XXX find a solution to not need to horizon?
        self.horizon = horizon  #: Just to know the memory to allocate for rewards. It could be implemented without knowing the horizon, by using lists to keep all the reward history, but this would be way slower!
        self.random_subsample = random_subsample  #: Whether to use a deterministic or random sub-sampling procedure.
        self._subsample_function = subsample_uniform if random_subsample else subsample_deterministic
        # --- Internal memory
        assert nbArms >= 2, "Error: BESA algorithm can only work for at least 2 arms."
        self._left = 0  # just keep them in memory to increase readability
        self._right = nbArms - 1  # just keep them in memory to increase readability
        self.all_rewards = np.zeros((nbArms, horizon))  #: Keep **all** rewards of each arms

    def __str__(self):
        """ -> str"""
        return "BESA{}".format("(non-random subsample)" if not self.random_subsample else "")

    def getReward(self, arm, reward):
        """ Add the current reward in the global history.

        .. note:: There is no need to normalize the reward in [0,1], that's one of the strong point of the BESA algorithm."""
        # XXX find a solution to not need to horizon?
        self.all_rewards[arm, self.t] = reward
        super(BESA, self).getReward(arm, reward)

    # --- Basic choice() and handleCollision() method

    def choice(self):
        """ Applies the BESA procedure with the current data history."""
        # if some arm has never been selected, force to explore it!
        if np.any(self.pulls < 1):
            return np.random.choice(np.arange(self.nbArms)[self.pulls < 1])
        else:
            return besa_K_actions(self.all_rewards, self.pulls, self._left, self._right, subsample_function=self._subsample_function, depth=0)

    # --- Others choice...() methods, partly implemented
    # FIXME write choiceWithRank, choiceFromSubSet, choiceMultiple also

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
