# -*- coding: utf-8 -*-
""" The Best Empirical Sampled Average (BESA) algorithm.

- Reference: [[Sub-Sampling For Multi Armed Bandits, Baransi et al., 2014]](https://hal.archives-ouvertes.fr/hal-01025651)
- See also: https://github.com/SMPyBandits/SMPyBandits/issues/103 and https://github.com/SMPyBandits/SMPyBandits/issues/116

.. warning:: This algorithm works VERY well but it is looks weird at first sight. It sounds "too easy", so take a look to the article before wondering why it should work.

.. note:: Right now, it is between 10 and 25 times slower than :class:`Policies.klUCB` and other single-player policies.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


import numpy as np
from .IndexPolicy import IndexPolicy


# --- Utility functions


def subsample_deterministic(n, m):
    r"""Returns :math:`\{1,\dots,n\}` if :math:`n < m` or :math:`\{1,\dots,m\}` if :math:`n \geq m` (*ie*, it is :math:`\{1,\dots,\min(n,m)\}`).

    .. warning:: The BESA algorithm is efficient only with the random sub-sampling, don't use this one except for comparing.

    >>> subsample_deterministic(5, 3)  # doctest: +ELLIPSIS
    array([0, 1, 2, 3])
    >>> subsample_deterministic(10, 20)  # doctest: +ELLIPSIS
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
    """
    return np.arange(min(n, m) + 1)


def subsample_uniform(n, m):
    r"""Returns a uniform sub-set of size :math:`n`, from :math:`\{1,dots, m\}`.

    - Fails if n > m.

    .. note:: The BESA algorithm is efficient only with the random sub-sampling.

    >>> np.random.seed(1234)  # reproducible results
    >>> subsample_uniform(3, 5)  # doctest: +ELLIPSIS
    array([4, 0, 1])
    >>> subsample_uniform(10, 20)  # doctest: +ELLIPSIS
    array([ 7, 16,  2,  3,  1, 18,  5,  4,  0,  8])
    """
    return np.random.choice(m, size=n, replace=False)


# --- BESA core function, base case and recursive case


#: Numerical tolerance when comparing two means. Should not be zero!
TOLERANCE = 1e-6


def inverse_permutation(permutation, j):
    """ Inverse the permutation for given input j, that is, it finds i such that p[i] = j.

    >>> permutation = [1, 0, 3, 2]
    >>> inverse_permutation(permutation, 1)
    0
    >>> inverse_permutation(permutation, 0)
    1
    """
    for i, pi in enumerate(permutation):
        if pi == j:
            return i
    raise ValueError("inverse_permutation({}, {}) failed.".format(permutation, j))


def besa_two_actions(rewards, pulls, a, b, subsample_function=subsample_uniform):
    """ Core algorithm for the BESA selection, for two actions a and b:

    - N = min(Na, Nb),
    - Sub-sample N values from rewards of arm a, and N values from rewards of arm b,
    - Compute mean of both samples of size N, call them m_a, m_b,
    - If m_a > m_b, choose a,
    - Else if m_a < m_b, choose b,
    - And in case of a tie, break by choosing i such that Ni is minimal (or random [a, b] if Na=Nb).

    .. note:: ``rewards`` can be a numpy array of shape (at least) ``(nbArms, max(Na, Nb))`` or a dictionary maping ``a,b`` to lists (or iterators) of lengths ``>= max(Na, Nb)``.

    >>> np.random.seed(2345)  # reproducible results
    >>> pulls = [6, 10]; K = len(pulls); N = max(pulls)
    >>> rewards = np.random.randn(K, N)
    >>> np.mean(rewards, axis=1)  # arm 1 is better
    >>> np.mean(rewards[:, :min(pulls)], axis=1)  # arm 0 is better in the first 6 samples
    >>> besa_two_actions(rewards, pulls, 0, 1, subsample_function=subsample_deterministic)  # doctest: +ELLIPSIS
    0
    >>> [besa_two_actions(rewards, pulls, 0, 1, subsample_function=subsample_uniform) for _ in range(10)]  # doctest: +ELLIPSIS
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    """
    if a == b:
        print("Error: no need to call 'besa_two_actions' if a = = {} = b = {}...".format(a, b))  # DEBUG
        return a
    Na, Nb = pulls[a], pulls[b]
    N = min(Na, Nb)
    Ia = subsample_function(N, Na)
    # assert all(0 <= i < Na for i in Ia), "Error: indexes in Ia should be between 0 and Na = {}".format(Na)  # DEBUG
    Ib = subsample_function(N, Nb)
    # assert all(0 <= i < Nb for i in Ib), "Error: indexes in Ib should be between 0 and Nb = {}".format(Nb)  # DEBUG
    # assert len(Ia) == len(Ib) == N, "Error in subsample_function, Ia of size = {} and Ib of size = {} should have size N = {} ...".format(len(Ia), len(Ib), N)  # DEBUG
    # Compute sub means
    if isinstance(rewards, np.ndarray):  # faster to compute this
        sub_mean_a = np.sum(rewards[a, Ia]) / N
        sub_mean_b = np.sum(rewards[b, Ib]) / N
    else:  # than this for other data type (eg. dict mapping int to list)
        sub_mean_a = sum(rewards[a][i] for i in Ia) / N
        sub_mean_b = sum(rewards[b][i] for i in Ib) / N
    # assert 0 <= min(sub_mean_a, sub_mean_b) <= max(sub_mean_a, sub_mean_b) <= 1
    # XXX I tested and these manual branching steps are the most efficient solution it is faster than using np.argmax()
    if sub_mean_a > (sub_mean_b + TOLERANCE):
        return a
    elif sub_mean_b > (sub_mean_a + TOLERANCE):
        return b
    else:  # 0 <= abs(sub_mean_a - sub_mean_b) <= TOLERANCE
        if Na < Nb:
            return a
        elif Na > Nb:
            return b
        else:  # if no way of breaking the tie, choose uniformly at random
            return np.random.choice([a, b])
            # chosen_arm = np.random.choice([a, b])
            # print("Warning: arms a = {} and b = {} had same sub-samples means = {:.3g} = {:.3g} and nb selections = {} = {}... so choosing uniformly at random {}!".format(a, b, sub_mean_a, sub_mean_b, Na, Nb, chosen_arm))  # WARNING
            # return chosen_arm


def besa_K_actions__non_randomized(rewards, pulls, left, right, subsample_function=subsample_uniform, depth=0):
    r""" BESA recursive selection algorithm for an action set of size :math:`\mathcal{K} \geq 1`.

    - I prefer to implement for a discrete action set :math:`\{\text{left}, \dots, \text{right}\}` (end *included*) instead of a generic ``actions`` vector, to speed up the code, but it is less readable.
    - The depth argument is just for pretty printing debugging information (useless).

    .. warning:: The binary tournament is NOT RANDOMIZED here, this version is only for testing.

    >>> np.random.seed(1234)  # reproducible results
    >>> pulls = [5, 6, 7, 8]; K = len(pulls); N = max(pulls)
    >>> rewards = np.random.randn(K, N)
    >>> np.mean(rewards, axis=1)  # arm 0 is better
    array([ 0.09876921, -0.18561207,  0.04463033,  0.0653539 ])
    >>> np.mean(rewards[:, :min(pulls)], axis=1)  # arm 1 is better in the first 6 samples
    array([-0.06401484,  0.17366346,  0.05323033, -0.09514708])
    >>> besa_K_actions__non_randomized(rewards, pulls, 0, K-1, subsample_function=subsample_deterministic)  # doctest: +ELLIPSIS
    3
    >>> [besa_K_actions__non_randomized(rewards, pulls, 0, K-1, subsample_function=subsample_uniform) for _ in range(10)]  # doctest: +ELLIPSIS
    [0, 3, 3, 0, 3, 0, 0, 0, 3, 1]
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
        chosen_left = besa_K_actions__non_randomized(rewards, pulls, left, pivot, subsample_function=subsample_function, depth=depth+1)
        # assert left <= chosen_left <= pivot, "Error: the output chosen_left = {} from tournament from left = {} to pivot = {} should be between the two...".format(chosen_left, left, pivot)  # DEBUG
        chosen_right = besa_K_actions__non_randomized(rewards, pulls, pivot + 1, right, subsample_function=subsample_function, depth=depth+1)
        # assert pivot + 1 <= chosen_right <= right, "Error: the output chosen_right = {} from tournament from pivot + 1 = {} to right = {} should be between the two...".format(chosen_right, pivot + 1, right)  # DEBUG
        # print("The two recursive calls gave chosen_left = {}, chosen_right = {}...".format(chosen_left, chosen_right))  # DEBUG
        chosen_arm = besa_two_actions(rewards, pulls, chosen_left, chosen_right, subsample_function=subsample_function)
    # print("{}In 'besa_K_actions', left = {} and right = {} gave chosen_arm = {}.".format("\t" * depth, left, right, chosen_arm))  # DEBUG
    return chosen_arm


def besa_K_actions__smart_divideandconquer(rewards, pulls, left, right, random_permutation_of_arm=None, subsample_function=subsample_uniform, depth=0):
    r""" BESA recursive selection algorithm for an action set of size :math:`\mathcal{K} \geq 1`.

    - I prefer to implement for a discrete action set :math:`\{\text{left}, \dots, \text{right}\}` (end *included*) instead of a generic ``actions`` vector, to speed up the code, but it is less readable.
    - The depth argument is just for pretty printing debugging information (useless).

    .. note:: The binary tournament is RANDOMIZED here, as it should be.

    >>> np.random.seed(1234)  # reproducible results
    >>> pulls = [5, 6, 7, 8]; K = len(pulls); N = max(pulls)
    >>> rewards = np.random.randn(K, N)
    >>> np.mean(rewards, axis=1)  # arm 0 is better
    array([ 0.09876921, -0.18561207,  0.04463033,  0.0653539 ])
    >>> np.mean(rewards[:, :min(pulls)], axis=1)  # arm 1 is better in the first 6 samples
    array([-0.06401484,  0.17366346,  0.05323033, -0.09514708])
    >>> besa_K_actions__smart_divideandconquer(rewards, pulls, 0, K-1, subsample_function=subsample_deterministic)  # doctest: +ELLIPSIS
    3
    >>> [besa_K_actions__smart_divideandconquer(rewards, pulls, 0, K-1, subsample_function=subsample_uniform) for _ in range(10)]  # doctest: +ELLIPSIS
    [3, 3, 2, 3, 3, 0, 0, 0, 2, 3]
    """
    # assert left <= right, "Error: in 'besa_K_actions__smart_divideandconquer' function, left = {} was not <= right = {}...".format(left, right)  # DEBUG
    # print("In 'besa_K_actions__smart_divideandconquer', left = {} and right = {} for this call.".format(left, right))  # DEBUG
    if left == right:
        chosen_arm = left
    elif right == left + 1:
        chosen_arm = besa_two_actions(rewards, pulls, left, right, subsample_function=subsample_function)
    else:
        pivot = (left + right) // 2
        # print("Using pivot = {}, left = {} and right = {}...".format(pivot, left, right))  # DEBUG
        chosen_left = besa_K_actions__smart_divideandconquer(rewards, pulls, left, pivot, random_permutation_of_arm=random_permutation_of_arm, subsample_function=subsample_function, depth=depth+1)
        # chosen_left = inverse_permutation(random_permutation_of_arm, chosen_left)
        # assert left <= chosen_left <= pivot, "Error: the output chosen_left = {} from tournament from left = {} to pivot = {} should be between the two...".format(chosen_left, left, pivot)  # DEBUG
        chosen_right = besa_K_actions__smart_divideandconquer(rewards, pulls, pivot + 1, right, random_permutation_of_arm=random_permutation_of_arm, subsample_function=subsample_function, depth=depth+1)
        # chosen_right = inverse_permutation(random_permutation_of_arm, chosen_right)
        # assert pivot + 1 <= chosen_right <= right, "Error: the output chosen_right = {} from tournament from pivot + 1 = {} to right = {} should be between the two...".format(chosen_right, pivot + 1, right)  # DEBUG
        # print("The two recursive calls gave chosen_left = {}, chosen_right = {}...".format(chosen_left, chosen_right))  # DEBUG
        if random_permutation_of_arm is not None:
            chosen_left, chosen_right = random_permutation_of_arm[chosen_left], random_permutation_of_arm[chosen_right]
        chosen_arm = besa_two_actions(rewards, pulls, chosen_left, chosen_right, subsample_function=subsample_function)
    # print("{}In 'besa_K_actions__smart_divideandconquer', left = {} and right = {} gave chosen_arm = {}.".format("\t" * depth, left, right, chosen_arm))  # DEBUG
    if random_permutation_of_arm is not None:
        return inverse_permutation(random_permutation_of_arm, chosen_arm)
    else:
        return chosen_arm


def besa_K_actions(rewards, pulls, actions, subsample_function=subsample_uniform, depth=0):
    r""" BESA recursive selection algorithm for an action set of size :math:`\mathcal{K} \geq 1`.

    - The divide and conquer is implemented for a generic list of actions, it's slower but simpler to write! Left and right divisions are just ``actions[:len(actions)//2]`` and ``actions[len(actions)//2:]``.
    - Actions is assumed to be shuffled *before* calling this function!
    - The depth argument is just for pretty printing debugging information (useless).

    .. note:: The binary tournament is RANDOMIZED here, *as it should be*.

    >>> np.random.seed(1234)  # reproducible results
    >>> pulls = [5, 6, 7, 8]; K = len(pulls); N = max(pulls)
    >>> actions = np.arange(K)
    >>> rewards = np.random.randn(K, N)
    >>> np.mean(rewards, axis=1)  # arm 0 is better
    array([ 0.09876921, -0.18561207,  0.04463033,  0.0653539 ])
    >>> np.mean(rewards[:, :min(pulls)], axis=1)  # arm 1 is better in the first 6 samples
    array([-0.06401484,  0.17366346,  0.05323033, -0.09514708])
    >>> besa_K_actions(rewards, pulls, actions, subsample_function=subsample_deterministic)  # doctest: +ELLIPSIS
    3
    >>> [besa_K_actions(rewards, pulls, actions, subsample_function=subsample_uniform) for _ in range(10)]  # doctest: +ELLIPSIS
    [3, 3, 2, 3, 3, 0, 0, 0, 2, 3]
    """
    # print("In 'besa_K_actions', actions = {} for this call.".format(actions))  # DEBUG
    if len(actions) <= 1:
        chosen_arm = actions[0]
    elif len(actions) == 2:
        chosen_arm = besa_two_actions(rewards, pulls, actions[0], actions[1], subsample_function=subsample_function)
    else:
        # actions is already shuffled!
        actions_left = actions[:len(actions)//2]
        actions_right = actions[len(actions)//2:]
        # print("Using actions_left = {} and actions_right = {}...".format(actions_left, actions_right))  # DEBUG
        chosen_left = besa_K_actions(rewards, pulls, actions_left, subsample_function=subsample_function, depth=depth+1)
        chosen_right = besa_K_actions(rewards, pulls, actions_right, subsample_function=subsample_function, depth=depth+1)
        # print("The two recursive calls gave chosen_left = {}, chosen_right = {}...".format(chosen_left, chosen_right))  # DEBUG
        chosen_arm = besa_two_actions(rewards, pulls, chosen_left, chosen_right, subsample_function=subsample_function)
    # print("{}In 'besa_K_actions', actions = {} gave chosen_arm = {}.".format("\t" * depth, actions, chosen_arm))  # DEBUG
    return chosen_arm


def besa_K_actions__non_binary(rewards, pulls, actions, subsample_function=subsample_uniform, depth=0):
    r""" BESA recursive selection algorithm for an action set of size :math:`\mathcal{K} \geq 1`.

    - Instead of doing this binary tree tournaments (which results in :math:`\mathcal{O}(K^2)` calls to the 2-arm procedure), we can do a line tournaments: 1 vs 2, winner vs 3, winner vs 4 etc, winner vs K-1 (which results in :math:`\mathcal{O}(K)` calls),
    - Actions is assumed to be shuffled *before* calling this function!
    - The depth argument is just for pretty printing debugging information (useless).

    >>> np.random.seed(1234)  # reproducible results
    >>> pulls = [5, 6, 7, 8]; K = len(pulls); N = max(pulls)
    >>> actions = np.arange(K)
    >>> rewards = np.random.randn(K, N)
    >>> np.mean(rewards, axis=1)  # arm 0 is better
    array([ 0.09876921, -0.18561207,  0.04463033,  0.0653539 ])
    >>> np.mean(rewards[:, :min(pulls)], axis=1)  # arm 1 is better in the first 6 samples
    array([-0.06401484,  0.17366346,  0.05323033, -0.09514708])
    >>> besa_K_actions__non_binary(rewards, pulls, actions, subsample_function=subsample_deterministic)  # doctest: +ELLIPSIS
    3
    >>> [besa_K_actions__non_binary(rewards, pulls, actions, subsample_function=subsample_uniform) for _ in range(10)]  # doctest: +ELLIPSIS
    [3, 3, 3, 2, 0, 3, 3, 3, 3, 3]
    """
    # print("In 'besa_K_actions__non_binary', actions = {} for this call.".format(actions))  # DEBUG
    if len(actions) <= 1:
        chosen_arm = actions[0]
    elif len(actions) == 2:
        chosen_arm = besa_two_actions(rewards, pulls, actions[0], actions[1], subsample_function=subsample_function)
    else:
        chosen_arm = actions[0]
        for i in range(1, len(actions)):
            chosen_arm = besa_two_actions(rewards, pulls, chosen_arm, actions[i], subsample_function=subsample_function)
    # print("{}In 'besa_K_actions__non_binary', actions = {} gave chosen_arm = {}.".format("\t" * depth, actions, chosen_arm))  # DEBUG
    return chosen_arm


def besa_K_actions__non_recursive(rewards, pulls, subsample_function=subsample_uniform):
    r""" BESA non-recursive selection algorithm for an action set of size :math:`\mathcal{K} \geq 1`.

    - No calls to :func:`besa_two_actions`, just generalize it to K actions instead of 2.
    - Actions is assumed to be shuffled *before* calling this function!

    >>> np.random.seed(1234)  # reproducible results
    >>> pulls = [5, 6, 7, 8]; K = len(pulls); N = max(pulls)
    >>> rewards = np.random.randn(K, N)
    >>> np.mean(rewards, axis=1)  # arm 0 is better
    array([ 0.09876921, -0.18561207,  0.04463033,  0.0653539 ])
    >>> np.mean(rewards[:, :min(pulls)], axis=1)  # arm 1 is better in the first 6 samples
    array([-0.06401484,  0.17366346,  0.05323033, -0.09514708])
    >>> besa_K_actions__non_recursive(rewards, pulls, subsample_function=subsample_deterministic)  # doctest: +ELLIPSIS
    3
    >>> [besa_K_actions__non_recursive(rewards, pulls, subsample_function=subsample_uniform) for _ in range(10)]  # doctest: +ELLIPSIS
    [1, 3, 0, 2, 2, 3, 1, 1, 3, 1]
    """
    K = len(pulls)
    min_pulls = np.min(pulls)
    sub_means = np.zeros(K)
    for k in range(K):
        Ik = subsample_function(min_pulls, pulls[k])
        # sub_means[k] = np.mean(rewards[k, Ik])
        if isinstance(rewards, np.ndarray):  # faster to compute this
            sub_means[k] = np.sum(rewards[k, Ik]) / min_pulls
        else:  # than this for other data type (eg. dict mapping int to list)
            sub_means[k] = sum(rewards[k][i] for i in Ik) / min_pulls
    max_sub_means = np.max(sub_means)
    which_are_best = np.nonzero(sub_means == max_sub_means)[0]
    best_less_sampled = np.asarray(pulls)[which_are_best]
    # return which_are_best[np.argmin(best_less_sampled)]
    return which_are_best[np.random.choice(np.nonzero(best_less_sampled == np.min(best_less_sampled))[0])]


# --- The BESA policy


class BESA(IndexPolicy):
    r""" The Best Empirical Sampled Average (BESA) algorithm.

    - Reference: [[Sub-Sampling For Multi Armed Bandits, Baransi et al., 2014]](https://arxiv.org/abs/1711.00400)

    .. warning:: The BESA algorithm requires to store all the history of rewards, so its memory usage for :math:`T` rounds with :math:`K` arms is :math:`\mathcal{O}(K T)`, which is huge for large :math:`T`, be careful! Aggregating different BESA instances is probably a bad idea because of this limitation!
    """

    def __init__(self, nbArms, horizon=None,
                 minPullsOfEachArm=1, randomized_tournament=True, random_subsample=True,
                 non_binary=False, non_recursive=False,
                 lower=0., amplitude=1.):
        super(BESA, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # --- Arguments
        # XXX find a solution to not need to horizon?
        self.horizon = horizon  #: Just to know the memory to allocate for rewards. It could be implemented without knowing the horizon, by using lists to keep all the reward history, but this would be way slower!
        self.minPullsOfEachArm = max(1, int(minPullsOfEachArm))  #: Minimum number of pulls of each arm before using the BESA algorithm. Using 1 might not be the best choice
        self.randomized_tournament = randomized_tournament  #: Whether to use a deterministic or random tournament.
        self.random_subsample = random_subsample  #: Whether to use a deterministic or random sub-sampling procedure.
        self.non_binary = non_binary  #: Whether to use :func:`besa_K_actions` or :func:`besa_K_actions__non_binary` for the selection of K arms.
        self.non_recursive = non_recursive  #: Whether to use :func:`besa_K_actions` or :func:`besa_K_actions__non_recursive` for the selection of K arms.
        assert not (non_binary and non_recursive), "Error: BESA cannot use simultaneously non_binary and non_recursive option..."  # DEBUG
        self._subsample_function = subsample_uniform if random_subsample else subsample_deterministic
        # --- Internal memory
        assert nbArms >= 2, "Error: BESA algorithm can only work for at least 2 arms."
        self._left = 0  # just keep them in memory to increase readability
        self._right = nbArms - 1  # just keep them in memory to increase readability
        self._actions = np.arange(nbArms)  # just keep them in memory to increase readability

        # Memory to store all the rewards
        self._has_horizon = (self.horizon is not None) and (self.horizon > 1)
        if self._has_horizon:
            self.all_rewards = np.zeros((nbArms, horizon + 1))  #: Keep **all** rewards of each arms. It consumes a :math:`\mathcal{O}(K T)` memory, that's really bad!!
            self.all_rewards.fill(-1e5)  # Just security, to be sure they don't count as zero in some computation
        else:
            self.all_rewards = { k : [] for k in range(nbArms) }

    def __str__(self):
        """ -> str"""
        b1, b2, b3, b4, b5, b6 = not self.random_subsample, not self.randomized_tournament, self.minPullsOfEachArm > 1, not self._has_horizon, self.non_binary, self.non_recursive
        return "BESA{}{}{}{}{}{}{}{}{}{}{}{}{}".format(
            "(" if (b1 or b2 or b3 or b4 or b5 or b6) else "",
            "non-random subsample" if b1 else "",
            ", " if b1 and (b2 or b3 or b4 or b5 or b6) else "",
            "non-random tournament" if b2 else "",
            ", " if b2 and (b3 or b4 or b5 or b6) else "",
            r"$T_0={}$".format(self.minPullsOfEachArm) if b3 else "",
            ", " if b3 and (b4 or b5 or b6) else "",
            "anytime" if b4 else "",
            ", " if b4 and (b5 or b6) else "",
            "non-binary" if b5 else "",
            ", " if b5 and b6 else "",
            "non-recursive" if b6 else "",
            ")" if (b1 or b2 or b3 or b4 or b5 or b6) else "",
        )

    def getReward(self, arm, reward):
        """ Add the current reward in the global history.

        .. note:: There is no need to normalize the reward in [0,1], that's one of the strong point of the BESA algorithm."""
        # XXX find a solution to not need to horizon?
        if self._has_horizon:
            self.all_rewards[arm, self.pulls[arm]] = reward
        else:
            self.all_rewards[arm].append(reward)
        super(BESA, self).getReward(arm, reward)

    # --- Basic choice() and handleCollision() method

    def choice(self):
        """ Applies the BESA procedure with the current data history."""
        # if some arm has never been selected, force to explore it!
        if np.any(self.pulls < self.minPullsOfEachArm):
            return np.random.choice(np.where(self.pulls < self.minPullsOfEachArm)[0])
        else:
            if self.randomized_tournament:
                np.random.shuffle(self._actions)
            # print("Calling 'besa_K_actions' with actions list = {}...".format(self._actions))  # DEBUG
            return besa_K_actions(self.all_rewards, self.pulls, self._actions, subsample_function=self._subsample_function, depth=0)

    # --- Others choice...() methods, partly implemented

    def choiceFromSubSet(self, availableArms='all'):
        """ Applies the BESA procedure with the current data history, to the restricted set of arm."""
        if availableArms == 'all':
            return self.choice()
        else:
            # if some arm has never been selected, force to explore it!
            if any(self.pulls[k] < self.minPullsOfEachArm for k in availableArms):
                return np.random.choice([k for k in availableArms if self.pulls[k] < self.minPullsOfEachArm])
            else:
                actions = list(availableArms)
                if self.randomized_tournament:
                    np.random.shuffle(actions)
                # print("Calling 'besa_K_actions' with actions list = {}...".format(actions))  # DEBUG
                return besa_K_actions(self.all_rewards, self.pulls, actions, subsample_function=self._subsample_function, depth=0)

    def choiceMultiple(self, nb=1):
        """ Applies the multiple-choice BESA procedure with the current data history:

        1. select a first arm with basic BESA procedure with full action set,
        2. remove it from the set of actions,
        3. restart step 1 with new smaller set of actions, until ``nb`` arm where chosen by basic BESA.

        .. note:: This was not studied or published before, and there is no theoretical results about it!

        .. warning:: This is very inefficient! The BESA procedure is already quite slow (with my current naive implementation), this is crazily slow!
        """
        if nb == 1:
            return np.array([self.choice()])
        else:
            actions = list(range(self.nbArms))
            choices = []
            for _ in range(nb):
                # if some arm has never been selected, force to explore it!
                if np.any(self.pulls[actions] < self.minPullsOfEachArm):
                    choice_n = actions[np.random.choice(np.where(self.pulls[actions] < self.minPullsOfEachArm)[0])]
                else:
                    if self.randomized_tournament:
                        np.random.shuffle(actions)
                    # print("Calling 'besa_K_actions' with actions list = {}...".format(actions))  # DEBUG
                    choice_n = besa_K_actions(self.all_rewards, self.pulls, actions, subsample_function=self._subsample_function, depth=0)
                # now, store it, remove it from action set
                choices.append(choice_n)
                actions.remove(choice_n)
            return np.array(choices)

    def choiceWithRank(self, rank=1):
        """ Applies the ranked BESA procedure with the current data history:

        1. use :meth:`choiceMultiplie` to select ``rank`` actions,
        2. then take the ``rank``-th chosen action (the last one).

        .. note:: This was not studied or published before, and there is no theoretical results about it!

        .. warning:: This is very inefficient! The BESA procedure is already quite slow (with my current naive implementation), this is crazily slow!
        """
        choices = self.choiceMultiple(nb=rank)
        return choices[-1]

    # XXX self.index is NOT used to choose arm, only to estimate their order

    def computeIndex(self, arm):
        """ Compute the current index of arm 'arm'.

        .. warning:: This index **is not** the one used for the choice of arm (which use sub sampling). It's just the empirical mean of the arm.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return self.rewards[arm] / self.pulls[arm]

    def computeAllIndex(self):
        """ Compute the current index of arm 'arm' (vectorized).

        .. warning:: This index **is not** the one used for the choice of arm (which use sub sampling). It's just the empirical mean of the arm.
        """
        self.index = self.rewards / self.pulls
        self.index[self.pulls < 1] = float('+inf')

    def handleCollision(self, arm, reward=None):
        """ Nothing special to do."""
        pass


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
