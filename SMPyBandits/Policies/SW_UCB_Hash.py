# -*- coding: utf-8 -*-
r""" The SW-UCB# policy for non-stationary bandits, from [["On Abruptly-Changing and Slowly-Varying Multiarmed Bandit Problems", by Lai Wei, Vaibhav Srivastava, 2018, arXiv:1802.08380]](https://arxiv.org/pdf/1802.08380)

- It runs on top of a simple policy, e.g., :class:`Policy.UCB.UCB`, and :func:`SWHash_IndexPolicy` is a generic policy using any simple policy with this "sliding window" trick:

    >>> policy = SWHash_IndexPolicy(nbArms, UCB, tau=100, threshold=0.1)
    >>> # use policy as usual, with policy.startGame(), r = policy.choice(), policy.getReward(arm, r)

- It uses an additional non-fixed :math:`\mathcal{O}(\tau(t,\alpha))` memory and an extra time complexity.

.. warning:: This is very experimental! TODO finish!
.. warning:: It can only work on basic index policy based on empirical averages (and an exploration bias), like :class:`Policy.UCB.UCB`, and cannot work on any Bayesian policy (for which we would have to remember all previous observations in order to reset the history with a small history)!
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


import numpy as np

try:
    from .BaseWrapperPolicy import BaseWrapperPolicy
    from .BasePolicy import BasePolicy
    from .UCB import UCB as DefaultPolicy, UCB
    from .klUCB import klUCB, klucbBern, c
except ImportError:
    from BaseWrapperPolicy import BaseWrapperPolicy
    from BasePolicy import BasePolicy
    from UCB import UCB as DefaultPolicy, UCB
    from klUCB import klUCB, klucbBern, c


# --- Parameter alpha

def alpha_for_abruptly_changing_env(nu=0.5):
    r"""For abruptly-changing environement, if the number of break-points is :math:`\Upsilon_T = \mathcal{O}(T^{\nu})`, then the SW-UCB# algorithm chooses :math:`\alpha = \frac{1-\nu}{2}`."""
    return (1 - nu) / 2


def alpha_for_slowly_varying_env(kappa=1):
    r"""For slowly-varying environement, if the change in mean reward between two time steps is bounded by :math:`\varepsilon_T = \mathcal{O}(T^{-\kappa})`, then the SW-UCB# algorithm chooses :math:`\alpha = \min{1, \frac{3\kappa}{4}}`."""
    return min(1, 3 * kappa / 4)


# --- Parameters rho, b etc

#: Default parameter for :math:`\alpha`.
ALPHA = 0.5

#: Default parameter for :math:`\lambda`.
LAMBDA = 1

def tau_t_alpha(t, alpha=ALPHA, lmbda=LAMBDA):
    r""" Compute :math:`\tau(t,\alpha) = \min(\lceil \lambda t^{\alpha} \rceil, t)`."""
    return int(min(np.ceil(lmbda * (t**alpha)), t))


# --- Class SWHash_IndexPolicy

# TODO the generic class, if SW_UCB_Hash works fine
class SWHash_IndexPolicy(BaseWrapperPolicy):
    r""" The SW-UCB# policy for non-stationary bandits, from [["On Abruptly-Changing and Slowly-Varying Multiarmed Bandit Problems", by Lai Wei, Vaibhav Srivastava, 2018, arXiv:1802.08380]](https://arxiv.org/pdf/1802.08380)
    """

    def __init__(self, nbArms, policy=DefaultPolicy,
            alpha0=ALPHA,
            lower=0., amplitude=1., *args, **kwargs
        ):
        super(SWHash_IndexPolicy, self).__init__(nbArms, policy=policy, lower=lower, amplitude=amplitude, *args, **kwargs)
        self.alpha0 = alpha0  #: The parameter :math:`\alpha` for the SW-UCB# algorithm (see article for reference).
        # Internal memory
        self.all_rewards = [[] for _ in range(nbArms)]  #: Keep in memory all the rewards obtained in the all :math:`\tau(t,\alpha)` steps (the size of the window is evolving!).
        self.all_pulls = np.full(nbArms, -1)  #: Keep in memory the times where each arm was all seen. Start with -1 (never seen).

    def __str__(self):
        return r"SW-Hash({})".format(self._policy.__name__)

    def startGame(self, createNewPolicy=True):
        """ Initialize the policy for a new game."""
        super(SWHash_IndexPolicy, self).startGame(createNewPolicy=createNewPolicy)
        self.last_rewards = [[] for _ in range(self.nbArms)]

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).

        - Reset the whole empirical average if the small average is too far away from it.
        """
        super(SWHash_IndexPolicy, self).getReward(arm, reward)
        # Get reward
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time
        self.last_pulls[arm] += 1
        # Store it in place for the empirical average of that arm
        self.last_rewards[arm, self.last_pulls[arm] % self._tau] = reward
        if self.last_pulls[arm] >= self._tau \
            and self.pulls[arm] >= self._tau:
            # Compute the empirical average for that arm
            empirical_average = self.rewards[arm] / self.pulls[arm]
            # And the small empirical average for that arm
            small_empirical_average = np.mean(self.last_rewards[arm])
            if np.abs(empirical_average - small_empirical_average) >= self._threshold:
                # Fully restart the algorithm ?!
                if self._full_restart_when_refresh:
                    self.startGame(createNewPolicy=False)
                # Or simply reset one of the empirical averages?
                else:
                    self.rewards[arm] = np.sum(self.last_rewards[arm])
                    self.pulls[arm] = 1 + (self.last_pulls[arm] % self._tau)


# --- Manually written


class SW_UCB_Hash(UCB):
    r""" The SW-UCB# policy for non-stationary bandits, from [["On Abruptly-Changing and Slowly-Varying Multiarmed Bandit Problems", by Lai Wei, Vaibhav Srivastava, 2018, arXiv:1802.08380]](https://arxiv.org/pdf/1802.08380)
    """

    def __init__(self, nbArms,
            alpha=ALPHA, lmbda=LAMBDA,
            lower=0., amplitude=1.,
            *args, **kwargs
        ):
        super(SW_UCB_Hash, self).__init__(nbArms, lower=lower, amplitude=amplitude, *args, **kwargs)
        self.alpha0 = alpha  #: The parameter :math:`\alpha` for the SW-UCB# algorithm (see article for reference).
        self.alpha = 1 + alpha  #: The parameter :math:`\alpha` for the UCB indexes
        self.lmbda = lmbda  #: The parameter :math:`\lambda` for the SW-UCB# algorithm (see article for reference).
        # Internal memory
        self.all_rewards = []  #: Keep in memory all the rewards obtained in the all the past steps (the size of the window is evolving!).
        self.all_pulls = []  #: Keep in memory all the pulls obtained in the all the past steps (the size of the window is evolving!). Start with -1 (never seen).

    def __str__(self):
        return r"SW-UCB#($\lambda={:.3g}$, $\alpha={:.3g}$)".format(self.lmbda, self.alpha0)

    @property
    def tau(self):
        r""" The current :math:`\tau(t,\alpha)`."""
        return tau_t_alpha(self.t, alpha=self.alpha0, lmbda=self.lmbda)

    def startGame(self):
        """ Initialize the policy for a new game."""
        super(SW_UCB_Hash, self).startGame()
        self.all_rewards = []
        self.all_pulls = []

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update total history and partial history of all arms (normalized in [0, 1]).

        .. warning:: So far this is badly implemented and the algorithm is VERY slow: it has to store all the past, as the window-length is increasing when t increases.
        """
        self.t += 1
        # Get reward, normalize it
        reward = (reward - self.lower) / self.amplitude
        # We seen it one more time at this time step?
        self.all_pulls.append(arm)
        self.all_rewards.append(reward)
        # compute the size of the current window
        current_tau = self.tau
        # print("For {} at time t = {}, current tau = {}, a reward = {} was seen from arm {}...".format(self, self.t, current_tau, reward, arm))  # DEBUG
        # it's highly innefficient but who cares
        partial_all_pulls = self.all_pulls[-current_tau:]
        # print("partial_all_pulls =", partial_all_pulls)  # DEBUG
        partial_pulls = np.bincount(partial_all_pulls, minlength=self.nbArms)
        # print("partial_pulls =", partial_pulls)  # DEBUG
        partial_all_rewards = self.all_rewards[-current_tau:]
        # print("partial_pulls =", partial_pulls)  # DEBUG

        # Compute fake pulls and fake average rewards
        for otherArm in range(self.nbArms):
            # Store it in place for the empirical average of that arm
            self.pulls[otherArm] = partial_pulls[otherArm]
            # print("self.pulls[otherArm] =", self.pulls[otherArm])  # DEBUG
            these_rewards = [partial_all_rewards[i] for i, p in enumerate(partial_all_pulls) if p == otherArm]
            self.rewards[otherArm] = np.mean(these_rewards)
            # print("self.rewards[otherArm] =", self.rewards[otherArm])  # DEBUG
