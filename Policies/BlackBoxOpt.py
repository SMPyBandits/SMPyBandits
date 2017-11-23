# -*- coding: utf-8 -*-
r""" An experimental "on-line" policy, using algorithms from black-box Bayesian optimization, using [scikit-optimize](https://scikit-optimize.github.io/).

- It uses an iterative black-box Bayesian optimizer, with two methods :meth:`ask` and :meth:`tell` to be used as :meth:`choice` and :meth:`getReward` for our Multi-Armed Bandit optimization environment.
- See https://scikit-optimize.github.io/notebooks/ask-and-tell.html for more details.

.. warning:: This is still **experimental**! It is NOT efficient in terms of storage, and **highly** NOT efficient either in terms of efficiency against a Bandit problem (i.e., regret, best arm identification etc).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np

# Ignore the UserWarning skopt/optimizer/optimizer.py:208:
# UserWarning: The objective has been evaluated at this point before.
from warnings import simplefilter
simplefilter("ignore", UserWarning)

# Cf. https://scikit-optimize.github.io/
import skopt.learning
from skopt import Optimizer

from .BasePolicy import BasePolicy


# --- Default estimator and optimizer

def default_estimator(*args, **kwargs):
    """Default estimator object.

    - Default is :class:`RandomForestRegressor` (https://scikit-optimize.github.io/learning/index.html#skopt.learning.RandomForestRegressor).
    - Another possibility is to use :class:`ExtraTreesRegressor` (https://scikit-optimize.github.io/learning/index.html#skopt.learning.ExtraTreesRegressor), but it is slower!
    - :class:`GaussianProcessRegressor` (https://scikit-optimize.github.io/learning/index.html#skopt.learning.GaussianProcessRegressor) was failing, don't really know why. I think it is not designed to work with Categorical inputs.
    - Any of https://scikit-optimize.github.io/learning/index.html can be used.
    """
    etr = skopt.learning.RandomForestRegressor(*args, **kwargs)
    # etr = skopt.learning.ExtraTreesRegressor(*args, **kwargs)
    # etr = skopt.learning.GaussianProcessRegressor(*args, **kwargs)
    return etr


def default_optimizer(nbArms, est, *args, **kwargs):
    """Default optimizer object.

    - Default is :class:`Optimizer` (https://scikit-optimize.github.io/#skopt.Optimizer).
    """
    opt = Optimizer([
                    list(range(nbArms))  # Categorical dimensions: arm index!
                    ],
                    est(*args, **kwargs),
                    acq_optimizer="sampling",
                    n_random_starts=3 * nbArms  # Sure ?
                    )
    return opt


# --- Decision Making Policy

class BlackBoxOpt(BasePolicy):
    r"""Black-box Bayesian optimizer for Multi-Armed Bandit, using Gaussian processes.

    - By default, it uses :func:`default_optimizer`.

    .. warning:: This is still **experimental**! It works fine, but it is EXTREMELY SLOW!
    """

    def __init__(self, nbArms,
                 opt=default_optimizer, est=default_estimator,
                 lower=0., amplitude=1.,  # not used, but needed for my framework
                 *args, **kwargs):
        self.nbArms = nbArms  #: Number of arms of the MAB problem.
        self.t = -1  #: Current time.
        # Black-box optimizer
        self._opt = opt  # Store it
        self._est = est  # Store it
        self._args = args  # Other non-kwargs args given to the estimator.
        self._kwargs = kwargs  # Other kwargs given to the estimator.
        self.opt = opt(nbArms, est, *args, **kwargs)  #: The black-box optimizer to use, initialized from the other arguments
        # Other attributes
        self.lower = lower  #: Known lower bounds on the rewards.
        self.amplitude = amplitude  #: Known amplitude of the rewards.

    # --- Easy methods

    def __str__(self):
        return "BlackBoxOpt({}, {})".format(self._opt.__name__, self._est.__name__)

    def startGame(self):
        """ Reinitialize the black-box optimizer."""
        self.t = -1
        self.opt = self._opt(self.nbArms, self._est, *self._args, **self._kwargs)  # The black-box optimizer to use, initialized from the other arguments

    def getReward(self, armId, reward):
        """ Store this observation `reward` for that arm `armId`.

        - In fact, :class:`skopt.Optimizer` is a *minimizer*, so `loss=1-reward` is stored, to maximize the rewards by minimizing the losses.
        """
        reward = (reward - self.lower) / self.amplitude  # project the reward to [0, 1]
        loss = 1. - reward  # flip
        # print("- A {} policy saw a reward = {} (= loss = {}) from arm = {}...".format(self, reward, loss, armId))  # DEBUG
        return self.opt.tell([armId], loss)

    def choice(self):
        r""" Choose an arm, according to the black-box optimizer."""
        self.t += 1
        asked = self.opt.ask()
        # That's a np.array of int, as we use Categorical input dimension!
        arm = int(np.round(asked[0]))
        # print("- At time t = {}, a {} policy chose to play arm = {}...".format(self, self.t, arm))  # DEBUG
        return arm
