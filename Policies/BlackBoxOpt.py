# -*- coding: utf-8 -*-
r""" An experimental "on-line" policy, using algorithms from black-box Bayesian optimization, using [scikit-optimize](https://scikit-optimize.github.io/).

It uses an iterative black-box Bayesian optimizer, with two methods :meth:`ask` and :meth:`tell` to be used as :meth:`choice` and :meth:`getReward` for our Multi-Armed Bandit optimization environment.
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np

# Cf. https://scikit-optimize.github.io/
from skopt.learning import ExtraTreesRegressor, GaussianProcessRegressor
from skopt import Optimizer


def default_estimator(*args, **kwargs):
    """Default estimator object.

    - Default is :class:`ExtraTreesRegressor` (https://scikit-optimize.github.io/learning/index.html#skopt.learning.ExtraTreesRegressor).
    - Any of https://scikit-optimize.github.io/learning/index.html can be used.
    """
    etr = ExtraTreesRegressor()
    etr = GaussianProcessRegressor()
    return etr


def default_optimizer(nbArms, est, *args, **kwargs):
    """Default optimizer object.

    - Default is :class:`Optimizer` (https://scikit-optimize.github.io/#skopt.Optimizer).
    """
    opt = Optimizer([
                    list(range(nbArms))  # Categorical
                    ],
                    est(*args, **kwargs),
                    acq_optimizer="sampling",
                    n_random_starts=100  # Sure ?
                    )
    return opt


# --- Decision Making Policy


class BlackBoxOpt(object):
    r"""Black-box Bayesian optimizer for Multi-Armed Bandit, using Gaussian processes.

    - By default, it uses :func:`default_optimizer`.

    .. warning:: This is still **experimental**!
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
        return "BlackBoxOpt({})".format(self.opt.__name__)

    def startGame(self):
        """ Reinitialize the black-box optimizer."""
        self.t = -1
        self.opt = self._opt(self.nbArms, self.est, *self._args, **self._kwargs)  #: The black-box optimizer to use, initialized from the other arguments

    def getReward(self, x, y):
        """ Store this observation `reward` for that arm `armId`."""
        return self.opt.tell(x, y)

    def choice(self):
        r""" Choose an arm, according to the black-box optimizer."""
        self.t += 1
        asked = self.opt.ask()
        # That's a np.array of float!
        arm = int(np.round(asked[0]))
        # Now it's a float, ok
        return arm

    # --- Other method

    def estimatedOrder(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means."""
        return np.argsort(self.index())  # FIXME find a way to do that!
