# -*- coding: utf-8 -*-
""" The 1/2-Tsallis-Inf policy for bounded bandit, (order) optimal for stochastic and adversarial bandits.

- Reference: [["An Optimal Algorithm for Stochastic and Adversarial Bandits", Julian Zimmert, Yevgeny Seldin, 2018, arXiv:1807.07623]](https://arxiv.org/abs/1807.07623)
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from math import sqrt
import numpy as np
import numpy.random as rn
import scipy.optimize as opt
try:
    from .Exp3 import Exp3
except ImportError:
    from Exp3 import Exp3

#: Default value for :math:`\alpha` the parameter of the Tsallis entropy.
#: We focus on the 1/2-Tsallis algorithm, ie, with :math:`\alpha=\frac{1}{2}`.
ALPHA = 0.5


class TsallisInf(Exp3):
    """ The 1/2-Tsallis-Inf policy for bounded bandit, (order) optimal for stochastic and adversarial bandits.

    - Reference: [["An Optimal Algorithm for Stochastic and Adversarial Bandits", Julian Zimmert, Yevgeny Seldin, 2018, arXiv:1807.07623]](https://arxiv.org/abs/1807.07623)
    """

    def __init__(self, nbArms, alpha=ALPHA, lower=0., amplitude=1.):
        super(TsallisInf, self).__init__(nbArms, unbiased=True, lower=lower, amplitude=amplitude)
        self.alpha = alpha  #: Store the constant :math:`\alpha` used by the Online-Mirror-Descent step using :math:`\alpha` Tsallis entropy.
        self.inverse_exponent = 1.0 / (self.alpha - 1.0)  #: Store :math:`\frac{1}{\alpha-1}` to only compute it once.
        self.cumulative_losses = np.zeros(nbArms)  #: Keep in memory the vector :math:`\hat{L}_t` of cumulative (unbiased estimates) of losses.

    def __str__(self):
        return r"Tsallis-Inf($\alpha={:.3g}$)".format(self.alpha)

    @property
    def eta(self):
        r""" Decreasing learning rate, :math:`\eta_t = \frac{1}{\sqrt{t}}`."""
        return 1.0 / sqrt(max(1, self.t))

    @property
    def trusts(self):
        r""" Trusts probabilities :math:`\mathrm{trusts}(t+1)` are just the normalized weights :math:`w_k(t)`.
        """
        return self.weights

    def getReward(self, arm, reward):
        r""" Give a reward: accumulate rewards on that arm k, then recompute the trusts.

        Compute the trusts probabilities :math:`w_k(t)` with one step of Online-Mirror-Descent for bandit, using the :math:`\alpha` Tsallis entropy for the :math:`\Psi_t` functions.

        .. math::

            \mathrm{trusts}'_k(t+1) &= \nabla (\Psi_t + \mathcal{I}_{\Delta^K})^* (- \hat{L}_{t-1}), \\
            \mathrm{trusts}(t+1) &= \mathrm{trusts}'(t+1) / \sum_{k=1}^{K} \mathrm{trusts}'_k(t+1).

        - If :math:`\Delta^K` is the probability simplex of dimension :math:`K`,
        - and :math:`\hat{L}_{t-1}` is the cumulative loss vector, ie, the sum of the (unbiased estimate) :math:`\hat{\ell}_t` for the previous time steps,
        - where :math:`\hat{\ell}_{t,i} = 1(I_t = i) \frac{\ell_{t,i}}{\mathrm{trusts}_i(t)}` is the unbiased estimate of the loss,
        - With :math:`\Psi_t = \Psi_{t,\alpha}(w) := - \sum_{k=1}^{K} \frac{w_k^{\alpha}}{\alpha \eta_t}`,
        - With learning rate :math:`\eta_t = \frac{1}{\sqrt{t}}` the (decreasing) learning rate.
        """
        super(TsallisInf, self).getReward(arm, reward)  # XXX Call to Exp3
        # normalize reward to [0,1]
        reward = (reward - self.lower) / self.amplitude
        # for one reward in [0,1], loss = 1 - reward
        biased_loss = 1.0 - reward
        # unbiased estimate, from the weights of the previous step
        unbiased_loss = biased_loss / self.weights[arm]
        self.cumulative_losses[arm] += unbiased_loss
        eta_t = self.eta

        # 1. solve f(x)=1 to get an approximation of the (unique) Lagrange multiplier x
        def objective_function(x):
            return (np.sum( (eta_t * (self.cumulative_losses - x)) ** self.inverse_exponent ) - 1) ** 2

        result_of_minimization = opt.minimize_scalar(objective_function)
        # result_of_minimization = opt.minimize(objective_function, 0.0)  # XXX is it not faster?
        x = result_of_minimization.x

        # 2. use x to compute the new weights
        new_weights = ( eta_t * (self.cumulative_losses - x) ) ** self.inverse_exponent

        # print("DEBUG: {} at time {} (seeing reward {} on arm {}), compute slack variable x = {}, \n    and new_weights = {}...".format(self, self.t, reward, arm, x, new_weights))  # DEBUG

        # XXX Handle weird cases, slow down everything but safer!
        if not np.all(np.isfinite(new_weights)):
            new_weights[~np.isfinite(new_weights)] = 0  # set bad values to 0
        # Bad case, where the sum is so small that it's only rounding errors
        # or where all values where bad and forced to 0, start with new_weights=[1/K...]
        if np.isclose(np.sum(new_weights), 0):
            # Normalize it!
            new_weights[:] = 1.0

        # 3. Renormalize weights at each step
        new_weights /= np.sum(new_weights)

        # 4. store weights
        self.weights =  new_weights
