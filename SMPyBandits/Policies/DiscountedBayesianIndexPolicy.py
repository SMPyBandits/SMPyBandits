# -*- coding: utf-8 -*-
""" Discounted Bayesian index policy.

- By default, it uses a DiscountedBeta posterior (:class:`Policies.Posterior.DiscountedBeta`), one by arm.
- Use discount factor :math:`\gamma\in(0,1)`.

.. warning:: This is still highly experimental!
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

try:
    from .BayesianIndexPolicy import BayesianIndexPolicy
    from .Posterior import DiscountedBeta
except ImportError:
    from BayesianIndexPolicy import BayesianIndexPolicy
    from Posterior import DiscountedBeta


# --- Constants

#: Default value for the discount factor :math:`\gamma\in(0,1)`.
#: ``0.95`` is empirically a reasonable value for short-term non-stationary experiments.
GAMMA = 0.95


# --- Class

class DiscountedBayesianIndexPolicy(BayesianIndexPolicy):
    r""" Discounted Bayesian index policy.

    - By default, it uses a DiscountedBeta posterior (:class:`Policies.Posterior.DiscountedBeta`), one by arm.
    - Use discount factor :math:`\gamma\in(0,1)`.

    - It keeps :math:`\widetilde{S_k}(t)` and :math:`\widetilde{F_k}(t)` the discounted counts of successes and failures (S and F), for each arm k.

    - But instead of using :math:`\widetilde{S_k}(t) = S_k(t)` and :math:`\widetilde{N_k}(t) = N_k(t)`, they are updated at each time step using the discount factor :math:`\gamma`:

    .. math::

        \widetilde{S_{A(t)}}(t+1) &= \gamma \widetilde{S_{A(t)}}(t) + r(t),\\
        \widetilde{S_{k'}}(t+1) &= \gamma \widetilde{S_{k'}}(t), \forall k' \neq A(t).

    .. math::

        \widetilde{F_{A(t)}}(t+1) &= \gamma \widetilde{F_{A(t)}}(t) + (1 - r(t)),\\
        \widetilde{F_{k'}}(t+1) &= \gamma \widetilde{F_{k'}}(t), \forall k' \neq A(t).
    """

    def __init__(self, nbArms,
        gamma=GAMMA, posterior=DiscountedBeta,
        lower=0., amplitude=1.,
        *args, **kwargs
    ):
        """ Create a new Bayesian policy, by creating a default posterior on each arm."""
        super(DiscountedBayesianIndexPolicy, self).__init__(nbArms, posterior=posterior, lower=lower, amplitude=amplitude, gamma=gamma)
        assert 0 < gamma <= 1, "Error: for a DiscountedBayesianIndexPolicy policy, the discount factor has to be in [0,1], but it was {}.".format(gamma)  # DEBUG
        if gamma == 1:
            print("Warning: gamma = 1 is stupid, just use a regular Beta posterior!")  # DEBUG
        self.gamma = gamma  #: Discount factor :math:`\gamma\in(0,1)`.

    def __str__(self):
        """ -> str"""
        return r"{}($\gamma={:.5g}${})".format(self.__class__.__name__, self.gamma, self._posterior_name if self._posterior_name != "DiscountedBeta" else "")

    def getReward(self, arm, reward):
        """ Update the posterior on each arm, with the normalized reward."""
        self.posterior[arm].update((reward - self.lower) / self.amplitude)
        # DONE we should update the other posterior with "no observation"
        for otherArm in range(self.nbArms):
            if otherArm != arm:
                self.posterior[arm].discount()
        self.t += 1