# -*- coding: utf-8 -*-
""" The Discounted Thompson (Bayesian) index policy.

- By default, it uses a DiscountedBeta posterior (:class:`Policies.Posterior.DiscountedBeta`), one by arm.
- Reference: [["Taming Non-stationary Bandits: A Bayesian Approach", Vishnu Raj & Sheetal Kalyani, arXiv:1707.09727](https://arxiv.org/abs/1707.09727)].

.. warning:: This is still highly experimental!
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

try:
    from .DiscountedBayesianIndexPolicy import DiscountedBayesianIndexPolicy
except (ImportError, SystemError):
    from DiscountedBayesianIndexPolicy import DiscountedBayesianIndexPolicy


class DiscountedThompson(DiscountedBayesianIndexPolicy):
    """The DiscountedThompson (Bayesian) index policy.

    - By default, it uses a DiscountedBeta posterior (:class:`Policies.Posterior.DiscountedBeta`), one by arm.
    - Reference: [["Taming Non-stationary Bandits: A Bayesian Approach", Vishnu Raj & Sheetal Kalyani, arXiv:1707.09727](https://arxiv.org/abs/1707.09727)].
    """

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, by sampling from the DiscountedBeta posterior.

        .. math::
            A(t) &\sim U(\arg\max_{1 \leq k \leq K} I_k(t)),\\
            I_k(t) &\sim \mathrm{Beta}(1 + \widetilde{S_k}(t), 1 + \widetilde{F_k}(t)).

        - It keeps :math:`\widetilde{S_k}(t)` and :math:`\widetilde{F_k}(t)` the discounted counts of successes and failures (S and F), for each arm k.

        - But instead of using :math:`\widetilde{S_k}(t) = S_k(t)` and :math:`\widetilde{N_k}(t) = N_k(t)`, they are updated at each time step using the discount factor :math:`\gamma`:

        .. math::

            \widetilde{S_{A(t)}}(t+1) &= \gamma \widetilde{S_{A(t)}}(t) + r(t),\\
            \widetilde{S_{k'}}(t+1) &= \gamma \widetilde{S_{k'}}(t), \forall k' \neq A(t).

        .. math::

            \widetilde{F_{A(t)}}(t+1) &= \gamma \widetilde{F_{A(t)}}(t) + (1 - r(t)),\\
            \widetilde{F_{k'}}(t+1) &= \gamma \widetilde{F_{k'}}(t), \forall k' \neq A(t).
        """
        return self.posterior[arm].sample()
