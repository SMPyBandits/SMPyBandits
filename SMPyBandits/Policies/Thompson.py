# -*- coding: utf-8 -*-
""" The Thompson (Bayesian) index policy.

- By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
- Reference: [Thompson - Biometrika, 1933].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann, Lilian Besson"
__version__ = "0.9"

try:
    from .BayesianIndexPolicy import BayesianIndexPolicy
except (ImportError, SystemError):
    from BayesianIndexPolicy import BayesianIndexPolicy


class Thompson(BayesianIndexPolicy):
    r"""The Thompson (Bayesian) index policy.

    - By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
    - Prior is initially flat, i.e., :math:`a=\alpha_0=1` and :math:`b=\beta_0=1`.

    - A non-flat prior for each arm can be given with parameters ``a`` and ``b``, for instance::

        nbArms = 2
        prior_failures  = a = 100
        prior_successes = b = 50
        policy = Thompson(nbArms, a=a, b=b)
        np.mean([policy.choice() for _ in range(1000)])  # 0.515 ~= 0.5: each arm has same prior!

    - A different prior for each arm can be given with parameters ``params_for_each_posterior``, for instance::

        nbArms = 2
        params0 = { 'a': 10, 'b': 5}  # mean 1/3
        params1 = { 'a': 5, 'b': 10}  # mean 2/3
        params = [params0, params1]
        policy = Thompson(nbArms, params_for_each_posterior=params)
        np.mean([policy.choice() for _ in range(1000)])  # 0.9719 ~= 1: arm 1 is better than arm 0 !

    - Reference: [Thompson - Biometrika, 1933].
    """

    def __str__(self):
        return "Thompson Sampling"
    
    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, giving :math:`S_k(t)` rewards of 1, by sampling from the Beta posterior:

        .. math::
            A(t) &\sim U(\arg\max_{1 \leq k \leq K} I_k(t)),\\
            I_k(t) &\sim \mathrm{Beta}(1 + \tilde{S_k}(t), 1 + \tilde{N_k}(t) - \tilde{S_k}(t)).
        """
        return self.posterior[arm].sample()
