# -*- coding: utf-8 -*-
""" Manipulate a posterior of Gaussian experiments, which happens to also be a Gaussian distribution if the prior is Gaussian. *Easy peasy!*

.. warning:: TODO I have to test it!

- Reference: [[*Further optimal regret bounds for Thompson sampling*, S. Agrawal and N. Goyal, In Artificial Intelligence and Statistics, pages 99–107, 2013.](XXX)]
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np

try:
    from numpy.random import normal as normalvariate  # Faster! Yes!
except ImportError:
    from random import gauss as normalvariate

from scipy.special import nrdtrimn, nrdtrisd


# Local imports
from .Posterior import Posterior


class Gauss(Posterior):
    r""" Manipulate a posterior of Gaussian experiments, which happens to also be a Gaussian distribution if the prior is Gaussian.

    The posterior distribution is a :math:`\mathcal{N}(\hat{\mu_k}(t), \hat{\sigma_k}^2(t))`, where

    .. math::

        \hat{\mu_k}(t) &= \frac{X_k(t)}{N_k(t)},
        \hat{\sigma_k}^2(t) &= \frac{1}{N_k(t)}.
    """

    def __init__(self, mu=0., sigma=1):
        r"""Create a posterior assuming the default is :math:`\mathcal{N}(0, 1)`."""
        self._mu = float(mu)
        self.mu = float(mu)  #: Parameter :math:`\mu` of the posterior
        assert sigma > 0, "Error: parameter 'sigma' for Gauss posterior has to be > 0."
        self._nu = float(sigma)
        self.sigma = float(sigma)  #: Parameter :math:`\sigma` of the posterior
        # internal memories
        self._nb_data = 0  # number of samples!
        self._sum_data = 0.0  # sum of samples!

    def __str__(self):
        return "Gauss({:.3g}, {:.3g})".format(self.mu, self.sigma)

    def reset(self, mu=None, sigma=None):
        r""" Reset the for parameters :math:`\mu, \sigma`, as when creating a new Gauss posterior."""
        if mu is None:
            self.mu = self._mu
        if sigma is None:
            self.sigma = self._nu

    def sample(self):
        r""" Get a random sample :math:`(x, \sigma^2)` from the Gaussian posterior (using :func:`scipy.stats.invgamma` for the variance :math:`\sigma^2` parameter and :func:`numpy.random.normal` for the mean :math:`x`).

        - Used only by :class:`Thompson` Sampling and :class:`AdBandits` so far.
        """
        return normalvariate(loc=self.mu, scale=self.sigma)

    def quantile(self, p):
        """ Return the p-quantile of the Gauss posterior.

        .. note:: It now works fine with :class:`Policies.BayesUCB` with Gauss posteriors, even if it is MUCH SLOWER than the Bernoulli posterior (:class:`Gamma`).
        """
        quantile_on_x = nrdtrimn(p, 1, self.sigma)
        quantile_on_sigma2 = nrdtrisd(p, 1, self.mu)
        return quantile_on_x * quantile_on_sigma2

    def mean(self):
        r""" Compute the mean, :math:`\mu` of the Gauss posterior (should be useless)."""
        return self.mu

    def variance(self):
        r""" Compute the variance, :math:`\sigma`, of the Gauss posterior (should be useless)."""
        return self.sigma

    def update(self, obs):
        r"""Add an observation :math:`x` or a vector of observations, assumed to be drawn from an unknown normal distribution.
        """
        # print("Info: calling Gauss.update() with obs = {} ...".format(obs))  # DEBUG
        # one more observation!
        self._nb_data += 1
        self._sum_data += float(obs)
        mu, sigma = self.mu, self.sigma
        # Update all parameters
        new_sigma = 1 / float(self._nb_data)  # n observations so far
        new_mu = self._sum_data * new_sigma  # update mean, easy
        # Storing the new parameters
        self.mu, self.sigma = new_mu, new_sigma

    def forget(self, obs):
        """Forget the last observation. Should work, but should also not be used..."""
        raise NotImplementedError
