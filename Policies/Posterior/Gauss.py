# -*- coding: utf-8 -*-
""" Manipulate a posterior of Gaussian experiments, i.e., a Normal-Inverse gamma distribution.

Cf. https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution
No need for tricks to handle non-binary rewards.

- See https://en.wikipedia.org/wiki/Normal_distribution#With_unknown_mean_and_unknown_variance
- And https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

from warnings import warn
import numpy as np

try:
    from numpy.random import normal as normalvariate  # Faster! Yes!
except ImportError:
    from random import gauss as normalvariate


from scipy.stats import invgamma, norm


#: Default value for the variance of a [0, 1] Gaussian arm
VARIANCE = 0.05


def inverse_gamma(alpha=1., beta=1.):
    r""" Sample sigma2 from an Inverse Gamma distribution of parameters :math:`\alpha, \beta`.

    >>> np.random.seed(0)
    >>> inverse_gamma(1, 1)  # doctest: +ELLIPSIS
    1.6666...
    >>> inverse_gamma(2, 1)  # doctest: +ELLIPSIS
    0.9470...
    >>> inverse_gamma(1, 2)  # doctest: +ELLIPSIS
    3.9507...
    >>> inverse_gamma(2, 2)  # doctest: +ELLIPSIS
    1.2996...

    - Cf. https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    """
    return invgamma.rvs(alpha, scale=beta)


def normal_inverse_gamma(mu=0., nu=1., alpha=1., beta=1.):
    r""" Sample (x, sigma2) from a Normal-Inverse Gamma distribution of four parameters :math:`\mu, \nu, \alpha, \beta`.

    >>> np.random.seed(0)
    >>> normal_inverse_gamma(0, 1, 1, 1)  # doctest: +ELLIPSIS
    (1.2359..., 1.6666...)
    >>> normal_inverse_gamma(-20, 1, 1, 1)  # doctest: +ELLIPSIS
    (-17.4424..., 1.6469...)
    >>> normal_inverse_gamma(20, 1, 1, 1)  # doctest: +ELLIPSIS
    (19.0187..., 1.1643...)

    - Cf. https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution#Generating_normal-inverse-gamma_random_variates
    """
    sigma2 = inverse_gamma(alpha=alpha, beta=beta)
    x = normalvariate(loc=mu, scale=sigma2 / nu)
    return (x, sigma2)


# Local imports
from .Posterior import Posterior


class Gauss(Posterior):
    """ Manipulate a posterior of Gaussian experiments, i.e., a Normal-Inverse gamma distribution.

    Cf. https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution
    """

    def __init__(self, mu=0., nu=VARIANCE, alpha=1., beta=1.):
        r"""Create a posterior assuming the default is :math:`\mathcal{N}(\mu, \nu)` with `loc = mu` and `scale**2 = nu`."""
        self._mu = float(mu)
        self.mu = float(mu)  #: Parameter :math:`\mu` of the posterior
        assert nu > 0, "Error: parameter 'nu' for Gauss posterior has to be > 0."
        self._nu = float(nu)
        self.nu = float(nu)  #: Parameter :math:`\nu` of the posterior
        assert alpha > 0, "Error: parameter 'alpha' for Gauss posterior has to be > 0."
        self._alpha = float(alpha)
        self.alpha = float(alpha)  #: Parameter :math:`\alpha` of the posterior
        assert beta > 0, "Error: parameter 'beta' for Gauss posterior has to be > 0."
        self._beta = float(beta)
        self.beta = float(beta)  #: Parameter :math:`\beta` of the posterior

    def __str__(self):
        return "Gauss({:.3g}, {:.3g}, {:.3g}, {:.3g})".format(self.mu, self.nu, self.alpha, self.beta)

    def reset(self, mu=None, nu=None, alpha=None, beta=None):
        r""" Reset the for parameters :math:`\mu, \nu, \alpha, \beta`, as when creating a new Gauss posterior."""
        if mu is None:
            self.mu = self._mu
        if nu is None:
            self.nu = self._nu
        if alpha is None:
            self.alpha = self._alpha
        if beta is None:
            self.beta = self._beta

    def sample(self):
        r""" Get a random sample :math:`(x, \sigma^2)` from the Gaussian posterior (using :func:`scipy.stats.invgamma` for the variance :math:`\sigma^2` parameter and :func:`numpy.random.normal` for the mean :math:`x`).

        - Used only by :class:`Thompson` Sampling and :class:`AdBandits` so far.
        """
        loc, scale = normal_inverse_gamma(mu=self.mu, nu=self.nu, alpha=self.alpha, beta=self.beta)
        return normalvariate(loc=loc, scale=scale)

    def quantile(self, p):
        """ Return the p-quantile of the Gauss posterior.

        .. warning:: Very experimental, I am not sure of what I did here...

        .. note:: I recommend to NOT use :class:`BayesUCB` with Gauss posteriors...
        """
        # warn("Gauss.quantile() : Not implemented for a 2 dimensional distribution!", RuntimeWarning)
        quantile_on_sigma2 = invgamma.ppf(p, self.alpha, scale=self.beta)
        scale = self.beta / float(self.alpha)  # mean of the Inverse-gamma
        quantile_on_x = norm.ppf(p, loc=self.mu, scale=scale / float(self.nu))
        return quantile_on_x * quantile_on_sigma2
        raise ValueError("Gauss.quantile() : Not implemented for a 2 dimensional distribution!")

    def mean(self):
        """Compute the mean of the Gauss posterior (should be useless)."""
        return self.mu

    def update(self, obs):
        r"""Add an observation :math:`x` or a vector of observations, assumed to be drawn from an unknown normal distribution.

        - Initial mean was estimated from `\nu` observations with sample mean :math:`\mu _{0}`;
        - Initial variance was estimated from :math:`2\alpha` observations with sample mean :math:`\mu _{0}` and sum of squared deviations :math:`2\beta`.

        Let :math:`n` the size of :math:`x` and :math:`\overline{x} = \frac{1}{n} \sum_{i=1}^{n} x_i` the mean of the observations :math:`x` (can be one sample, or more).

        Then the four parameters :math:`\mu, \nu, \alpha, \beta` are updated like this:

        .. math::

           \mu' &:= \frac{\nu \mu + n \overline{x}}{\nu + n}, \\
           \nu' &:= \nu + n, \\
           \alpha' &:= \alpha + \frac{n}{2}, \\
           \beta' &:= \beta + \frac{1}{2} \sum_{i=1}^{n} (x_i - \overline{x})^2 + \frac{n \nu}{\nu + n} \frac{(\overline{x} - \mu)^2}{2}.\\

        """
        # print("Info: calling Gauss.update() with obs = {} ...".format(obs))  # DEBUG
        mu, nu, alpha, beta = self.mu, self.nu, self.alpha, self.beta
        n = np.size(obs)
        x = np.reshape(obs, (-1))  # = obs.flatten()
        # Compute
        xhat = np.mean(x)
        # Update all parameters
        new_mu = ((nu * mu) + (n * xhat)) / (nu + n)  # update mean, easy
        new_nu = nu + n  # n new observations
        new_alpha = alpha + (n / 2.)  # easy
        new_beta = beta + (np.sum((x - xhat)**2) / 2.) + ((nu * n) / (nu + n)) * ( (xhat - mu)**2 / 2.)  # crazy formula, cf. https://en.wikipedia.org/wiki/Conjugate_prior#cite_ref-11
        # Storing the new parameters
        self.mu, self.nu, self.alpha, self.beta = new_mu, new_nu, new_alpha, new_beta

    def forget(self, obs):
        """Forget the last observation. Should work, but should also not be used..."""
        print("Info: calling Gauss.forget() with obs = {} ...".format(obs))  # DEBUG
        new_mu, new_nu, new_alpha, new_beta = self.mu, self.nu, self.alpha, self.beta
        n = np.size(obs)
        x = np.reshape(obs, (-1))  # = obs.flatten()
        # Compute
        xhat = np.mean(x)
        # Update all parameters
        nu = new_nu - n  # n new observations
        mu = ((new_nu * new_mu) + (n * xhat)) / nu  # update mean, easy
        alpha = new_alpha - (n / 2.)  # easy
        beta = new_beta - (np.sum((x - xhat)**2) / 2.) - ((new_nu * n) / (new_nu + n)) * ( (xhat - new_mu)**2 / 2.)  # crazy formula, cf. https://en.wikipedia.org/wiki/Conjugate_prior#cite_ref-11
        # Storing the new parameters
        self.mu, self.nu, self.alpha, self.beta = mu, nu, alpha, beta
