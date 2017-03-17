# -*- coding: utf-8 -*-
""" Manipulate a Gamma posterior. No need for tricks to handle non-binary rewards.

- See https://en.wikipedia.org/wiki/Gamma_distribution#Conjugate_prior
- And https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions
"""

__author__ = "Emilie Kaufmann, Lilian Besson"
__version__ = "0.5"

try:
    from numpy.random import gamma as gammavariate  # Faster! Yes!
except ImportError:
    from random import gammavariate

from scipy.special import gdtrix


# Local imports
from .Posterior import Posterior


class Gamma(Posterior):
    """ Manipulate a Gamma posterior."""

    def __init__(self, k=1, lmbda=1):
        """Create a Gamma posterior."""
        assert k > 0, "Error: parameter 'k' for Beta posterior has to be > 0."
        self.k0 = self.k = k
        assert lmbda > 0, "Error: parameter 'lmbda' for Beta posterior has to be > 0."
        self.lmbda0 = self.lmbda = lmbda

    def __str__(self):
        return "Gamma({}, {})".format(self.k, self.lmbda)

    def reset(self, k=None, lmbda=None):
        """Reset k and lmbda, both to 1 as when creating a new default Gamma."""
        if k is None:
            self.k = self.k0
        if lmbda is None:
            self.lmbda = self.lmbda0

    def sample(self):
        """Get a random sample from the Beta posterior (using :func:`numpy.random.gammavariate`).

        - Used only by Thompson Sampling so far.
        """
        return gammavariate(self.k, 1. / self.lmbda)

    def quantile(self, p):
        """Return the p quantile of the Gamma posterior (using :func:`scipy.stats.gdtrix`).

        - Used only by BayesUCB so far.
        """
        return gdtrix(self.k, 1. / self.lmbda, p)

    def mean(self):
        """Compute the mean of the Gamma posterior (should be useless)."""
        return self.k / float(self.lmbda)

    def forget(self, obs):
        """Forget the last observation."""
        # print("Info: calling Gamma.forget() with obs = {} ...".format(obs))  # DEBUG
        self.k += self.k0
        self.lmbda += obs

    def update(self, obs):
        """Add an observation: increase k by k0, and lmbda by obs (do not have to be normalized)."""
        # print("Info: calling Gamma.update() with obs = {} ...".format(obs))  # DEBUG
        self.k += self.k0
        self.lmbda += obs
