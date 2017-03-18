# -*- coding: utf-8 -*-
""" Arms : contains different types of bandit arms:
Uniform, Bernoulli, Poisson, Gaussian, Exponential.

Also contain makeMeans, a small function to generate uniformly spacen means of arms.
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np

if __name__ != "__main__":
    from .Uniform import Uniform
    from .Bernoulli import Bernoulli
    from .Binomial import Binomial
    from .Poisson import Poisson, UnboundedPoisson
    from .Gaussian import Gaussian, UnboundedGaussian
    from .Exponential import Exponential, ExponentialFromMean, UnboundedExponential
    from .Gamma import Gamma, GammaFromMean, UnboundedGamma


def makeMeans(nbArms=3, delta=0.1, lower=0., amplitude=1.):
    """Return a list of means of arms, well spacen:

    - in [lower, lower + amplitude],
    - sorted in increasing order,
    - starting from lower + amplitude * delta, up to lower + amplitude * (1 - delta),
    - and there is nbArms arms.

    >>> makeMeans(2, 0.1)
    array([ 0.1,  0.9])
    >>> makeMeans(3, 0.1)
    array([ 0.1,  0.5,  0.9])
    >>> makeMeans(9, 1 / (1. + 9))
    array([ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])
    """
    assert nbArms >= 1, "Error: nbArms has to be >= 1."
    assert 0 < delta < 1, "Error: delta has to be in (0, 1)."
    # return list(lower + amplitude * np.linspace(delta, 1 - delta, nbArms))
    return lower + amplitude * np.linspace(delta, 1 - delta, nbArms)


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
