# -*- coding: utf-8 -*-
""" Arms : contains different types of bandit arms:
:class:`Constant`, :class:`Uniform`, :class:`Bernoulli`, :class:`Binomial`, :class:`Poisson`, :class:`Gaussian`, :class:`Exponential`, :class:`Gamma`.

Each arm class follows the same interface:

>>> my_arm = Arm(params)
>>> my_arm.mean
0.5
>>> my_arm.draw()  # one random draw
0.0
>>> my_arm.draw_nparray(20)  # or ((3, 10)), many draw
array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,
        1.,  0.,  0.,  0.,  1.,  1.,  1.])


Also contains:

- :func:`uniformMeans`, a small function to generate uniformly spaced means of arms.
- :func:`uniformMeansWithSparsity`, a small function to generate uniformly spaced means of arms, with sparsity constraints.
- :func:`randomMeans`, generate randomly spaced means of arms.
- :func:`randomMeansWithSparsity`, generate randomly spaced means of arms, with sparsity constraints.
- :func:`shuffled`, to return a shuffled version of a list.
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

from random import shuffle
from copy import copy
import numpy as np

if __name__ != "__main__":
    from .Constant import Constant
    from .Uniform import Uniform
    from .Bernoulli import Bernoulli
    from .Binomial import Binomial
    from .Poisson import Poisson, UnboundedPoisson
    from .Gaussian import Gaussian, UnboundedGaussian
    from .Exponential import Exponential, ExponentialFromMean, UnboundedExponential
    from .Gamma import Gamma, GammaFromMean, UnboundedGamma


def shuffled(mylist):
    """Returns a shuffled version of the input 1D list. sorted() exists instead of list.sort(), but shuffled() does not exist instead of random.shuffle()...

    >>> from random import seed; seed(1234)  # reproducible results
    >>> mylist = [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9]
    >>> shuffled(mylist)
    [0.9, 0.4, 0.3, 0.6, 0.5, 0.7, 0.1, 0.2, 0.8]
    >>> shuffled(mylist)
    [0.4, 0.3, 0.7, 0.5, 0.8, 0.1, 0.9, 0.6, 0.2]
    >>> shuffled(mylist)
    [0.4, 0.6, 0.9, 0.5, 0.7, 0.2, 0.1, 0.3, 0.8]
    >>> shuffled(mylist)
    [0.8, 0.7, 0.3, 0.1, 0.9, 0.5, 0.6, 0.2, 0.4]
    """
    copiedlist = copy(mylist)
    shuffle(copiedlist)
    return copiedlist


def uniformMeans(nbArms=3, delta=0.1, lower=0., amplitude=1., isSorted=True):
    """Return a list of means of arms, well spaced:

    - in [lower, lower + amplitude],
    - sorted in increasing order,
    - starting from lower + amplitude * delta, up to lower + amplitude * (1 - delta),
    - and there is nbArms arms.

    >>> np.array(uniformMeans(2, 0.1))
    array([ 0.1,  0.9])
    >>> np.array(uniformMeans(3, 0.1))
    array([ 0.1,  0.5,  0.9])
    >>> np.array(uniformMeans(9, 1 / (1. + 9)))
    array([ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])
    """
    assert nbArms >= 1, "Error: 'nbArms' = {} has to be >= 1.".format(nbArms)  # DEBUG
    assert amplitude > 0, "Error: 'amplitude' = {:.3g} has to be > 0.".format(amplitude)  # DEBUG
    assert 0. < delta < 1., "Error: 'delta' = {:.3g} has to be in (0, 1).".format(delta)  # DEBUG
    mus = lower + amplitude * np.linspace(delta, 1 - delta, nbArms)
    if isSorted:
        return sorted(list(mus))
    else:
        return shuffled(list(mus))


def uniformMeansWithSparsity(nbArms=10, sparsity=3, delta=0.1, lower=0., lowerNonZero=0.5, amplitude=1., isSorted=True):
    """Return a list of means of arms, well spaced, in [lower, lower + amplitude].

    - Exactly ``nbArms-sparsity`` arms will have a mean = ``lower`` and the others are randomly sampled uniformly in [lowerNonZero, lower + amplitude].
    - All means will be different, except if ``mingap=None``, with a min gap > 0.

    >>> import numpy as np; np.random.seed(1234)  # reproducible results
    >>> np.array(uniformMeansWithSparsity(nbArms=6, sparsity=2))  # doctest: +ELLIPSIS
    array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.55,  0.95])
    >>> np.array(uniformMeansWithSparsity(nbArms=6, sparsity=2, lowerNonZero=0.8, delta=0.03))  # doctest: +ELLIPSIS
    array([ 0.   ,  0.   ,  0.   ,  0.   ,  0.806,  0.994])
    >>> np.array(uniformMeansWithSparsity(nbArms=10, sparsity=2))  # doctest: +ELLIPSIS
    array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.55,  0.95])
    >>> np.array(uniformMeansWithSparsity(nbArms=6, sparsity=2, delta=0.05))  # doctest: +ELLIPSIS
    array([ 0.   ,  0.   ,  0.   ,  0.   ,  0.525,  0.975])
    >>> np.array(uniformMeansWithSparsity(nbArms=10, sparsity=4, delta=0.05))  # doctest: +ELLIPSIS
    array([ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.525,  0.675,
            0.825,  0.975])
    """
    assert nbArms >= 1, "Error: 'nbArms' = {} has to be >= 1.".format(nbArms)  # DEBUG
    assert amplitude > 0, "Error: 'amplitude' = {:.3g} has to be > 0.".format(amplitude)  # DEBUG
    assert 0. < delta < 1., "Error: 'delta' = {:.3g} has to be in (0, 1).".format(delta)  # DEBUG
    assert 0 <= sparsity <= nbArms, "Error: 'sparsity' = {} has to be 0 <= sparsity <= nbArms = {} ...".format(sparsity, nbArms)  # DEBUG
    assert lower < lowerNonZero, "Error: 'lower' = {:.3g} has to be < 'lowerNonZero' = {:.3g} ...".format(lower, lowerNonZero)  # DEBUG
    mus = np.sort(np.random.rand(sparsity))
    bad_mus = [lower] * (nbArms - sparsity)
    good_mus = list(lowerNonZero + (lower + amplitude - lowerNonZero) * np.linspace(delta, 1 - delta, sparsity))
    mus = list(bad_mus) + list(good_mus)
    if isSorted:
        return sorted(list(mus))
    else:
        return shuffled(list(mus))


def randomMeans(nbArms=3, mingap=None, lower=0., amplitude=1., isSorted=True):
    """Return a list of means of arms, randomly sampled uniformly in [lower, lower + amplitude], with a min gap >= mingap.

    - All means will be different, except if ``mingap=None``, with a min gap > 0.

    >>> import numpy as np; np.random.seed(1234)  # reproducible results
    >>> randomMeans(nbArms=3, mingap=0.05)  # doctest: +ELLIPSIS
    [0.191..., 0.437..., 0.622...]
    >>> randomMeans(nbArms=3, mingap=0.1)  # doctest: +ELLIPSIS
    [0.276..., 0.801..., 0.958...]

    - Means are sorted, except if ``isSorted=False``.

    >>> import random; random.seed(1234)  # reproducible results
    >>> randomMeans(nbArms=5, mingap=0.1, isSorted=True)  # doctest: +ELLIPSIS
    [0.006..., 0.229..., 0.416..., 0.535..., 0.899...]
    >>> randomMeans(nbArms=5, mingap=0.1, isSorted=False)  # doctest: +ELLIPSIS
    [0.419..., 0.932..., 0.072..., 0.755..., 0.650...]
    """
    assert nbArms >= 1, "Error: 'nbArms' = {} has to be >= 1.".format(nbArms)  # DEBUG
    assert amplitude > 0, "Error: 'amplitude' = {:.3g} has to be > 0.".format(amplitude)  # DEBUG
    mus = np.random.rand(nbArms)
    if mingap is not None and mingap > 0:
        assert nbArms * 2 * mingap < amplitude, "Error: 'mingap' = {:.3g} is too large, it might be impossible to find a vector of means with such a large gap for {} arms.".format(mingap, nbArms)  # DEBUG
        while len(set(mus)) == nbArms and np.min(np.diff(mus)) <= mingap:  # Ensure a min gap > mingap
            mus = np.random.rand(nbArms)
    if isSorted:
        return sorted(list(lower + (amplitude * mus)))
    else:
        np.random.shuffle(mus)  # Useless
        return list(lower + (amplitude * mus))


def randomMeansWithSparsity(nbArms=10, sparsity=3, mingap=0.01, lower=0., lowerNonZero=0.5, amplitude=1., isSorted=True):
    """Return a list of means of arms, in [lower, lower + amplitude], with a min gap >= mingap.

    - Exactly ``nbArms-sparsity`` arms will have a mean = ``lower`` and the others are randomly sampled uniformly in [lowerNonZero, lower + amplitude].
    - All means will be different, except if ``mingap=None``, with a min gap > 0.

    >>> import numpy as np; np.random.seed(1234)  # reproducible results
    >>> randomMeansWithSparsity(nbArms=6, sparsity=2, mingap=0.05)  # doctest: +ELLIPSIS
    [0.0, 0.0, 0.0, 0.0, 0.595..., 0.811...]
    >>> randomMeansWithSparsity(nbArms=6, sparsity=2, mingap=0.1)  # doctest: +ELLIPSIS
    [0.0, 0.0, 0.0, 0.0, 0.718..., 0.892...]

    - Means are sorted, except if ``isSorted=False``.

    >>> import random; random.seed(1234)  # reproducible results
    >>> randomMeansWithSparsity(nbArms=6, sparsity=2, mingap=0.1, isSorted=True)  # doctest: +ELLIPSIS
    [0.0, 0.0, 0.0, 0.0, 0.636..., 0.889...]
    >>> randomMeansWithSparsity(nbArms=6, sparsity=2, mingap=0.1, isSorted=False)  # doctest: +ELLIPSIS
    [0.0, 0.0, 0.900..., 0.638..., 0.0, 0.0]
    """
    assert nbArms >= 1, "Error: 'nbArms' = {} has to be >= 1.".format(nbArms)  # DEBUG
    assert amplitude > 0, "Error: 'amplitude' = {:.3g} has to be > 0.".format(amplitude)  # DEBUG
    assert 0 <= sparsity <= nbArms, "Error: 'sparsity' = {} has to be 0 <= sparsity <= nbArms = {} ...".format(sparsity, nbArms)  # DEBUG
    assert lower < lowerNonZero, "Error: 'lower' = {:.3g} has to be < 'lowerNonZero' = {:.3g} ...".format(lower, lowerNonZero)  # DEBUG
    mus = np.sort(np.random.rand(sparsity))
    if mingap is not None and mingap > 0:
        while len(set(mus)) == sparsity and np.min(np.diff(mus)) <= mingap:  # Ensure a min gap > mingap
            mus = np.sort(np.random.rand(sparsity))
    bad_mus = [lower] * (nbArms - sparsity)
    good_mus = lowerNonZero + ((lower + amplitude - lowerNonZero) * mus)
    mus = list(bad_mus) + list(good_mus)
    if isSorted:
        return sorted(list(mus))
    else:
        return shuffled(list(mus))


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
