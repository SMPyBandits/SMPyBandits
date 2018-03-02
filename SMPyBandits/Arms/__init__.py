# -*- coding: utf-8 -*-
""" Arms : contains different types of bandit arms:
:class:`Constant`, :class:`UniformArm`, :class:`Bernoulli`, :class:`Binomial`, :class:`Poisson`, :class:`Gaussian`, :class:`Exponential`, :class:`Gamma`.

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

- :func:`uniformMeans`, to generate uniformly spaced means of arms.
- :func:`uniformMeansWithSparsity`, to generate uniformly spaced means of arms, with sparsity constraints.
- :func:`randomMeans`, to generate randomly spaced means of arms.
- :func:`randomMeansWithGapBetweenMbestMworst`, to generate randomly spaced means of arms, with a constraint on the gap between the M-best arms and the (K-M)-worst arms.
- :func:`randomMeans`, to generate randomly spaced means of arms.
- :func:`shuffled`, to return a shuffled version of a list.
- Utility functions :func:`array_from_str` :func:`list_from_str` and :func:`tuple_from_str` to obtain a `numpy.ndarray`, a `list` or a `tuple` from a string (used for the CLI env variables interface).
- :func:`optimal_selection_probabilities`.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from random import shuffle
from copy import copy
import json
import numpy as np

if __name__ != "__main__":
    from .Constant import Constant
    from .UniformArm import UniformArm
    from .Bernoulli import Bernoulli
    from .Binomial import Binomial
    from .Poisson import Poisson, UnboundedPoisson
    from .Gaussian import Gaussian, Gaussian_0_1, Gaussian_0_2, Gaussian_0_5, Gaussian_0_10, Gaussian_0_100, Gaussian_m1_1, Gaussian_m2_2, Gaussian_m5_5, Gaussian_m10_10, Gaussian_m100_100, UnboundedGaussian
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
        assert (nbArms * mingap) < (amplitude / 2.), "Error: 'mingap' = {:.3g} is too large, it might be impossible to find a vector of means with such a large gap for {} arms.".format(mingap, nbArms)  # DEBUG
        # while len(set(mus)) == nbArms and np.min(np.abs(np.diff(mus))) <= mingap:  # Ensure a min gap > mingap
        while np.min(np.abs(np.diff(mus))) <= mingap:  # Ensure a min gap > mingap
            mus = np.random.rand(nbArms)
    if isSorted:
        return sorted(list(lower + (amplitude * mus)))
    else:
        np.random.shuffle(mus)  # Useless
        return list(lower + (amplitude * mus))


def randomMeansWithGapBetweenMbestMworst(nbArms=3, mingap=None, nbPlayers=2, lower=0., amplitude=1., isSorted=True):
    """Return a list of means of arms, randomly sampled uniformly in [lower, lower + amplitude], with a min gap >= mingap between the set Mbest and Mworst.
    """
    assert nbArms >= 1, "Error: 'nbArms' = {} has to be >= 1.".format(nbArms)  # DEBUG
    assert amplitude > 0, "Error: 'amplitude' = {:.3g} has to be > 0.".format(amplitude)  # DEBUG
    mus = np.random.rand(nbArms)
    if mingap is not None and mingap > 0 and nbPlayers < nbArms:
        assert mingap < amplitude, "Error: 'mingap' = {:.3g} is too large, it might be impossible to find a vector of means with such a large gap for {} arms.".format(mingap, nbArms)  # DEBUG
        def gap(mus):
            sorted_mus = sorted(mus)
            mu_Mbest = sorted_mus[-nbPlayers]
            mu_Mworst = sorted_mus[-nbPlayers-1]
            return mu_Mbest - mu_Mworst
        # while len(set(mus)) == nbArms and gap(mus) <= mingap:  # Ensure a min gap > mingap
        while gap(mus) <= mingap:  # Ensure a min gap > mingap
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
        while len(set(mus)) == sparsity and np.min(np.abs(np.diff(mus))) <= mingap:  # Ensure a min gap > mingap
            mus = np.sort(np.random.rand(sparsity))
    bad_mus = [lower] * (nbArms - sparsity)
    good_mus = lowerNonZero + ((lower + amplitude - lowerNonZero) * mus)
    mus = list(bad_mus) + list(good_mus)
    if isSorted:
        return sorted(list(mus))
    else:
        return shuffled(list(mus))


def array_from_str(my_str):
    """Convert a string like "[0.1, 0.2, 0.3]" to a numpy array `[0.1, 0.2, 0.3]`, using safe `json.loads` instead of `exec`.

    >>> array_from_str("[0.1, 0.2, 0.3]")
    array([ 0.1,  0.2,  0.3])
    >>> array_from_str("0.1, 0.2, 0.3")
    array([ 0.1,  0.2,  0.3])
    >>> array_from_str("0.9")
    array(0.9)
    """
    # print("array_from_str called with my_str =", my_str)  # DEBUG
    if my_str is None or isinstance(my_str, np.ndarray):
        return my_str
    try:
        if not ('[' in my_str and ']' in my_str):
            my_str = '[%s]' % my_str
        dict_str = '{"XXX": %s}' % my_str
        fake_dict = json.loads(dict_str)
        return np.array(fake_dict["XXX"])
    except:
        print("Error while interpreting the string {} as an array...".format(my_str))  # DEBUG
        return None


def list_from_str(my_str):
    """Convert a string like "[0.1, 0.2, 0.3]" to a list `(0.1, 0.2, 0.3)`, using safe `json.loads` instead of `exec`.

    >>> list_from_str("[0.1, 0.2, 0.3]")
    [0.1, 0.2, 0.3]
    >>> list_from_str("0.1, 0.2, 0.3")
    [0.1, 0.2, 0.3]
    >>> list_from_str("0.9")
    [0.9]
    """
    # print("list_from_str called with my_str =", my_str)  # DEBUG
    if my_str is None:
        return my_str
    if isinstance(my_str, (tuple, list)):
        return list(my_str)
    try:
        if not ('[' in my_str and ']' in my_str):
            my_str = '[%s]' % my_str
        dict_str = '{"XXX": %s}' % my_str
        fake_dict = json.loads(dict_str)
        return np.array(fake_dict["XXX"]).tolist()
    except:
        print("Error while interpreting the string {} as a list...".format(my_str))  # DEBUG
        return None


def tuple_from_str(my_str):
    """Convert a string like "[0.1, 0.2, 0.3]" to a tuple `(0.1, 0.2, 0.3)`, using safe `json.loads` instead of `exec`.

    >>> tuple_from_str("[0.1, 0.2, 0.3]")
    (0.1, 0.2, 0.3)
    >>> tuple_from_str("0.1, 0.2, 0.3")
    (0.1, 0.2, 0.3)
    >>> tuple_from_str("0.9")
    (0.9,)
    """
    # print("tuple_from_str called with my_str =", my_str)  # DEBUG
    if my_str is None:
        return my_str
    if isinstance(my_str, (tuple, list)):
        return tuple(my_str)
    try:
        if not ('[' in my_str and ']' in my_str):
            my_str = '[%s]' % my_str
        dict_str = '{"XXX": %s}' % my_str
        fake_dict = json.loads(dict_str)
        return tuple(np.array(fake_dict["XXX"]).tolist())
    except:
        print("Error while interpreting the string {} as a tuple...".format(my_str))  # DEBUG
        return None


def optimal_selection_probabilities(M, mu):
    r""" Compute the optimal selection probabilities of K arms of means :math:`\mu_i` by :math:`1 \leq M \leq K` players, if they all observe each other pulls and rewards, as derived in (15) p3 of [[The Effect of Communication on Noncooperative Multiplayer Multi-Armed Bandit Problems, by Noyan Evirgen, Alper Kose, IEEE ICMLA 2017]](https://arxiv.org/abs/1711.01628v1).

    .. warning:: They consider a different collision model than I usually do, when two (or more) players ask for the same resource at same time t, I usually consider than all the colliding players receive a zero reward (see :func:`Environment.CollisionModels.onlyUniqUserGetsReward`), but they consider than exactly one of the colliding players gets the reward, and all the others get a zero reward (see :func:`Environment.CollisionModels.rewardIsSharedUniformly`).

    Example:

    >>> optimal_selection_probabilities(3, [0.1,0.1,0.1])
    array([ 0.33333333,  0.33333333,  0.33333333])

    >>> optimal_selection_probabilities(3, [0.1,0.2,0.3])  # weird ? not really...
    array([ 0.        ,  0.43055556,  0.56944444])

    >>> optimal_selection_probabilities(3, [0.1,0.3,0.9])  # weird ? not really...
    array([ 0.        ,  0.45061728,  0.54938272])

    >>> optimal_selection_probabilities(3, [0.7,0.8,0.9])
    array([ 0.15631866,  0.35405647,  0.48962487])

    .. note:: These results may sound counter-intuitive, but again they use a different collision models: in my usual collision model, it makes no sense to completely drop an arm when K=M=3, no matter the probabilities :math:`\mu_i`, but in their collision model, a player wins more (in average) if she has a :math:`50\%` chance of being alone on an arm with mean :math:`0.3` than if she is sure to be alone on an arm with mean :math:`0.1` (see examples 3 and 4).
    """
    K = len(mu)
    assert 1 <= M <= K, "Error: number of arm M must be 1 <= M <= K but M = {} and K = {}.".format(M, K)  # DEBUG
    mup = np.asarray(mu) ** (M - 1)
    c = 1. - ((K - 1) / (np.sum(1. / mup))) / (mup)
    c[c <= 0] = 0
    c /= np.sum(c)
    return c


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
