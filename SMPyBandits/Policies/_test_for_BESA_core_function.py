# -*- coding: utf-8 -*-
""" Test of the core function of BESA algorithm.

$ ipython
In [1]: run _test_for_BESA_core_function.py

In [2]: %timeit manualbranching(random_samples(a, mu_a, N, 2 * N), random_samples(b, mu_b, N, 2 * N))
46.3 µs ± 3.95 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

In [3]: %timeit numpytest(random_samples(a, mu_a, N, 2 * N), random_samples(b, mu_b, N, 2 * N))
61.9 µs ± 6.76 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"
import numpy as np
import timeit


def manualbranching(tuple_a, tuple_b):
    Na, mean_a, a = tuple_a
    Nb, mean_b, b = tuple_b
    if mean_a > mean_b:
        return a
    elif mean_a < mean_b:
        return b
    else:
        if Na < Nb:
            return a
        elif Na > Nb:
            return b
        else:  # if no way of breaking the tie, choose uniformly at random
            return np.random.choice([a, b])


def numpytest(tuple_a, tuple_b):
    Na, mean_a, samples_a, a = tuple_a
    Nb, mean_b, samples_b, b = tuple_b
    if mean_a != mean_b:
        return [a, b][np.argmax([mean_a, mean_b])]
    else:
        return [a, b][np.argmin([Na, Nb])]


def random_samples(i, mu, N1, N2):
    N1, N2 = min(N1, N2), max(N1, N2)
    N = np.random.randint(N1, high=N2)
    samples = np.asarray(np.random.binomial(1, mu, N), dtype=float)
    mean = np.mean(samples)
    return N, mean, samples, i


def main(N=10, mu_a=0.5, mu_b=0.5):
    a, b = 0, 1
    print("For the function 'manualbranching' run:")
    print("%timeit manualbranching(random_samples(a, mu_a, N, 2 * N), random_samples(b, mu_b, N, 2 * N))")
    print("For the function 'numpytest' run:")
    print("%timeit numpytest(random_samples(a, mu_a, N, 2 * N), random_samples(b, mu_b, N, 2 * N))")


if __name__ == '__main__':
    main()
