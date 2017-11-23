# -*- coding: utf-8 -*-
""" Plot the C(mu) Lai & Robbins term and the HOI(mu) OI factor for various Bernoulli MAB problem."""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

from itertools import product
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from .usenumba import jit
from .usetqdm import tqdm
from .plotsettings import maximizeWindow, legend

# Local imports
from Arms import *

oneLR = Bernoulli.oneLR
oneHOI = Bernoulli.oneHOI


@jit
def cmu(mu):
    """One LR term for Bernoulli problems."""
    best = max(mu)
    return sum(oneLR(best, m) for m in mu if m != best)


@jit
def oi(mu):
    """One HOI term for Bernoulli problems."""
    best = max(mu)
    return sum(oneHOI(best, m) for m in mu if m != best) / float(len(mu))


def addit(c, o, mu):
    """Add cmu(mu) to c and o(mu) to c if mu are not all equal."""
    if len(set(mu)) > 1:
        c.append(cmu(mu))
        o.append(oi(mu))


def main(K, N=50000, T=10):
    """Plot."""
    print("Starting for K =", K)

    c1, o1 = [], []
    for _ in tqdm(range(N), desc="Uniformly random (%d)" % N):
        mu = np.random.random(K)
        addit(c1, o1, mu)
    print("c: min =", min(c1), "max =", max(c1))
    print("o: min =", min(o1), "max =", max(o1))

    c2, o2 = [], []
    for _ in tqdm(range(N), desc="Gaussian (%d)" % N):
        mu = np.minimum(1, np.maximum(0, np.random.normal(loc=0.5, scale=0.2, size=K)))
        addit(c2, o2, mu)
    print("c: min =", min(c2), "max =", max(c2))
    print("o: min =", min(o2), "max =", max(o2))

    c3, o3 = [], []
    for mu in tqdm(product(np.linspace(0, 1, T), repeat=K), desc="Evenly spacen (%d)" % (T**K)):
        addit(c3, o3, mu)
    print("c: min =", min(c3), "max =", max(c3))
    print("o: min =", min(o3), "max =", max(o3))

    # for method in [plt.plot, plt.semilogx]:
    for method in [plt.semilogx]:
        plt.figure()
        method(c1, o1, 'o', ms=2, label="Uniform")
        method(c2, o2, 'x', ms=2, label="Gaussian")
        method(c3, o3, 'd', ms=2, label="Evenly spacen")
        legend()
        plt.xlabel(r"Lai & Robbins complexity constant, $C_{\mu}$")
        plt.ylabel(r"Navikkumar Modi HOI factor, $H_{OI}(\mu)$")
        plt.title("Comparison of two complexity criterion, for Bernoulli MAB problems, with $K = {}$ arms.".format(K))
        maximizeWindow()
        plt.show()


if __name__ == '__main__':
    for K in [3, 5, 7]:
        main(K)
