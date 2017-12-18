#!/usr/bin/env python
#-*- coding: utf8 -*-

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt

np.seterr(
    divide='ignore',
    over='ignore',
)  # XXX dangerous in general, controlled here!


def logb(b, x):
    """ Logarithm in a certain basis."""
    # print(f"logb : b = {b:>3g} and x = {x:>3g}")  # DEBUG
    return np.log(x) / np.log(b)


def gamma_delta(T0, alpha, beta):
    """ Compute the values of gamma and delta from T0, alpha, beta."""
    assert beta > 1
    assert alpha > 0
    gamma = 1 / (alpha ** (1 / (beta - 1)))
    delta = T0 / gamma
    return gamma, delta


# FIXME Change here the values
def initialize(T0=10, alpha=1, beta=2, debug=False):
    assert isinstance(T0, int) and T0 >= 1
    if debug: print(f"T0    = {T0:>8g}")
    assert alpha >= 1
    if debug: print(f"alpha = {alpha:>8g}")
    assert beta > 1
    if debug: print(f"beta  = {beta:>8g}")

    gamma, delta = gamma_delta(T0, alpha, beta)
    if debug: print(f"gamma = {gamma:>8g}")
    if debug: print(f"delta = {delta:>8g}")

    x = np.sqrt(delta)
    if debug: print(f"x     = {x:>8g}")
    return T0, alpha, beta, gamma, delta, x


def maintest(T0, alpha, beta):
    T0, alpha, beta, gamma, delta, x = initialize(T0, alpha, beta)


    def a(T):
        """ Left limit of the sum."""
        return 0

    def b(T):
        """ Right limit of the sum."""
        return int(np.ceil(logb(beta, logb(delta, T / gamma)))) - 1

    def b_i(i):
        """ Right limit of the sum."""
        return int(i)


    def f1(i):
        """ Term to sum."""
        return x ** (beta ** i)


    def f2(i):
        """ Term to sum."""
        return (x ** (beta ** i)) * (np.sqrt(beta) ** i)


    def computed(f):
        """ Compute the sum for a generic function f."""
        def s(T, a, b):
            return sum(f(i) for i in range(a(T), b(T) + 1))
        return s


    def compute_r1r2r3(Ts, b, debug=False):
        # Compare the computed and predicted values
        X1s = np.array([computed(f1)(T, a, b) for T in Ts])
        Y1s = np.array([f1(b(T)) for T in Ts])
        X2s = np.array([computed(f2)(T, a, b) for T in Ts])
        Y2s = np.array([f2(b(T)) for T in Ts])
        # Valeurs
        if debug: print(f"\nPour i de a = 0 à b = T :")
        for T, s1, s2 in zip(Ts, X1s, X2s):
            if np.isnan(s1) or np.isnan(s2): break
            if ~np.isfinite(s1) or ~np.isfinite(s2): break
            if debug: print(f"Pour T = {T:>3g}, la somme avec x^(b^i) vaut = {s1:>8g} et la somme avec x^(b^i) * sqrt(b)^i vaut = {s2:>8g}.")
        # Ratios
        if debug: print(f"\nPour i de a = 0 à b = T :")

        # XXX weird behavior, have to do the catch_warnings instead of try: ... except RuntimeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratio1s, ratio2s, ratio3s = X1s / Y1s, X2s / Y2s, X1s / X2s

        ratio1s, ratio2s, ratio3s = ratio1s[~np.isnan(ratio1s)], ratio2s[~np.isnan(ratio2s)], ratio3s[~np.isnan(ratio3s)]
        for T, r1, r2, r3 in zip(Ts, ratio1s, ratio2s, ratio3s):
            if np.isnan(r1) or np.isnan(r2) or np.isnan(r3): break
            if debug: print(f"Pour T = {T:>3g}, le ratio avec x^(b^i) vaut = {r1:>8g} et le ratio avec x^(b^i) * sqrt(b)^i vaut = {r2:>8g} et le ratio entre les deux sommes vaut {r3:>8g}.")
        # Conclure
        if debug: print(f"\n\nAvec T0 = {T0:>3g}, alpha = {alpha:>3g}, beta = {beta:>3g}, gamma = {gamma:>3g}, delta = {delta:>3g}, x = {x:>3g} ...")
        if debug: print(f"Pour T de {np.min(Ts)} à {np.max(Ts)} ...")
        r1, r2, r3 = np.max(ratio1s), np.max(ratio2s), np.max(ratio3s)
        if debug: print(f"Maximum ratio avec x^(b^i) vaut = {r1:>8g} et maximum ratio avec x^(b^i) * sqrt(b)^i vaut = {r2:>8g} et maximum ratio entre les deux sommes vaut = {r3:>8g}.")
        # DONE
        return r1, r2, r3


    def test2(debug=True):
        if debug: print(f"\nAvec T0 = {T0:>3g}, alpha = {alpha:>3g}, beta = {beta:>3g}, gamma = {gamma:>3g}, delta = {delta:>3g}, x = {x:>3g} ...")

        # Ts = np.logspace(1, 80, 100)
        # # Ts = np.linspace(10, 1e20, 100)
        # # Ts = np.linspace(10, 100, 10)
        # test(Ts, b)

        Ts = np.arange(1, 50)
        r1, r2, r3 = compute_r1r2r3(Ts, b_i)
        if debug: print(f"Maximum ratio avec x^(b^i) vaut = {r1:>8g} et maximum ratio avec x^(b^i) * sqrt(b)^i vaut = {r2:>8g} et maximum ratio entre les deux sommes vaut = {r3:>8g}.")

        return r1, r2, r3

    return test2


def main():
    T0_s = [
        10,
        100,
        1000,
        10000,
    ]
    alpha_s = [
        1,
        2,
    ]
    beta_s = [
        2,
        1.5,
        1.1,
        1.01,
        1.001,
        1.00001,
    ]

    # for T0, alpha, beta in zip(T0_s, alpha_s, beta_s):
    for T0, alpha, beta in itertools.product(T0_s, alpha_s, beta_s):
        test2 = maintest(T0, alpha, beta)
        r1, r2, r3 = test2()


if __name__ == '__main__':
    main()
