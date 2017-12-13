#!/usr/bin/env python
#-*- coding: utf8 -*-

import numpy as np
import matplotlib.pyplot as plt


def a(T):
    """ Left limit of the sum."""
    return 0

def b(T):
    """ Right limit of the sum."""
    return int(np.floor(np.log2(T)) - 1)


def f(i):
    """ Term to sum."""
    return (i * 2**i) ** 0.5

def computed(T, a=a, b=b, f=f):
    """ Compute the sum."""
    return sum(f(i) for i in range(a(T), b(T) + 1))

# def computed(T):
# return np.sum([f(i) for i in range(a(T), b(T) + 1)])


def predicted(T):
    """ Non-scaled prediction of the value of the sum (case specific)."""
    return (T * np.log(T)) ** 0.5


def main():
    """ Test function, compute and display the plot."""
    Ts = np.logspace(1, 80, 1000)
    # Ts = np.linspace(10, 1e20, 100)
    # Compare the computed and predicted values
    Xs = np.array([computed(T) for T in Ts])
    Ys = np.array([predicted(T) for T in Ts])
    # Hacky "Linear regression"
    alpha_min = np.min(Xs / Ys)
    alpha_mean = np.mean(Xs / Ys)
    alpha_max = np.max(Xs / Ys)
    # Figure
    fig = plt.figure()
    plt.xlabel("Times T")
    plt.ylabel("Bound")
    plt.title(r"Bound on regret : predicted values are $\alpha \sqrt{T \log{T}}$ for different $\alpha$")
    plt.plot(Ts, Xs, marker='+', lw=2, label="Computed values")
    plt.plot(Ts, alpha_min * Ys, marker='o', label=rf"Predicted values (min) $\alpha={alpha_min:.3g}$")
    plt.plot(Ts, alpha_mean * Ys, marker='s', label=rf"Predicted values (mean) $\alpha={alpha_mean:.3g}$")
    plt.plot(Ts, alpha_max * Ys, marker='d', label=rf"Predicted values (max) $\alpha={alpha_max:.3g}$")
    plt.legend()
    plt.show()
    return fig


if __name__ == '__main__':
    main()
