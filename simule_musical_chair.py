#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Small script to simulate musical chair plays.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.6"


try:
    from tqdm import trange as tqdm_range
except ImportError:
    print("ERROR")
    def tqdm_range(N, desc=""):
        return range(N)

import numpy as np
from collections import Counter
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# use a clever color palette, eg http://seaborn.pydata.org/api.html#color-palettes
sns.set(context="talk", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.05)

DPI = 110  #: DPI to use for the figures
FIGSIZE = (19.80, 10.80)  #: Figure size, in inches!

# Use tex by default http://matplotlib.org/2.0.0/users/dflt_style_changes.html#math-text
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.sans-serif'] = "DejaVu Sans"
mpl.rcParams['mathtext.fontset'] = "cm"
mpl.rcParams['mathtext.rm'] = "serif"

# Configure the DPI of all images, once and for all!
mpl.rcParams['figure.dpi'] = DPI
# print(" - Setting dpi of all figures to", DPI, "...")

# Configure figure size, even of if saved directly and not displayed, use HD screen
# cf. https://en.wikipedia.org/wiki/Computer_display_standard
mpl.rcParams['figure.figsize'] = FIGSIZE
# print(" - Setting 'figsize' of all figures to", FIGSIZE, "...")


def title(s):
    print(s)
    plt.title(s)


def new_chair(M):
    return np.random.randint(M)


def first_chairs(M):
    return [new_chair(M) for i in range(M)]


def colliding_players(M, chairs):
    counts = np.bincount(chairs, minlength=M)
    return [i for i in range(M) if counts[chairs[i]] > 1]


def simulate_musical_chair(M=1, sticky=False, DEBUG=False):
    if DEBUG: print("==> Starting a new musical chair game with", M, "players...")
    t = 1
    all_players = set(range(M))
    collisions = 0
    chairs = first_chairs(M)
    sitted = np.full(M, False)
    if DEBUG: print("At time t =", t, "the players are on", chairs)
    while len(set(chairs)) != M:
        wrong_guys = colliding_players(M, chairs)
        collisions += len(wrong_guys)
        if DEBUG: print("Not yet orthogonal, drawing new chairs for the colliding players", wrong_guys, "...")
        # Sticky version
        if sticky:
            # all guys alone will sit and not move
            for i in all_players - set(wrong_guys):
                sitted[i] = True
            # all colliding non-sited guys will change chairs
            for i in wrong_guys:
                if sitted[i]:
                    pass
                else:
                    chairs[i] = new_chair(M)
        # Non sticky
        else:
            for i in wrong_guys:
                chairs[i] = new_chair(M)
        t += 1
        if DEBUG: print("At time t =", t, "the players are on", chairs)
    return t, collisions


def simulate_count_and_print(M, nbRepetitions=5000, sticky=False, DEBUG=False):
    if DEBUG: print("Starting", nbRepetitions, "simulations with M =", M, "players...")
    results = [ None ] * nbRepetitions
    for i in tqdm_range(nbRepetitions, desc="Repetitions"):
        results[i] = simulate_musical_chair(M, sticky=sticky)
    # First for absorption time
    absorption_times = [ t for t,_ in results ]
    count = Counter(absorption_times)
    if DEBUG:
        print("It gave a count of absorption times =", count)
        for t in count:
            print("    Absorption time t =", t, "was seen", count[t], "times, =", count[t] / float(nbRepetitions), "%.")
    # Now for collisions
    collisions = [ c for _,c in results ]
    count = Counter(collisions)
    if DEBUG:
        print("It gave a count of collisions =", count)
        for t in count:
            print("    Collisions =", t, "was seen", count[t], "times, =", count[t] / float(nbRepetitions), "%.")
    # done
    return absorption_times, collisions


def plot_absorption(M, absorption_times, sticky=False):
    plt.figure()
    plt.hist(absorption_times, bins=max(absorption_times), normed=True)
    plt.xticks(sorted(list(set(absorption_times))))
    title("{}Musical Chairs, repartitions of the absorption times for M = {} players, mean = {}".format("Sticky " if sticky else "", M, np.mean(absorption_times)))
    plt.show()


def plot_collisions(M, collisions, sticky=False):
    plt.figure()
    plt.hist(collisions, bins=max(collisions), normed=True)
    plt.xticks(sorted(list(set(collisions))))
    title("{}Musical Chairs, repartitions of the collisions for M = {} players, mean = {}".format("Sticky " if sticky else "", M, np.mean(collisions)))
    plt.show()


def plot_all(M, nbRepetitions=1000, sticky=False):
    absorption_times, collisions = simulate_count_and_print(M, nbRepetitions=nbRepetitions, sticky=sticky)
    plot_absorption(M, absorption_times, sticky=sticky)
    plot_collisions(M, collisions, sticky=sticky)


def main(M, sticky=False):
    plot_all(M, sticky=sticky)

if __name__ == '__main__':
    from sys import argv
    M = int(argv[1]) if len(argv) > 1 else 3
    main(M, sticky=False)
    main(M, sticky=True)