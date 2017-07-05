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

if __name__ == '__main__':
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


def simulate_musical_chair(M=1, sticky=True, DEBUG=False):
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


def simulate_count_and_print(M, nbRepetitions=5000, sticky=True, DEBUG=False):
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


def plot_one_distr(M, data, sticky=True, onlyTitle=False, name="??", nbRepetitions=5000, savefig=False):
    str_title = "{}Musical Chairs, repartitions of the {} for M = {} players, mean = {}, for {} repetitions".format("Sticky " if sticky else "", name, M, np.mean(data), nbRepetitions)
    if onlyTitle:
        print(str_title)
        return
    plt.figure()
    plt.hist(data, bins=max(data), normed=True)
    plt.xticks(sorted(list(set(data))))
    title(str_title)
    if savefig:
        print("Saving to plots/" + savefig + ".png at dpi =", DPI, "...")  # DEBUG
        plt.savefig("plots/" + savefig + ".png", dpi=DPI)
        print("Saving to plots/" + savefig + ".pdf at dpi =", DPI, "...")  # DEBUG
        plt.savefig("plots/" + savefig + ".pdf", dpi=DPI)
    plt.show()


def plot_all_data(maxM, all_data, sticky=True, name="??", nbRepetitions=5000, savefig=False):
    plt.figure()
    plt.plot(all_data, 'ro-', lw=3)
    title("{}Musical Chairs, repartitions of mean of the {} for M = 1...{} players, for {} repetitions".format("Sticky " if sticky else "", name, maxM, nbRepetitions))
    plt.ylabel("Average number of {}".format(name))
    plt.xlabel("Evolution of number of players M = 1 ... {}".format(maxM))
    if savefig:
        print("Saving to plots/" + savefig + ".png at dpi =", DPI, "...")  # DEBUG
        plt.savefig("plots/" + savefig + ".png", dpi=DPI)
        print("Saving to plots/" + savefig + ".pdf at dpi =", DPI, "...")  # DEBUG
        plt.savefig("plots/" + savefig + ".pdf", dpi=DPI)
    plt.show()


def plot_one(M=5, nbRepetitions=10000, sticky=True, onlyTitle=False):
    absorption_times, collisions = simulate_count_and_print(M, nbRepetitions=nbRepetitions, sticky=sticky)
    plot_one_distr(M, absorption_times, sticky=sticky, name="absorbtion times", onlyTitle=onlyTitle)
    plot_one_distr(M, collisions, sticky=sticky, name="collisions", onlyTitle=onlyTitle)
    return absorption_times, collisions


def plot_evolution_of_both(maxM=20, nbRepetitions=5000, sticky=True):
    all_absorption_times = []
    all_collisions = []
    # for M in range(maxM):
    for M in tqdm_range(maxM, desc="Value of M"):
        absorption_times, collisions = plot_one(1 + M, nbRepetitions=nbRepetitions, sticky=sticky, onlyTitle=True)
        all_absorption_times.append(np.mean(absorption_times))
        all_collisions.append(np.mean(collisions))
    mainfig = "{}Musical_Chairs__Repartitions_of_XXX__until_maxM={}__{}_repetitions".format("Sticky_" if sticky else "", maxM, nbRepetitions)
    savefig = mainfig.replace("XXX", "absorption_times")
    plot_all_data(maxM, all_absorption_times, sticky=sticky, name="absorbtion times", nbRepetitions=nbRepetitions, savefig=savefig)
    savefig = mainfig.replace("XXX", "collisions")
    plot_all_data(maxM, all_collisions, sticky=sticky, name="collisions", nbRepetitions=nbRepetitions, savefig=savefig)


def main(M, nbRepetitions=5000, sticky=True, evolution=False):
    if evolution:
        plot_evolution_of_both(maxM=M, sticky=sticky, nbRepetitions=nbRepetitions)
    else:
        plot_one(M=M, sticky=sticky, nbRepetitions=nbRepetitions)

if __name__ == '__main__':
    # Durty command line argument handling
    from sys import argv
    M = int(argv[1]) if len(argv) > 1 else 3
    nbRepetitions = int(argv[2]) if len(argv) > 2 else 5000
    sticky = not("nonsticky" in argv[1:])
    evolution = "evolution" in argv[1:]
    main(M, nbRepetitions=nbRepetitions, sticky=sticky, evolution=evolution)