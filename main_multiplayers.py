#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them, for the multi-players case.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.1"

# Generic imports
from os import mkdir
import os.path
import matplotlib.pyplot as plt
# Local imports
from Environment import EvaluatorMultiPlayers
from configuration_multiplayers import configuration

# Parameters for the plots (where to save them) and what to draw
plot_dir = "plots"
semilogx = False
piechart = True
# Whether to do the plots or not
do_plot = False
do_plot = True
# Whether to show all plots, or one by one
interactive = True
interactive = False  # Seems to be the only mode which is working well


if __name__ == '__main__':
    if os.path.isdir(plot_dir):
        print("{} is already a directory here...".format(plot_dir))
    elif os.path.isfile(plot_dir):
        raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
    else:
        mkdir(plot_dir)
    # (almost) unique hash from the configuration
    hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))
    evaluation = EvaluatorMultiPlayers(configuration)
    # Start the evaluation and then print final ranking and plot, for each environment
    M = evaluation.nbPlayers
    N = len(evaluation.envs)
    for envId, env in enumerate(evaluation.envs):
        # Evaluate just that env
        evaluation.start_one_env(envId, env)
        # Display the final rankings for that env
        print("Giving the final ranks ...")
        evaluation.printFinalRanking(envId)
        if not do_plot:
            break

        # Sub folder with a useful name
        subfolder = "MP__M{}_T{}_N{}__{}_algos".format(M, configuration['horizon'], configuration['repetitions'], len(configuration['players']))
        # Get the name of the output file
        imagename = "main____env{}-{}_{}.png".format(envId + 1, N, hashvalue)
        # Create the sub folder
        plot_dir = os.path.join(plot_dir, subfolder)
        if os.path.isdir(plot_dir):
            print("{} is already a directory here...".format(plot_dir))
        elif os.path.isfile(plot_dir):
            raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
        else:
            mkdir(plot_dir)

        # Set plotting mode to interactive
        if interactive:
            plt.interactive(True)

        savefig = os.path.join(plot_dir, imagename)
        # Plotting the decentralized rewards
        print("- Plotting the decentralized rewards, and saving the plot to {} ...".format(savefig))
        evaluation.plotRewards(envId, savefig=savefig, semilogx=semilogx)

        # Plotting the centralized rewards
        savefig = savefig.replace('main', 'main_RewardsCentralized')
        print("- Plotting the centralized  rewards, and saving the plot to {} ...".format(savefig))
        evaluation.plotRegretsCentralized(envId, savefig=savefig, semilogx=semilogx)

        # # Also plotting the probability of picking the best arm
        # savefig = savefig.replace('main', 'main_BestArmPulls')
        # print(" - Plotting the probability of picking the best arm, and saving the plot to {} ...".format(savefig))
        # evaluation.plotBestArmPulls(envId, savefig=savefig)
        evaluation.plotBestArmPulls(envId)  # XXX To plot without saving

        # # Also plotting the probability of transmission on a free channel
        # savefig = savefig.replace('main', 'main_FreeTransmissions')
        # print(" - Plotting the probability of transmission on a free channel, and saving the plot to {} ...".format(savefig))
        # evaluation.plotFreeTransmissions(envId, savefig=savefig)
        # evaluation.plotFreeTransmissions(envId)  # XXX To plot without saving

        # Also plotting the frequency of collision in each arm
        savefig = savefig.replace('main', 'main_FrequencyCollisions')
        print(" - Plotting the frequency of collision in each arm, and saving the plot to {} ...".format(savefig))
        # evaluation.plotFrequencyCollisions(envId, savefig=savefig, piechart=piechart)
        evaluation.plotFrequencyCollisions(envId, piechart=piechart)

        if interactive:
            print(input("\n\nCan we continue to the next environment? [Enter]"))
    # Done
