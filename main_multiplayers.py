#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them, for the multi-players case.
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

# Generic imports
from os import mkdir
import os.path
# Local imports
from Environment import EvaluatorMultiPlayers
from configuration_multiplayers import configuration

plot_dir = "plots"
semilogx = False

# Whether to do the plots or not
do_plot = False
do_plot = True


if __name__ == '__main__':
    if os.path.isdir(plot_dir):
        print("{} is already a directory here...".format(plot_dir))
    elif os.path.isfile(plot_dir):
        raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
    else:
        mkdir(plot_dir)
    evaluation = EvaluatorMultiPlayers(configuration)
    # Start the evaluation and then print final ranking and plot, for each environment
    M = evaluation.nbPlayers
    N = len(evaluation.envs)
    for envId, env in enumerate(evaluation.envs):
        # (almost) unique hash from the configuration
        hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))
        # Evaluate just that env
        evaluation.start_one_env(envId, env)
        # Display the final rankings for that env
        print("Giving the final ranks ...")
        evaluation.giveFinalRanking(envId)
        if do_plot:
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
            savefig = os.path.join(plot_dir, imagename)
            print("Plotting the results, and saving the plot to {} ...".format(savefig))
            # evaluation.plotRewards(envId, semilogx=not semilogx)
            evaluation.plotRewards(envId, savefig=savefig, semilogx=semilogx)
            # Also plotting the probability of picking the best arm
            savefig = savefig.replace('main', 'main_BestArmPulls')
            print(" - Plotting the results, and saving the plot to {} ...".format(savefig))
            evaluation.plotBestArmPulls(envId, savefig=savefig)
    # Done
