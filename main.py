#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.2"

# Generic imports
from os import mkdir
import os.path
import matplotlib.pyplot as plt
# Local imports
from Environment import Evaluator
from configuration import configuration


# Parameters for the plots (where to save them) and what to draw
plot_dir = "plots"
semilogx = False
meanRegret = True
normalizedRegret = True
plotSTD = False

saveallfigs = True  # XXX dont keep it like this when experimenting
saveallfigs = False

# Parameters for the Evaluator object
finalRanksOnAverage = True     # Use an average instead of the last value for the final ranking of the tested policies
averageOn = 5e-3               # Average the final rank on the 0.5% last time steps

# Whether to do the plots or not
do_plot = False
do_plot = True

# Whether to show all plots, or one by one
interactive = True
interactive = False  # Seems to be the only mode which is working well


if __name__ == '__main__':
    if os.path.isdir(plot_dir):
        print("{}/ is already a directory here...".format(plot_dir))
    elif os.path.isfile(plot_dir):
        raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
    else:
        mkdir(plot_dir)
    evaluation = Evaluator(configuration,
                           finalRanksOnAverage=finalRanksOnAverage,
                           averageOn=averageOn
                           )
    # Start the evaluation and then print final ranking and plot, for each environment
    N = len(evaluation.envs)
    for envId, env in enumerate(evaluation.envs):
        # (almost) unique hash from the configuration
        hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))
        # Evaluate just that env
        evaluation.startOneEnv(envId, env)
        # Display the final rankings for that env
        print("Giving the final ranks ...")
        evaluation.printFinalRanking(envId)
        if not do_plot:
            break

        # Sub folder with a useful name
        subfolder = "T{}_N{}__{}_algos".format(configuration['horizon'], configuration['repetitions'], len(configuration['policies']))
        # Get the name of the output file
        imagename = "main____env{}-{}_{}.png".format(envId + 1, N, hashvalue)
        if saveallfigs:
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

        mainfig = os.path.join(plot_dir, imagename)
        savefig = mainfig

        if saveallfigs:
            print(" - Plotting the cumulative rewards, and saving the plot to {} ...".format(savefig))
            evaluation.plotRegrets(envId, savefig=savefig, semilogx=False)  # XXX To save the figure
            savefig = mainfig.replace('main', 'main_semilogx')
            evaluation.plotRegrets(envId, savefig=savefig, semilogx=True)  # XXX To save the figure
        else:
            evaluation.plotRegrets(envId, semilogx=False, plotSTD=False)
            evaluation.plotRegrets(envId, semilogx=True, plotSTD=False)
            # if configuration['repetitions'] > 1: evaluation.plotRegrets(envId, semilogx=semilogx, plotSTD=True)

        if meanRegret:
            if saveallfigs:
                savefig = mainfig.replace('main', 'main_MeanRewards')
                print(" - Plotting the mean rewards, and saving the plot to {} ...".format(savefig))
                evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx, meanRegret=True)  # XXX To save the figure
            else:
                evaluation.plotRegrets(envId, semilogx=semilogx, meanRegret=True, plotSTD=False)
                # if configuration['repetitions'] > 1: evaluation.plotRegrets(envId, semilogx=semilogx, meanRegret=True, plotSTD=True)

        if normalizedRegret:
            if saveallfigs:
                savefig = mainfig.replace('main', 'main_Normalized')
                print(" - Plotting the mean rewards, and saving the plot to {} ...".format(savefig))
                evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx, normalizedRegret=True)  # XXX To save the figure
            else:
                evaluation.plotRegrets(envId, semilogx=semilogx, normalizedRegret=True, plotSTD=False)
                # if configuration['repetitions'] > 1: evaluation.plotRegrets(envId, semilogx=semilogx, normalizedRegret=True, plotSTD=True)

        # --- Also plotting the probability of picking the best arm
        if evaluation.random_shuffle or evaluation.random_invert:
            print(" - Not plotting probability of picking the best arm as we used random events ...")
            print("   ==> FIXME correct this bug")
        else:
            if saveallfigs:
                savefig = mainfig.replace('main', 'main_BestArmPulls')
                print(" - Plotting the results, and saving the plot to {} ...".format(savefig))
                evaluation.plotBestArmPulls(envId, savefig=savefig)  # XXX To save the figure
            else:
                evaluation.plotBestArmPulls(envId)

        if interactive:
            print(input("\n\nCan we continue to the next environment? [Enter]"))
    # Done
