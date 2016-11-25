#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them.
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

# Generic imports
from os import mkdir
import os.path
# Local imports
from Environment import Evaluator
from configuration import configuration

plot_dir = "plots"
semilogx = False

# Parameters for the Evaluator object
finalRanksOnAverage = True     # Use an average instead of the last value for the final ranking of the tested policies
averageOn = 5e-3               # Average the final rank on the 0.5% last time steps
useJoblibForPolicies = False

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
    # useJoblibForPolicies = True  # FIXME it does not work - yet
    evaluation = Evaluator(configuration,
                           finalRanksOnAverage=finalRanksOnAverage,
                           averageOn=averageOn,
                           useJoblibForPolicies=useJoblibForPolicies
                           )
    # evaluation.start_all_env()
    # Start the evaluation and then print final ranking and plot, for each environment
    N = len(evaluation.envs)
    for envId, env in enumerate(evaluation.envs):
        # (almost) unique hash from the configuration
        hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))
        # Evaluate just that env
        evaluation.start_one_env(envId, env)
        # Display the final rankings for that env
        print("Giving the final ranks ...")
        evaluation.giveFinalRanking(envId)
        if not do_plot:
            break
        # Sub folder with a useful name
        subfolder = "T{}_N{}__{}_algos".format(configuration['horizon'], configuration['repetitions'], len(configuration['policies']))
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
        print(" - Plotting the results, and saving the plot to {} ...".format(savefig))
        evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx)

        # Also plotting the probability of picking the best arm
        savefig = savefig.replace('main', 'main_BestArmPulls')
        print(" - Plotting the results, and saving the plot to {} ...".format(savefig))
        evaluation.plotBestArmPulls(envId, savefig=savefig)

        # input("\n\nCan we continue to the next environment? [Enter]")  # DEBUG
    # Done
