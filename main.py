#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them.
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

from os import mkdir
import os.path
from Environment import Evaluator
from configuration import configuration

plot_dir = "plots"
semilogx = False

# Parameters for the Evaluator object
finalRanksOnAverage = True     # Use an average instead of the last value for the final ranking of the tested policies
averageOn = 5e-3               # Average the final rank on the 0.5% last time steps
useJoblibForPolicies = False


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
    # Start the evaluation
    evaluation.start()

    # Print final ranking and plot, for each environment
    N = len(evaluation.envs)
    for i in range(N):
        evaluation.giveFinalRanking(i)
        # XXX be more explicit here
        hashvalue = abs(hash((tuple(configuration.keys()), configuration.values())))  # (almost) unique hash from the configuration
        imagename = "main__{}_{}-{}.png".format(hashvalue, i + 1, N)
        evaluation.plotResults(i, savefig=os.path.join(plot_dir, imagename), semilogx=semilogx)
        # evaluation.plotResults(i, semilogx=not semilogx)
