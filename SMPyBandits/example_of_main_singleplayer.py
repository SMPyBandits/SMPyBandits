#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An example of a simple 'main' script.
Main scripts load the config, run the simulations, and plot them, for the single player case.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import sys
if __name__ != '__main__':
    sys.exit(0)

from Environment import Evaluator, notify

if 'very_simple_configuration' in sys.argv or 'very_simple_configuration.py' in sys.argv:
    from very_simple_configuration import configuration
else:
    from example_of_configuration_singleplayer import configuration

configuration['showplot'] = True

evaluation = Evaluator(configuration)

# Start the evaluation and then print final ranking and plot, for each environment
for envId, env in enumerate(evaluation.envs):
    # Evaluate just that env
    evaluation.startOneEnv(envId, env)

# Compare them
for envId, env in enumerate(evaluation.envs):
    evaluation.plotHistoryOfMeans(envId)  # XXX To plot without saving

    print("\nGiving all the vector of final regrets ...")
    evaluation.printLastRegrets(envId)
    print("\nGiving the final ranking ...")
    evaluation.printFinalRanking(envId)

    print("\n\n- Plotting the last regrets...")
    evaluation.plotLastRegrets(envId, boxplot=True)

    print("\nGiving the mean and std running times ...")
    evaluation.printRunningTimes(envId)
    evaluation.plotRunningTimes(envId)

    print("\nGiving the mean and std running times ...")
    evaluation.printMemoryConsumption(envId)
    evaluation.plotMemoryConsumption(envId)

    print("\n\n- Plotting the mean reward...")
    evaluation.plotRegrets(envId, meanReward=True)

    print("\n\n- Plotting the regret...")
    evaluation.plotRegrets(envId)

    print("\n- Plotting the probability of picking the best arm of time...")
    evaluation.plotBestArmPulls(envId)

    print("\n- Plotting the histograms of regrets...")
    evaluation.plotLastRegrets(envId, sharex=True, sharey=True)

# Done
print("Done for simulations example_of_main_singleplayer ...")
notify("Done for simulations example_of_main_singleplayer ...")
