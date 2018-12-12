#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them, for the multi-players case.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from Environment import EvaluatorMultiPlayers, notify

from example_of_configuration_multiplayers import configuration
configuration['showplot'] = True

N_players = len(configuration["successive_players"])

# List to keep all the EvaluatorMultiPlayers objects
evaluators = [[None] * N_players] * len(configuration["environment"])

if __name__ != '__main__':
    from sys import exit
    exit(0)

for playersId, players in enumerate(configuration["successive_players"]):
    print("\n\n\nConsidering the list of players :\n", players)  # DEBUG
    configuration['playersId'] = playersId
    configuration['players'] = players

    evaluation = EvaluatorMultiPlayers(configuration)

    # Start the evaluation and then print final ranking and plot, for each environment
    M = evaluation.nbPlayers
    N = len(evaluation.envs)

    for envId, env in enumerate(evaluation.envs):
        # Evaluate just that env
        evaluation.startOneEnv(envId, env)
        evaluators[envId][playersId] = evaluation

#
# Compare different MP strategies on the same figures
#
N = len(configuration["environment"])
for envId, env in enumerate(configuration["environment"]):

    e0, eothers = evaluators[envId][0], evaluators[envId][1:]
    M = e0.nbPlayers

    print("\nGiving all the vector of final regrets ...")
    e0.printLastRegrets(envId, evaluators=eothers)
    print("\nGiving the final ranking ...")
    e0.printFinalRankingAll(envId, evaluators=eothers)

    print("\nGiving the mean and std running times ...")
    e0.printRunningTimes(envId, evaluators=eothers)
    e0.plotRunningTimes(envId, evaluators=eothers)

    print("\n\n- Plotting the centralized regret for all 'players' values")
    e0.plotRegretCentralized(envId, normalized=False, evaluators=eothers)

    print("\n\n- Plotting the centralized regret for all 'players' values, in semilogx scale")
    e0.plotRegretCentralized(envId, semilogx=True, normalized=False, evaluators=eothers)

    print("\n\n- Plotting the centralized regret for all 'players' values, in semilogy scale")
    e0.plotRegretCentralized(envId, semilogy=True, normalized=False, evaluators=eothers)

    print("\n\n- Plotting the centralized regret for all 'players' values, in loglog scale")
    e0.plotRegretCentralized(envId, loglog=True, normalized=False, evaluators=eothers)

    print("\n\n- Plotting the centralized fairness (STD)")
    e0.plotFairness(envId, fairness='STD', evaluators=eothers)

    print("\n- Plotting the cumulated total nb of collision as a function of time for all 'players' values")
    e0.plotNbCollisions(envId, cumulated=True, evaluators=eothers)

    print("\n\n- Plotting the number of switches as a function of time for all 'players' values")
    e0.plotNbSwitchsCentralized(envId, cumulated=True, evaluators=eothers)

    print("\n- Plotting the histograms of regrets")
    e0.plotLastRegrets(envId, sharex=True, sharey=True, evaluators=eothers)
    # e0.plotLastRegrets(envId, all_on_separate_figures=True, evaluators=eothers)

# Done
print("Done for simulations example_of_main_multiplayers_more.py ...")
notify("Done for simulations example_of_main_multiplayers_more.py ...")
