#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them, for the multi-players case.
"""
from __future__ import print_function, division

__author__ = "Lilian Besson"
__version__ = "0.5"

# Generic imports
from os import mkdir
import os.path
import matplotlib.pyplot as plt

# Local imports
from Environment import EvaluatorMultiPlayers, notify
from configuration_multiplayers import configuration


# Parameters for the plots (where to save them) and what to draw
PLOT_DIR = "plots"
piechart = True
averageRegret = True
normalized = True
fairnessAmplitude = False
subTerms = True

saveallfigs = False
saveallfigs = True  # XXX dont keep it like this

# if not saveallfigs:
#     plt.xkcd()  # XXX turn on XKCD-like style ?! cf. http://matplotlib.org/xkcd/ for more details

# Whether to do the plots or not
do_simple_plot = False
do_simple_plot = True

do_all_plot = False
do_all_plot = True

# Whether to show plots, one by one, or not at all and just save them
interactive = True
interactive = False  # Seems to be the only mode which is working well

# Update configuration
configuration['showplot'] = interactive
del configuration['players']

_hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))


if __name__ == '__main__':
    if os.path.isdir(PLOT_DIR):
        print("{}/ is already a directory here...".format(PLOT_DIR))
    elif os.path.isfile(PLOT_DIR):
        raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(PLOT_DIR))
    else:
        mkdir(PLOT_DIR)

    N_players = len(configuration["successive_players"])

    # List to keep all the EvaluatorMultiPlayers objects
    evaluators = [[None] * N_players] * len(configuration["environment"])

    for playersId, players in enumerate(configuration["successive_players"]):
        print("\n\n\nConsidering the list of players :\n", players)  # DEBUG
        configuration['players'] = players

        # (almost) unique hash from the configuration
        hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))
        evaluation = EvaluatorMultiPlayers(configuration)
        # Start the evaluation and then print final ranking and plot, for each environment
        M = evaluation.nbPlayers
        N = len(evaluation.envs)
        for envId, env in enumerate(evaluation.envs):
            # Evaluate just that env
            evaluation.startOneEnv(envId, env)
            evaluators[envId][playersId] = evaluation
            # Display the final rankings for that env
            print("Giving the final ranks ...")
            evaluation.printFinalRanking(envId)

            # Sub folder with a useful name
            subfolder = "MP__M{}_T{}_N{}__{}_algos".format(M, configuration['horizon'], configuration['repetitions'], len(players))
            # Get the name of the output file
            imagename = "main____env{}-{}_{}.png".format((playersId + envId * N_players) + 1, N * N_players, hashvalue)
            # Create the sub folder
            plot_dir = os.path.join(PLOT_DIR, subfolder)
            if saveallfigs:
                if os.path.isdir(plot_dir):
                    print("{} is already a directory here...".format(plot_dir))
                elif os.path.isfile(plot_dir):
                    raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
                else:
                    mkdir(plot_dir)

            if not do_simple_plot:
                break

            mainfig = os.path.join(plot_dir, imagename)
            savefig = mainfig

            # Plotting the decentralized rewards
            print("\n\n- Plotting the decentralized rewards")
            if saveallfigs:
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotRewards(envId, savefig=savefig, semilogx=False)
            else:
                evaluation.plotRewards(envId, semilogx=False)  # XXX To plot without saving

            # Plotting the centralized fairness
            for fairnessAmplitude in [True, False]:
                savefig = mainfig.replace('main', 'main_Fairness%s' % ('Ampl' if fairnessAmplitude else 'Std'))
                print("\n\n- Plotting the centralized fairness")
                if saveallfigs:
                    print("  and saving the plot to {} ...".format(savefig))
                    evaluation.plotFairness(envId, savefig=savefig, semilogx=False, amplitude=fairnessAmplitude)
                else:
                    evaluation.plotFairness(envId, semilogx=False, amplitude=fairnessAmplitude)  # XXX To plot without saving

            # Plotting the centralized regret
            savefig = mainfig.replace('main', 'main_RegretCentralized')
            print("\n\n- Plotting the centralized regret")
            if saveallfigs:
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotRegretCentralized(envId, savefig=savefig, semilogx=False, normalized=False, subTerms=subTerms)
            else:
                evaluation.plotRegretCentralized(envId, semilogx=False, normalized=False, subTerms=subTerms)  # XXX To plot without saving

            # Plotting the centralized regret in semilogx
            savefig = mainfig.replace('main', 'main_RegretCentralized_semilogx')
            print("\n\n- Plotting the centralized regret")
            if saveallfigs:
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotRegretCentralized(envId, savefig=savefig, semilogx=True, normalized=False, subTerms=subTerms)
            else:
                evaluation.plotRegretCentralized(envId, semilogx=True, normalized=False, subTerms=subTerms)  # XXX To plot without saving

            # # Plotting the normalized centralized rewards
            # savefig = mainfig.replace('main', 'main_NormalizedRegretCentralized')
            # print("\n\n- Plotting the normalized centralized regret")
            # if saveallfigs:
            #     print("  and saving the plot to {} ...".format(savefig))
            #     evaluation.plotRegretCentralized(envId, savefig=savefig, semilogx=False, normalized=True, subTerms=subTerms)
            # else:
            #     evaluation.plotRegretCentralized(envId, semilogx=False, normalized=True, subTerms=subTerms)  # XXX To plot without saving

            # Plotting the number of switches
            savefig = mainfig.replace('main', 'main_NbSwitchs')
            print("\n\n- Plotting the number of switches")
            if saveallfigs:
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotNbSwitchs(envId, savefig=savefig, semilogx=False, cumulated=False)
            else:
                evaluation.plotNbSwitchs(envId, semilogx=False, cumulated=False)  # XXX To plot without saving

            # Plotting the cumulative number of switches
            savefig = mainfig.replace('main', 'main_CumNbSwitchs')
            print("\n\n- Plotting the cumulative number of switches")
            if saveallfigs:
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotNbSwitchs(envId, savefig=savefig, semilogx=False, cumulated=True)
            else:
                evaluation.plotNbSwitchs(envId, semilogx=False, cumulated=True)  # XXX To plot without saving

            # Also plotting the probability of picking the best arm
            savefig = mainfig.replace('main', 'main_BestArmPulls')
            print(" - Plotting the probability of picking the best arm")
            if saveallfigs:
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotBestArmPulls(envId, savefig=savefig)
            else:
                evaluation.plotBestArmPulls(envId)  # XXX To plot without saving

            # # Also plotting the probability of transmission on a free channel
            # savefig = mainfig.replace('main', 'main_FreeTransmissions')
            # print(" - Plotting the probability of transmission on a free channel")
            # if saveallfigs:
            #     print("  and saving the plot to {} ...".format(savefig))
            #     evaluation.plotFreeTransmissions(envId, savefig=savefig)
            # else:
            #     evaluation.plotFreeTransmissions(envId)  # XXX To plot without saving

            # # Also plotting the number of pulls of all arms
            # savefig = mainfig.replace('main', 'main_AllPulls')
            # print(" - Plotting the number of pulls of all arms")
            # if saveallfigs:
            #     print("  and saving the plot to {} ...".format(savefig))
            #     evaluation.plotAllPulls(envId, savefig=savefig, cumulated=False, normalized=False)
            # else:
            #     evaluation.plotAllPulls(envId, cumulated=False, normalized=False)  # XXX To plot without saving

            # # Also plotting the cumulative number of pulls of all arms
            # savefig = mainfig.replace('main', 'main_CumAllPulls')
            # print(" - Plotting the cumulative number of pulls of all arms")
            # if saveallfigs:
            #     print("  and saving the plot to {} ...".format(savefig))
            #     evaluation.plotAllPulls(envId, savefig=savefig, cumulated=True, normalized=False)
            # else:
            #     evaluation.plotAllPulls(envId, cumulated=True, normalized=False)  # XXX To plot without saving

            # # XXX Also plotting the cumulative number of pulls of all arms
            # savefig = mainfig.replace('main', 'main_NormalizedAllPulls')
            # print(" - Plotting the cumulative number of pulls of all arms")
            # if saveallfigs:
            #     print("  and saving the plot to {} ...".format(savefig))
            #     evaluation.plotAllPulls(envId, savefig=savefig, cumulated=True, normalized=True)
            # else:
            #     evaluation.plotAllPulls(envId, cumulated=True, normalized=True)  # XXX To plot without saving

            # Also plotting the total nb of collision as a function of time
            savefig = mainfig.replace('main', 'main_NbCollisions')
            print(" - Plotting the total nb of collision as a function of time")
            if saveallfigs:
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotNbCollisions(envId, savefig=savefig, cumulated=False)
            else:
                evaluation.plotNbCollisions(envId, cumulated=False)  # XXX To plot without saving

            # Also plotting the total nb of collision as a function of time
            savefig = mainfig.replace('main', 'main_CumNbCollisions')
            print(" - Plotting the cumulated total nb of collision as a function of time")
            if saveallfigs:
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotNbCollisions(envId, savefig=savefig, cumulated=True)
            else:
                evaluation.plotNbCollisions(envId, cumulated=True)  # XXX To plot without saving

            # Also plotting the frequency of collision in each arm
            savefig = mainfig.replace('main', 'main_FrequencyCollisions')
            print(" - Plotting the frequency of collision in each arm")
            if saveallfigs:
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotFrequencyCollisions(envId, savefig=savefig, piechart=piechart)
            else:
                evaluation.plotFrequencyCollisions(envId, piechart=piechart)  # XXX To plot without saving

            print("\n\n==> To see the figures, do :\neog", os.path.join(plot_dir, "main*{}.png".format(hashvalue)))  # DEBUG

    #
    # Compare different MP strategies on the same figures
    #
    N = len(configuration["environment"])
    for envId, env in enumerate(configuration["environment"]):
        if not do_all_plot:
            break

        e0, eothers = evaluators[envId][0], evaluators[envId][1:]
        M = e0.nbPlayers

        # Get the name of the output file
        imagename = "all____env{}-{}_{}.png".format(envId + 1, N, _hashvalue)
        mainfig = os.path.join(plot_dir, imagename)
        savefig = mainfig

        # Plotting the centralized regret
        savefig = mainfig.replace('all', 'all_RegretCentralized')
        print("\n\n- Plotting the centralized regret for all 'players' values")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            e0.plotRegretCentralized(envId, savefig=savefig, semilogx=False, normalized=False, evaluators=eothers)
        else:
            e0.plotRegretCentralized(envId, semilogx=False, normalized=False, evaluators=eothers)  # XXX To plot without saving

        # Plotting the centralized regret in semilogx
        savefig = mainfig.replace('all', 'all_RegretCentralized_semilogx')
        print("\n\n- Plotting the centralized regret for all 'players' values")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            e0.plotRegretCentralized(envId, savefig=savefig, semilogx=True, normalized=False, evaluators=eothers)
        else:
            e0.plotRegretCentralized(envId, semilogx=True, normalized=False, evaluators=eothers)  # XXX To plot without saving

        # Plotting the fairness
        for fairnessAmplitude in [True, False]:
            savefig = mainfig.replace('all', 'all_Fairness%s' % ('Ampl' if fairnessAmplitude else 'Std'))
            print("\n\n- Plotting the fairness for all 'players' values")
            if saveallfigs:
                print("  and saving the plot to {} ...".format(savefig))
                e0.plotFairness(envId, savefig=savefig, semilogx=False, amplitude=fairnessAmplitude, evaluators=eothers)
            else:
                e0.plotFairness(envId, semilogx=False, amplitude=fairnessAmplitude, evaluators=eothers)  # XXX To plot without saving

        # Also plotting the total nb of collision as a function of time
        savefig = mainfig.replace('all', 'all_NbCollisions')
        print(" - Plotting the total nb of collision as a function of time for all 'players' values")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            e0.plotNbCollisions(envId, savefig=savefig, cumulated=False, evaluators=eothers)
        else:
            e0.plotNbCollisions(envId, cumulated=False, evaluators=eothers)  # XXX To plot without saving

        # Also plotting the total nb of collision as a function of time
        savefig = mainfig.replace('all', 'all_CumNbCollisions')
        print(" - Plotting the cumulated total nb of collision as a function of time for all 'players' values")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            e0.plotNbCollisions(envId, savefig=savefig, cumulated=True, evaluators=eothers)
        else:
            e0.plotNbCollisions(envId, cumulated=True, evaluators=eothers)  # XXX To plot without saving

        # Plotting the number of switches as a function of time
        savefig = mainfig.replace('all', 'all_CumNbSwitchs')
        print("\n\n- Plotting the number of switches as a function of time for all 'players' values")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            e0.plotNbSwitchsCentralized(envId, savefig=savefig, semilogx=False, cumulated=True, evaluators=eothers)
        else:
            e0.plotNbSwitchsCentralized(envId, semilogx=False, cumulated=True, evaluators=eothers)  # XXX To plot without saving

        print("\n\n==> To see the figures, do :\neog", os.path.join(plot_dir, "all*{}.png".format(_hashvalue)))  # DEBUG

    # Done
    print("Done for simulations main_multiplayers.py ...")
    notify("Done for simulations main_multiplayers.py ...")
