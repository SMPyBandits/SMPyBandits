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
plot_dir = "plots"
piechart = True
averageRegret = True
normalized = True

saveallfigs = False
saveallfigs = True  # XXX dont keep it like this

# if not saveallfigs:
#     plt.xkcd()  # XXX turn on XKCD-like style ?! cf. http://matplotlib.org/xkcd/ for more details

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
    # (almost) unique hash from the configuration
    hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))
    evaluation = EvaluatorMultiPlayers(configuration)
    # Start the evaluation and then print final ranking and plot, for each environment
    M = evaluation.nbPlayers
    N = len(evaluation.envs)
    for envId, env in enumerate(evaluation.envs):
        # Evaluate just that env
        evaluation.startOneEnv(envId, env)
        # Display the final rankings for that env
        print("Giving the final ranks ...")
        evaluation.printFinalRanking(envId)
        if not do_plot:
            break

        # Sub folder with a useful name
        subfolder = "MP__M{}_T{}_N{}__{}_algos".format(M, configuration['horizon'], configuration['repetitions'], len(configuration['players']))
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
        # Plotting the decentralized rewards
        print("\n\n- Plotting the decentralized rewards")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRewards(envId, savefig=savefig, semilogx=False)
        else:
            evaluation.plotRewards(envId, semilogx=False)  # XXX To plot without saving

        # Plotting the centralized regret
        savefig = mainfig.replace('main', 'main_RegretCentralized')
        print("\n\n- Plotting the centralized regret")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRegretCentralized(envId, savefig=savefig, semilogx=False, normalized=False)
        else:
            evaluation.plotRegretCentralized(envId, semilogx=False, normalized=False)  # XXX To plot without saving

        # Plotting the centralized regret in semilogx
        savefig = mainfig.replace('main', 'main_RegretCentralized')
        print("\n\n- Plotting the centralized regret")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRegretCentralized(envId, savefig=savefig, semilogx=True, normalized=False)
        else:
            evaluation.plotRegretCentralized(envId, semilogx=True, normalized=False)  # XXX To plot without saving

        # Plotting the normalized centralized rewards
        savefig = mainfig.replace('main', 'main_NormalizedRewardsCentralized')
        print("\n\n- Plotting the normalized centralized rewards")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRegretCentralized(envId, savefig=savefig, semilogx=False, normalized=True)
        else:
            evaluation.plotRegretCentralized(envId, semilogx=False, normalized=True)  # XXX To plot without saving

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

        # if not interactive:
        #     plt.interactive(True)
        # Also plotting the frequency of collision in each arm
        savefig = mainfig.replace('main', 'main_FrequencyCollisions')
        print(" - Plotting the frequency of collision in each arm")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotFrequencyCollisions(envId, savefig=savefig, piechart=piechart)
        else:
            evaluation.plotFrequencyCollisions(envId, piechart=piechart)  # XXX To plot without saving

        if interactive:
            print(input("\n\nCan we continue to the next environment? [Enter]"))
    # Done
    print("Done for simulations main_multiplayers.py ...")
    notify("Done for simulations main_multiplayers.py ...")
