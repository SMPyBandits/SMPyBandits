#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them, for the multi-players case.
"""
from __future__ import print_function, division

__author__ = "Lilian Besson"
__version__ = "0.6"

# Generic imports
from os import mkdir
import os.path
from os import getenv

# if __name__ != '__main__':
#     print("Warning: this script 'main.py' does not expose any documentation ...")  # DEBUG
#     exit(0)

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

# Whether to do the plots or not
do_plot = False
do_plot = True

# Whether to show plots, one by one, or not at all and just save them
interactive = True  # XXX dont keep it like this
interactive = False

if getenv('DEBUG', False) and __name__ == '__main__':
    saveallfigs, interactive = False, True

if interactive and not saveallfigs:
    import matplotlib.pyplot as plt
    plt.xkcd()  # XXX turn on XKCD-like style ?! cf. http://matplotlib.org/xkcd/ for more details


if __name__ == '__main__':
    # Update configuration
    configuration['showplot'] = interactive

    if os.path.isdir(PLOT_DIR):
        print("{}/ is already a directory here...".format(PLOT_DIR))
    elif os.path.isfile(PLOT_DIR):
        raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(PLOT_DIR))
    else:
        mkdir(PLOT_DIR)
    # (almost) unique hash from the configuration
    hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))
    evaluation = EvaluatorMultiPlayers(configuration)
    # Start the evaluation and then print final ranking and plot, for each environment
    M = evaluation.nbPlayers
    N = len(evaluation.envs)
    for envId, env in enumerate(evaluation.envs):
        # Plot histogram for rewards for that env
        if do_plot and interactive:
            env.plotHistogram(evaluation.horizon * evaluation.repetitions)

        # Evaluate just that env
        evaluation.startOneEnv(envId, env)
        # Display the final rankings for that env
        print("Giving the final ranks ...")
        evaluation.printFinalRanking(envId)
        if not do_plot:
            break

        # Sub folder with a useful name
        subfolder = "MP__M{}_T{}_N{}".format(len(configuration['players']), configuration['horizon'], configuration['repetitions'])
        # Get the name of the output file
        imagename = "main____env{}-{}_{}".format(envId + 1, N, hashvalue)
        plot_dir = os.path.join(PLOT_DIR, subfolder)
        if saveallfigs:
            # Create the sub folder
            if os.path.isdir(plot_dir):
                print("{} is already a directory here...".format(plot_dir))
            elif os.path.isfile(plot_dir):
                raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
            else:
                mkdir(plot_dir)

        mainfig = os.path.join(plot_dir, imagename)
        savefig = mainfig
        # Plotting the decentralized rewards
        print("\n\n- Plotting the decentralized rewards")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRewards(envId, savefig=savefig)
        else:
            evaluation.plotRewards(envId)  # XXX To plot without saving

        # Plotting the centralized fairness
        for fairness in ['STD'] if savefig else ['Ampl', 'STD', 'RajJain', 'Mean']:
            savefig = mainfig.replace('main', 'main_Fairness%s' % fairness)
            print("\n\n- Plotting the centralized fairness (%s)" % fairness)
            if saveallfigs:
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotFairness(envId, savefig=savefig, fairness=fairness)
            else:
                evaluation.plotFairness(envId, fairness=fairness)  # XXX To plot without saving

        # Plotting the centralized regret
        savefig = mainfig.replace('main', 'main_RegretCentralized')
        print("\n\n- Plotting the centralized regret")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRegretCentralized(envId, savefig=savefig, normalized=False, subTerms=subTerms)
        else:
            evaluation.plotRegretCentralized(envId, normalized=False, subTerms=subTerms)  # XXX To plot without saving

        # Plotting the centralized regret in semilogx
        savefig = mainfig.replace('main', 'main_RegretCentralized_semilogx')
        print("\n\n- Plotting the centralized regret")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRegretCentralized(envId, savefig=savefig, semilogx=True, normalized=False, subTerms=subTerms)
        else:
            evaluation.plotRegretCentralized(envId, semilogx=True, normalized=False, subTerms=subTerms)  # XXX To plot without saving

        # Plotting the centralized regret in semilogy
        savefig = mainfig.replace('main', 'main_RegretCentralized_semilogy')
        print("\n\n- Plotting the centralized regret")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRegretCentralized(envId, savefig=savefig, semilogy=True, normalized=False, subTerms=subTerms)
        else:
            evaluation.plotRegretCentralized(envId, semilogy=True, normalized=False, subTerms=subTerms)  # XXX To plot without saving

        # Plotting the centralized regret in loglog
        savefig = mainfig.replace('main', 'main_RegretCentralized_loglog')
        print("\n\n- Plotting the centralized regret")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRegretCentralized(envId, savefig=savefig, loglog=True, normalized=False, subTerms=subTerms)
        else:
            evaluation.plotRegretCentralized(envId, loglog=True, normalized=False, subTerms=subTerms)  # XXX To plot without saving

        # # Plotting the normalized centralized rewards
        # savefig = mainfig.replace('main', 'main_NormalizedRegretCentralized')
        # print("\n\n- Plotting the normalized centralized regret")
        # if saveallfigs:
        #     print("  and saving the plot to {} ...".format(savefig))
        #     evaluation.plotRegretCentralized(envId, savefig=savefig, normalized=True)
        # else:
        #     evaluation.plotRegretCentralized(envId, normalized=True)  # XXX To plot without saving

        # # Plotting the number of switches
        # savefig = mainfig.replace('main', 'main_NbSwitchs')
        # print("\n\n- Plotting the number of switches")
        # if saveallfigs:
        #     print("  and saving the plot to {} ...".format(savefig))
        #     evaluation.plotNbSwitchs(envId, savefig=savefig, cumulated=False)
        # else:
        #     evaluation.plotNbSwitchs(envId, cumulated=False)  # XXX To plot without saving

        # Plotting the cumulative number of switches
        savefig = mainfig.replace('main', 'main_CumNbSwitchs')
        print("\n\n- Plotting the cumulative number of switches")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotNbSwitchs(envId, savefig=savefig, cumulated=True)
        else:
            evaluation.plotNbSwitchs(envId, cumulated=True)  # XXX To plot without saving

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

        # # Also plotting the total nb of collision as a function of time
        # savefig = mainfig.replace('main', 'main_NbCollisions')
        # print(" - Plotting the total nb of collision as a function of time")
        # if saveallfigs:
        #     print("  and saving the plot to {} ...".format(savefig))
        #     evaluation.plotNbCollisions(envId, savefig=savefig, cumulated=False)
        # else:
        #     evaluation.plotNbCollisions(envId, cumulated=False)  # XXX To plot without saving

        # Also plotting the total nb of collision as a function of time
        savefig = mainfig.replace('main', 'main_CumNbCollisions')
        print(" - Plotting the cumulated total nb of collision as a function of time")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotNbCollisions(envId, savefig=savefig, cumulated=True, upperbound=False)
        else:
            evaluation.plotNbCollisions(envId, cumulated=True, upperbound=False)  # XXX To plot without saving

        # Also plotting the frequency of collision in each arm
        savefig = mainfig.replace('main', 'main_FrequencyCollisions')
        print(" - Plotting the frequency of collision in each arm")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotFrequencyCollisions(envId, savefig=savefig, piechart=piechart)
        else:
            evaluation.plotFrequencyCollisions(envId, piechart=piechart)  # XXX To plot without saving

        if saveallfigs:
            print("\n\n==> To see the figures, do :\neog", os.path.join(plot_dir, "main*{}.png".format(hashvalue)))  # DEBUG
    # Done
    print("Done for simulations main_multiplayers.py ...")
    notify("Done for simulations main_multiplayers.py ...")
