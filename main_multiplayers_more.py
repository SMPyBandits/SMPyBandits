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

# Backup evaluation object
import pickle
# import h5py

# Local imports
from Environment import EvaluatorMultiPlayers, notify
from configuration_multiplayers import configuration

# Solving https://github.com/Naereen/AlgoBandits/issues/15#issuecomment-292484493
# For instance, call SLEEP=12h to delay the simulation for 12hours
if getenv('SLEEP', False):
    from subprocess import call
    SLEEP = str(getenv('SLEEP'))
    print("\nSleeping for", SLEEP, "seconds before starting the simulation...")  # DEBUG
    call(["sleep", SLEEP])  # more general
    print("Done Sleeping for", SLEEP, "seconds ... Now I can start the simulation...")

USE_PICKLE = True   #: Should we save the Evaluator object to a .pickle file at the end of the simulation?

# Parameters for the plots (where to save them) and what to draw
PLOT_DIR = "plots"  #: Directory for the plots
piechart = True  #: Plot a piechart for collision counts? Otherwise, plot an histogram.
piechart = False  #: Plot a piechart for collision counts? Otherwise, plot an histogram.
averageRegret = True  #: Use average regret ?
normalized = True  #: Plot normalized regret?
fairnessAmplitude = False  #: Use amplitude measure for the fairness or std?
subTerms = True  #: Plot the 3 sub terms for the regret

saveallfigs = False  #: Save all the figures ?
saveallfigs = True  # XXX dont keep it like this

#: Whether to do the plots for single experiments or not
do_simple_plots = True
do_simple_plots = False

#: Whether to do the plots for comparison experiments or not
do_comparison_plots = False
do_comparison_plots = True

#: Whether to show plots, one by one, or not at all and just save them
interactive = True  # XXX dont keep it like this
interactive = False

if str(getenv('DEBUG', False)) == 'True' and __name__ == '__main__':
    print("====> TURNING DEBUG MODE ON <=====")
    saveallfigs, interactive = False, True

if str(getenv('SAVEALL', False)) == 'True' and __name__ == '__main__':
    print("====> SAVING FIGURES <=====")
    saveallfigs = True

if str(getenv('XKCD', False)) == 'True' and interactive and not saveallfigs:
    import matplotlib.pyplot as plt
    plt.xkcd()  # XXX turn on XKCD-like style ?! cf. http://matplotlib.org/xkcd/ for more details


if __name__ == '__main__':
    # Update configuration
    configuration['showplot'] = interactive
    del configuration['players']  # Be sure to only use "successive_players" value

    _hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))

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
            # # Plot histogram for rewards for that env
            # if do_simple_plots and interactive:
            #     env.plotHistogram(evaluation.horizon * evaluation.repetitions)

            # Evaluate just that env
            evaluation.startOneEnv(envId, env)
            if do_comparison_plots:
                evaluators[envId][playersId] = evaluation

            # Display the final rankings for that env
            print("Giving the final ranks ...")
            evaluation.printFinalRanking(envId)

            # Sub folder with a useful name
            subfolder = "MP__M{}_T{}_N{}__{}_algos".format(M, configuration['horizon'], configuration['repetitions'], len(players))
            # Get the name of the output file
            imagename = "main____env{}-{}_{}".format((playersId + envId * N_players) + 1, N * N_players, hashvalue)
            # Create the sub folder
            plot_dir = os.path.join(PLOT_DIR, subfolder)

            mainfig = os.path.join(plot_dir, imagename)
            savefig = mainfig
            picklename = mainfig + '.pickle'
            # FIXME finish this to also save result in a HDF5 file!
            # h5pyname = mainfig + '.hdf5'
            # h5pyfile = h5py.File(h5pyname, 'w')

            if saveallfigs:
                if os.path.isdir(plot_dir):
                    print("{} is already a directory here...".format(plot_dir))
                elif os.path.isfile(plot_dir):
                    raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
                else:
                    mkdir(plot_dir)

                # Save it to a pickle file
                # TODO use numpy.savez_compressed instead ? https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed
                if USE_PICKLE:
                    with open(picklename, 'wb') as picklefile:
                        print("Saving the EvaluatorMultiPlayers 'evaluation' objet to", picklename, "...")
                        pickle.dump(evaluation, picklefile, pickle.HIGHEST_PROTOCOL)

            if not do_simple_plots:
                break

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

            # # Plotting the centralized regret in loglog
            # savefig = mainfig.replace('main', 'main_RegretCentralized_loglog')
            # print("\n\n- Plotting the centralized regret")
            # if saveallfigs:
            #     print("  and saving the plot to {} ...".format(savefig))
            #     evaluation.plotRegretCentralized(envId, savefig=savefig, loglog=True, normalized=False, subTerms=subTerms)
            # else:
            #     evaluation.plotRegretCentralized(envId, loglog=True, normalized=False, subTerms=subTerms)  # XXX To plot without saving

            # # Plotting the normalized centralized rewards
            # savefig = mainfig.replace('main', 'main_NormalizedRegretCentralized')
            # print("\n\n- Plotting the normalized centralized regret")
            # if saveallfigs:
            #     print("  and saving the plot to {} ...".format(savefig))
            #     evaluation.plotRegretCentralized(envId, savefig=savefig, normalized=True, subTerms=subTerms)
            # else:
            #     evaluation.plotRegretCentralized(envId, normalized=True, subTerms=subTerms)  # XXX To plot without saving

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
            for piechart in [False, True]:
                savefig = mainfig.replace('main', 'main_FrequencyCollisions%s' % ('' if piechart else 'Hist'))
                print(" - Plotting the frequency of collision in each arm")
                if saveallfigs:
                    print("  and saving the plot to {} ...".format(savefig))
                    evaluation.plotFrequencyCollisions(envId, savefig=savefig, piechart=piechart)
                else:
                    evaluation.plotFrequencyCollisions(envId, piechart=piechart)  # XXX To plot without saving

            if saveallfigs:
                print("\n\n==> To see the figures, do :\neog", os.path.join(plot_dir, "main*{}.png".format(hashvalue)))  # DEBUG

    #
    # Compare different MP strategies on the same figures
    #
    N = len(configuration["environment"])
    for envId, env in enumerate(configuration["environment"]):
        if not do_comparison_plots:
            break

        e0, eothers = evaluators[envId][0], evaluators[envId][1:]
        M = e0.nbPlayers

        # Get the name of the output file
        imagename = "all____env{}-{}_{}".format(envId + 1, N, _hashvalue)
        mainfig = os.path.join(plot_dir, imagename)
        savefig = mainfig

        # Plotting the centralized regret
        savefig = mainfig.replace('all', 'all_RegretCentralized')
        print("\n\n- Plotting the centralized regret for all 'players' values")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            e0.plotRegretCentralized(envId, savefig=savefig, normalized=False, evaluators=eothers)
        else:
            e0.plotRegretCentralized(envId, normalized=False, evaluators=eothers)  # XXX To plot without saving

        # Plotting the centralized regret in semilogx
        savefig = mainfig.replace('all', 'all_RegretCentralized_semilogx')
        print("\n\n- Plotting the centralized regret for all 'players' values, in semilogx scale")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            e0.plotRegretCentralized(envId, savefig=savefig, semilogx=True, normalized=False, evaluators=eothers)
        else:
            e0.plotRegretCentralized(envId, semilogx=True, normalized=False, evaluators=eothers)  # XXX To plot without saving

        # Plotting the centralized regret in semilogy
        savefig = mainfig.replace('all', 'all_RegretCentralized_semilogy')
        print("\n\n- Plotting the centralized regret for all 'players' values, in semilogy scale")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            e0.plotRegretCentralized(envId, savefig=savefig, semilogy=True, normalized=False, evaluators=eothers)
        else:
            e0.plotRegretCentralized(envId, semilogy=True, normalized=False, evaluators=eothers)  # XXX To plot without saving

        # Plotting the centralized regret in loglog
        savefig = mainfig.replace('all', 'all_RegretCentralized_loglog')
        print("\n\n- Plotting the centralized regret for all 'players' values, in loglog scale")
        if saveallfigs:
            print("  and saving the plot to {} ...".format(savefig))
            e0.plotRegretCentralized(envId, savefig=savefig, loglog=True, normalized=False, evaluators=eothers)
        else:
            e0.plotRegretCentralized(envId, loglog=True, normalized=False, evaluators=eothers)  # XXX To plot without saving

        # Plotting the fairness
        for fairness in ['STD'] if savefig else ['Ampl', 'STD', 'RajJain', 'Mean']:
            savefig = mainfig.replace('main', 'main_Fairness%s' % fairness)
            print("\n\n- Plotting the centralized fairness (%s)" % fairness)
            if saveallfigs:
                print("  and saving the plot to {} ...".format(savefig))
                e0.plotFairness(envId, savefig=savefig, fairness=fairness, evaluators=eothers)
            else:
                e0.plotFairness(envId, fairness=fairness, evaluators=eothers)  # XXX To plot without saving

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
            e0.plotNbSwitchsCentralized(envId, savefig=savefig, cumulated=True, evaluators=eothers)
        else:
            e0.plotNbSwitchsCentralized(envId, cumulated=True, evaluators=eothers)  # XXX To plot without saving

        if saveallfigs:
            print("\n\n==> To see the figures, do :\neog", os.path.join(plot_dir, "all*{}.png".format(_hashvalue)))  # DEBUG

    # Done
    print("Done for simulations main_multiplayers.py ...")
    notify("Done for simulations main_multiplayers.py ...")
