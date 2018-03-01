#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them, for the multi-players case.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.7"

# Generic imports
from os import mkdir
import os.path
from os import getenv
from itertools import product

# Backup evaluation object
import pickle
# import h5py

# Local imports
from Environment import EvaluatorMultiPlayers, notify
from configuration_multiplayers import configuration

# Solving https://github.com/SMPyBandits/SMPyBandits/issues/15#issuecomment-292484493
# For instance, call SLEEP=12h to delay the simulation for 12hours
if getenv('SLEEP', 'False') != 'False':
    from subprocess import call
    SLEEP = str(getenv('SLEEP'))
    print("\nSleeping for", SLEEP, "seconds before starting the simulation...")  # DEBUG
    call(["sleep", SLEEP])  # more general
    print("Done Sleeping for", SLEEP, "seconds... Now I can start the simulation...")

USE_PICKLE = False   #: Should we save the Evaluator object to a .pickle file at the end of the simulation?

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

#: Whether to do the plots or not
do_plots = True

#: Whether to show plots, one by one, or not at all and just save them
interactive = True  # XXX dont keep it like this
interactive = False

if getenv('DEBUG', 'False') == 'True' and __name__ == '__main__':
    print("====> TURNING DEBUG MODE ON <=====")
    saveallfigs, interactive = False, True

if getenv('SAVEALL', 'False') == 'True' and __name__ == '__main__':
    print("====> SAVING FIGURES <=====")
    saveallfigs = True

if getenv('XKCD', 'False') == 'True' and interactive and not saveallfigs:
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
        # # Plot histogram for rewards for that env
        # if do_plots and interactive:
        #     env.plotHistogram(evaluation.horizon * evaluation.repetitions)

        # Evaluate just that env
        evaluation.startOneEnv(envId, env)

        # Display the final rankings for that env
        print("\n\nGiving the final ranks ...")
        evaluation.printFinalRanking(envId)

        print("\n\nGiving the vector of final regrets ...")
        evaluation.printLastRegrets(envId)

        # Sub folder with a useful name
        subfolder = "MP__K{}_M{}_T{}_N{}".format(env.nbArms, len(configuration['players']), configuration['horizon'], configuration['repetitions'])
        # Get the name of the output file
        imagename = "main____env{}-{}_{}".format(envId + 1, N, hashvalue)
        plot_dir = os.path.join(PLOT_DIR, subfolder)

        mainfig = os.path.join(plot_dir, imagename)
        savefig = mainfig
        picklename = mainfig + '.pickle'

        # FIXME finish this to also save result in a HDF5 file!
        # h5pyname = mainfig + '.hdf5'
        # h5pyfile = h5py.File(h5pyname, 'w')

        if saveallfigs:
            # Create the sub folder
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

        if not do_plots:
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
            print("\n\n- Plotting the centralized fairness (%s)" % fairness)
            if saveallfigs:
                savefig = mainfig.replace('main', 'main_Fairness%s' % fairness)
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotFairness(envId, savefig=savefig, fairness=fairness)
            else:
                evaluation.plotFairness(envId, fairness=fairness)  # XXX To plot without saving

        # Plotting the centralized regret
        print("\n\n- Plotting the centralized regret")
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_RegretCentralized')
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRegretCentralized(envId, savefig=savefig, normalized=False, subTerms=subTerms)
        else:
            evaluation.plotRegretCentralized(envId, normalized=False, subTerms=subTerms)  # XXX To plot without saving

        # Plotting the centralized regret in semilogx
        print("\n\n- Plotting the centralized regret")
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_RegretCentralized_semilogx')
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRegretCentralized(envId, savefig=savefig, semilogx=True, normalized=False, subTerms=subTerms)
        else:
            evaluation.plotRegretCentralized(envId, semilogx=True, normalized=False, subTerms=subTerms)  # XXX To plot without saving

        # Plotting the centralized regret in semilogy
        print("\n\n- Plotting the centralized regret")
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_RegretCentralized_semilogy')
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRegretCentralized(envId, savefig=savefig, semilogy=True, normalized=False, subTerms=subTerms)
        else:
            evaluation.plotRegretCentralized(envId, semilogy=True, normalized=False, subTerms=subTerms)  # XXX To plot without saving

        # Plotting the centralized regret in loglog
        print("\n\n- Plotting the centralized regret")
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_RegretCentralized_loglog')
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotRegretCentralized(envId, savefig=savefig, loglog=True, normalized=False, subTerms=subTerms)
        else:
            evaluation.plotRegretCentralized(envId, loglog=True, normalized=False, subTerms=subTerms)  # XXX To plot without saving

        # # Plotting the normalized centralized rewards
        # print("\n\n- Plotting the normalized centralized regret")
        # if saveallfigs:
        #     savefig = mainfig.replace('main', 'main_NormalizedRegretCentralized')
        #     print("  and saving the plot to {} ...".format(savefig))
        #     evaluation.plotRegretCentralized(envId, savefig=savefig, normalized=True)
        # else:
        #     evaluation.plotRegretCentralized(envId, normalized=True)  # XXX To plot without saving

        # # Plotting the number of switches
        # print("\n\n- Plotting the number of switches")
        # if saveallfigs:
        #     savefig = mainfig.replace('main', 'main_NbSwitchs')
        #     print("  and saving the plot to {} ...".format(savefig))
        #     evaluation.plotNbSwitchs(envId, savefig=savefig, cumulated=False)
        # else:
        #     evaluation.plotNbSwitchs(envId, cumulated=False)  # XXX To plot without saving

        # Plotting the cumulative number of switches
        print("\n\n- Plotting the cumulative number of switches")
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_CumNbSwitchs')
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotNbSwitchs(envId, savefig=savefig, cumulated=True)
        else:
            evaluation.plotNbSwitchs(envId, cumulated=True)  # XXX To plot without saving

        # --- Also plotting the probability of picking the best arm
        print("\n- Plotting the probability of picking the best arm")
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_BestArmPulls')
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotBestArmPulls(envId, savefig=savefig)
        else:
            evaluation.plotBestArmPulls(envId)  # XXX To plot without saving

        # --- Also plotting the histograms of regrets
        print("\n- Plotting the histograms of regrets")
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_HistogramsRegret')
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotLastRegrets(envId, subplots=False, savefig=savefig)
        else:
            evaluation.plotLastRegrets(envId, subplots=False)  # XXX To plot without saving

        # # --- Also plotting the probability of transmission on a free channel
        # print("\n- Plotting the probability of transmission on a free channel")
        # if saveallfigs:
        #     savefig = mainfig.replace('main', 'main_FreeTransmissions')
        #     print("  and saving the plot to {} ...".format(savefig))
        #     evaluation.plotFreeTransmissions(envId, savefig=savefig)
        # else:
        #     evaluation.plotFreeTransmissions(envId)  # XXX To plot without saving

        # # --- Also plotting the number of pulls of all arms
        # print("\n- Plotting the number of pulls of all arms")
        # if saveallfigs:
        #     savefig = mainfig.replace('main', 'main_AllPulls')
        #     print("  and saving the plot to {} ...".format(savefig))
        #     evaluation.plotAllPulls(envId, savefig=savefig, cumulated=False, normalized=False)
        # else:
        #     evaluation.plotAllPulls(envId, cumulated=False, normalized=False)  # XXX To plot without saving

        # # --- Also plotting the cumulative number of pulls of all arms
        # print("\n- Plotting the cumulative number of pulls of all arms")
        # if saveallfigs:
        #     savefig = mainfig.replace('main', 'main_CumAllPulls')
        #     print("  and saving the plot to {} ...".format(savefig))
        #     evaluation.plotAllPulls(envId, savefig=savefig, cumulated=True, normalized=False)
        # else:
        #     evaluation.plotAllPulls(envId, cumulated=True, normalized=False)  # XXX To plot without saving

        # # XXX Also plotting the cumulative number of pulls of all arms
        # print("\n- Plotting the cumulative number of pulls of all arms")
        # if saveallfigs:
        #     savefig = mainfig.replace('main', 'main_NormalizedAllPulls')
        #     print("  and saving the plot to {} ...".format(savefig))
        #     evaluation.plotAllPulls(envId, savefig=savefig, cumulated=True, normalized=True)
        # else:
        #     evaluation.plotAllPulls(envId, cumulated=True, normalized=True)  # XXX To plot without saving

        # # --- Also plotting the total nb of collision as a function of time
        # print("\n- Plotting the total nb of collision as a function of time")
        # if saveallfigs:
        #     savefig = mainfig.replace('main', 'main_NbCollisions')
        #     print("  and saving the plot to {} ...".format(savefig))
        #     evaluation.plotNbCollisions(envId, savefig=savefig, cumulated=False)
        # else:
        #     evaluation.plotNbCollisions(envId, cumulated=False)  # XXX To plot without saving

        # --- Also plotting the total nb of collision as a function of time
        print("\n- Plotting the cumulated total nb of collision as a function of time")
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_CumNbCollisions')
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotNbCollisions(envId, savefig=savefig, cumulated=True, upperbound=False)
        else:
            evaluation.plotNbCollisions(envId, cumulated=True, upperbound=False)  # XXX To plot without saving

        # --- Also plotting the frequency of collision in each arm
        for piechart in [False, True]:
            print("\n- Plotting the frequency of collision in each arm")
            if saveallfigs:
                savefig = mainfig.replace('main', 'main_FrequencyCollisions%s' % ('' if piechart else 'Hist'))
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotFrequencyCollisions(envId, savefig=savefig, piechart=piechart)
            else:
                evaluation.plotFrequencyCollisions(envId, piechart=piechart)  # XXX To plot without saving

        if saveallfigs:
            print("\n\n==> To see the figures, do :\neog", os.path.join(plot_dir, "main*{}.png".format(hashvalue)))  # DEBUG
    # Done
    print("Done for simulations main_multiplayers.py ...")
    notify("Done for simulations main_multiplayers.py ...")
