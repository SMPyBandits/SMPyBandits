#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

# Generic imports
import sys
from os import mkdir
import os.path
from os import getenv
from itertools import product

# Backup evaluation object
import pickle

# Local imports
try:
    from Environment import Evaluator, notify, start_tracemalloc, display_top_tracemalloc
    # Import a configuration file
    if 'configuration_comparing_aggregation_algorithms' in sys.argv:
        from configuration_comparing_aggregation_algorithms import configuration
    elif 'configuration_all_singleplayer' in sys.argv:
        from configuration_all_singleplayer import configuration
    elif 'configuration_comparing_doubling_algorithms' in sys.argv:
        from configuration_comparing_doubling_algorithms import configuration
    elif 'configuration_markovian' in sys.argv:
        from configuration_markovian import configuration
    elif 'configuration_sparse' in sys.argv:
        from configuration_sparse import configuration
    else:
        from configuration import configuration
except ImportError:
    from SMPyBandits.Environment import Evaluator, notify
    # Import a configuration file
    if 'configuration_comparing_aggregation_algorithms' in sys.argv:
        from SMPyBandits.configuration_comparing_aggregation_algorithms import configuration
    elif 'configuration_all_singleplayer' in sys.argv:
        from SMPyBandits.configuration_all_singleplayer import configuration
    elif 'configuration_comparing_doubling_algorithms' in sys.argv:
        from SMPyBandits.configuration_comparing_doubling_algorithms import configuration
    elif 'configuration_markovian' in sys.argv:
        from SMPyBandits.configuration_markovian import configuration
    elif 'configuration_sparse' in sys.argv:
        from SMPyBandits.configuration_sparse import configuration
    else:
        from SMPyBandits.configuration import configuration

# Solving https://github.com/SMPyBandits/SMPyBandits/issues/15#issuecomment-292484493
# For instance, call SLEEP=12h to delay the simulation for 12hours
if getenv('SLEEP', 'False') != 'False':
    from subprocess import call
    SLEEP = str(getenv('SLEEP'))
    print("\nSleeping for", SLEEP, "seconds before starting the simulation...")  # DEBUG
    call(["sleep", SLEEP])  # more general
    print("Done Sleeping for", SLEEP, "seconds... Now I can start the simulation...")

USE_PICKLE = False   #: Should we save the Evaluator object to a .pickle file at the end of the simulation?
USE_HD5 = True   #: Should we save the data to a .hdf5 file at the end of the simulation?

# Parameters for the plots (where to save them) and what to draw
PLOT_DIR = getenv('PLOT_DIR', 'plots')  #: Directory for the plots
semilogx = False  #: Plot in semilogx by default?
semilogy = False  #: Plot in semilogy by default?
loglog   = False  #: Plot in loglog   by default?
meanRegret = True  #: Plot mean regret ?
normalizedRegret = True  #: Plot instantaneous regret?

plotSTD = True   #: Plot regret with a STD?
plotSTD = False  #: Plot regret with a STD?

plotMaxMin = True   #: Plot +- max - min (amplitude) for regret.
plotMaxMin = False  #: Plot +- max - min (amplitude) for regret.

saveallfigs = False  #: Save all the figures ?
saveallfigs = True  # XXX dont keep it like this when experimenting

# Parameters for the Evaluator object
finalRanksOnAverage = True     #: Use an average instead of the last value for the final ranking of the tested policies
averageOn = 1e-2               #: Average the final rank on the 1% last time steps

#: Whether to do the plots or not
do_plots = True

if getenv('NOPLOTS', 'False') == 'True' and __name__ == '__main__':
    print("====> TURNING NOPLOTS MODE ON <=====")
    do_plots = False

#: Whether to show plots, one by one, or not at all and just save them
interactive = True  # XXX dont keep it like this
interactive = False

#: Debug the memory consumption? Using :func:`Environment.memory_consumption.display_top_tracemalloc`.
debug_memory = False

if getenv('DEBUG', 'False') == 'True' and __name__ == '__main__':
    print("====> TURNING DEBUG MODE ON <=====")
    saveallfigs, interactive = False, True

if getenv('DEBUGMEMORY', 'False') == 'True' and __name__ == '__main__':
    print("====> TURNING DEBUGMEMORY MODE ON <=====")
    debug_memory = True

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

    evaluation = Evaluator(configuration, finalRanksOnAverage=finalRanksOnAverage, averageOn=averageOn)
    # Start the evaluation and then print final ranking and plot, for each environment
    N = len(evaluation.envs)

    for envId, env in enumerate(evaluation.envs):
        # # Plot histogram for rewards for that env
        # if do_plots and interactive:
        #     env.plotHistogram(evaluation.horizon * evaluation.repetitions)

        # (almost) unique hash from the configuration
        hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))

        if debug_memory: start_tracemalloc()  # DEBUG

        # Evaluate just that env
        evaluation.startOneEnv(envId, env)

        # Display the final regrets and rankings for that env
        print("\nGiving the vector of final regrets ...")
        evaluation.printLastRegrets(envId)
        print("\nGiving the final ranks ...")
        evaluation.printFinalRanking(envId)
        print("\nGiving the mean and std running times ...")
        evaluation.printRunningTimes(envId)
        print("\nGiving the mean and std memory consumption ...")
        evaluation.printMemoryConsumption(envId)
        if debug_memory: display_top_tracemalloc()  # DEBUG

        # Sub folder with a useful name
        subfolder = "SP__K{}_T{}_N{}__{}_algos".format(env.nbArms, configuration['horizon'], configuration['repetitions'], len(configuration['policies']))
        plot_dir = os.path.join(PLOT_DIR, subfolder)

        # Get the name of the output file
        imagename = "main____env{}-{}_{}".format(envId + 1, N, hashvalue)
        mainfig = os.path.join(plot_dir, imagename)
        savefig = mainfig
        picklename = mainfig + '.pickle'
        h5pyname = mainfig + '.hdf5'

        if saveallfigs:
            # Create the sub folder
            if os.path.isdir(plot_dir):
                print("{} is already a directory here...".format(plot_dir))
            elif os.path.isfile(plot_dir):
                raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
            else:
                mkdir(plot_dir)

            # Save it to a pickle file
            if USE_PICKLE:
                with open(picklename, 'wb') as picklefile:
                    print("Saving the Evaluator 'evaluation' objet to", picklename, "...")
                    pickle.dump(evaluation, picklefile, pickle.HIGHEST_PROTOCOL)
            if USE_HD5:
                evaluation.saveondisk(h5pyname)

        if not do_plots:
            break

        # --- Also plotting the running times
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_RunningTimes')
            print(" - Plotting the running times, and saving the plot to {} ...".format(savefig))
            evaluation.plotRunningTimes(envId, savefig=savefig)  # XXX To save the figure
        else:
            evaluation.plotRunningTimes(envId)  # XXX To plot without saving

        # --- Also plotting the memory consumption
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_MemoryConsumption')
            print(" - Plotting the memory consumption, and saving the plot to {} ...".format(savefig))
            evaluation.plotMemoryConsumption(envId, savefig=savefig)  # XXX To save the figure
        else:
            evaluation.plotMemoryConsumption(envId)  # XXX To plot without saving

        # --- Also plotting the regret
        if saveallfigs:
            print(" - Plotting the cumulative rewards, and saving the plot to {} ...".format(savefig))
            savefig = mainfig
            evaluation.plotRegrets(envId, savefig=savefig)  # XXX To save the figure
            if configuration['horizon'] >= 1000:
                savefig = mainfig.replace('main', 'main_semilogx')
                evaluation.plotRegrets(envId, savefig=savefig, semilogx=True)  # XXX To save the figure
            savefig = mainfig.replace('main', 'main_semilogy')
            evaluation.plotRegrets(envId, savefig=savefig, semilogy=True)  # XXX To save the figure
            if configuration['horizon'] >= 1000:
                savefig = mainfig.replace('main', 'main_loglog')
                evaluation.plotRegrets(envId, savefig=savefig, loglog=True)  # XXX To save the figure
            if configuration['repetitions'] > 1:
                if plotSTD:
                    savefig = savefig.replace('main', 'main_STD')
                    evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy, loglog=loglog, plotSTD=True)  # XXX To save the figure
                if plotMaxMin:
                    savefig = savefig.replace('main', 'main_MaxMin')
                    evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy, loglog=loglog, plotMaxMin=True)  # XXX To save the figure
        else:
            evaluation.plotRegrets(envId)  # XXX To plot without saving
            if configuration['horizon'] >= 1000:
                evaluation.plotRegrets(envId, semilogx=True)  # XXX To plot without saving
            evaluation.plotRegrets(envId, semilogy=True)  # XXX To plot without saving
            if configuration['horizon'] >= 1000:
                evaluation.plotRegrets(envId, loglog=True)
            if configuration['repetitions'] > 1:
                if plotSTD:
                    evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, plotSTD=True)  # XXX To plot without saving
                if plotMaxMin:
                    evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, plotMaxMin=True)  # XXX To plot without saving

        if meanRegret:
            if saveallfigs:
                savefig = mainfig.replace('main', 'main_MeanRewards')
                print(" - Plotting the mean rewards, and saving the plot to {} ...".format(savefig))
                evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy, loglog=loglog, meanRegret=True)  # XXX To save the figure
            else:
                evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, meanRegret=True)  # XXX To plot without saving

        if normalizedRegret:
            if saveallfigs:
                savefig = mainfig.replace('main', 'main_Normalized')
                print(" - Plotting the mean rewards, and saving the plot to {} ...".format(savefig))
                evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy, loglog=loglog, normalizedRegret=True)  # XXX To save the figure
                if configuration['repetitions'] > 1:
                    if plotSTD:
                        savefig = savefig.replace('main', 'main_STD')
                        evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy, loglog=loglog, normalizedRegret=True, plotSTD=True)  # XXX To save the figure
                    if plotMaxMin:
                        savefig = savefig.replace('main', 'main_MaxMin')
                        evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy, loglog=loglog, normalizedRegret=True, plotMaxMin=True)  # XXX To save the figure
            else:
                evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, normalizedRegret=True)  # XXX To plot without saving
                if configuration['repetitions'] > 1:
                    if plotSTD:
                        evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, normalizedRegret=True, plotSTD=True)  # XXX To plot without saving
                    if plotMaxMin:
                        evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, normalizedRegret=True, plotMaxMin=True)  # XXX To plot without saving

        # --- Also plotting the probability of picking the best arm
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_BestArmPulls')
            print(" - Plotting the probability of picking the best arm, and saving the plot to {} ...".format(savefig))
            evaluation.plotBestArmPulls(envId, savefig=savefig)  # XXX To save the figure
        else:
            evaluation.plotBestArmPulls(envId)  # XXX To plot without saving

        # --- Also plotting the histograms of regrets
        if saveallfigs:
            evaluation.plotLastRegrets(envId, subplots=False)
            savefig = mainfig.replace('main', 'main_HistogramsRegret')
            print(" - Plotting the histograms of regrets, and saving the plot to {} ...".format(savefig))
            for sharex, sharey in product([True, False], repeat=2):
                savefig = mainfig.replace('main', 'main_HistogramsRegret{}{}'.format(
                    "_shareX" if sharex else "",
                    "_shareY" if sharey else "",
                ))
                print("  and saving the plot to {} ...".format(savefig))
                evaluation.plotLastRegrets(envId, savefig=savefig, sharex=sharex, sharey=sharey)  # XXX To save the figure
            print(" - Plotting the histograms of regrets for each algorithm separately, and saving the plots ...")
            savefig = mainfig.replace('main', 'main_HistogramsRegret')
            print("  and saving the plot to {} ...".format(savefig))
            evaluation.plotLastRegrets(envId, all_on_separate_figures=True, savefig=savefig)  # XXX To save the figure
        else:
            evaluation.plotLastRegrets(envId, subplots=False)  # XXX To plot without saving
            for sharex, sharey in product([True, False], repeat=2):
                evaluation.plotLastRegrets(envId, sharex=sharex, sharey=sharey)  # XXX To plot without saving
            # evaluation.plotLastRegrets(envId, all_on_separate_figures=True)  # XXX To plot without saving

        if saveallfigs:
            print("\n\n==> To see the figures, do :\neog", os.path.join(plot_dir, "main*{}.png".format(hashvalue)))  # DEBUG
    # Done
    print("Done for simulations main.py ...")
    notify("Done for simulations main.py ...")
