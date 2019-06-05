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
from os import mkdir, getenv
import os.path
import importlib
# Backup evaluation object
import pickle

# Local imports
configuration_module = None
try:
    from save_configuration_for_reproducibility import save_configuration_for_reproducibility
    from Environment import Evaluator, notify, start_tracemalloc, display_top_tracemalloc
    # Import a configuration file
    for arg in sys.argv:
        if "configuration" in arg:
            filename = arg.replace('.py', '')
            dirname, module_name = os.path.dirname(filename), os.path.basename(filename)
            sys.path.insert(0, dirname)
            print("Reading argument from command line, importing the configuration module from arg = {} (module = {} in directory {})...".format(arg, module_name, dirname))
            configuration_module = importlib.import_module(module_name)
    if configuration_module is None:
        import configuration as configuration_module
except ImportError:
    from SMPyBandits.save_configuration_for_reproducibility import save_configuration_for_reproducibility
    from SMPyBandits.Environment import Evaluator, notify, start_tracemalloc, display_top_tracemalloc
    for arg in sys.argv:
        if "configuration" in arg:
            filename = arg.replace('.py', '')
            dirname, module_name = os.path.dirname(filename), os.path.basename(filename)
            sys.path.insert(0, dirname)
            print("Reading argument from command line, importing the configuration module from arg = {} (module = {} in directory {})...".format(arg, module_name, dirname))
            configuration_module = importlib.import_module('.{}'.format(module_name), package='SMPyBandits')
    if configuration_module is None:
        import SMPyBandits.configuration as configuration_module

# Get the configuration dictionnary
configuration = configuration_module.configuration

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
meanReward = True  #: Plot mean regret ?
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
    import matplotlib as mpl
    FIGSIZE = (19.80, 10.80)  #: Figure size, in inches!
    # FIGSIZE = (16, 9)  #: Figure size, in inches!
    mpl.rcParams['figure.figsize'] = FIGSIZE

if getenv('XKCD', 'False') == 'True' and interactive and not saveallfigs:
    import matplotlib.pyplot as plt
    plt.xkcd()  # XXX turn on XKCD-like style ?! cf. http://matplotlib.org/xkcd/ for more details

# FIXED try to switch to a non interactive backend when running without DEBUG=True
# https://matplotlib.org/api/matplotlib_configuration_api.html?highlight=matplotlib%20use#matplotlib.use
if not interactive:
    import matplotlib
    print("Warning: Non interactive simulations, switching from '{}' backend to 'agg'...".format(matplotlib.get_backend()))  # DEBUG
    matplotlib.use("agg", warn=True, force=True)


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

        # --- Also plotting the history of means
        if interactive:
            evaluation.plotHistoryOfMeans(envId)  # XXX To plot without saving

        # Evaluate just that env
        evaluation.startOneEnv(envId, env)

        # Display the final regrets and rankings for that env
        evaluation.printLastRegrets(envId)
        evaluation.printFinalRanking(envId, moreAccurate=True)
        evaluation.printRunningTimes(envId)
        evaluation.printMemoryConsumption(envId)
        evaluation.printNumberOfCPDetections(envId)
        if debug_memory: display_top_tracemalloc()  # DEBUG

        # Sub folder with a useful name
        subfolder = "SP__K{}_T{}_N{}__{}_algos".format(env.nbArms, configuration['horizon'], configuration['repetitions'], len(configuration['policies']))
        plot_dir = os.path.join(PLOT_DIR, subfolder)
        # Get the name of the output file
        imagename = "main____env{}-{}_{}".format(envId + 1, N, hashvalue)
        mainfig = os.path.join(plot_dir, imagename)
        savefig = mainfig
        picklename = mainfig + '.pickle'
        h5pyname   = mainfig + '.hdf5'

        if saveallfigs:
            # Create the sub folder
            if os.path.isdir(plot_dir):
                print("{} is already a directory here...".format(plot_dir))
            elif os.path.isfile(plot_dir):
                raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
            else:
                mkdir(plot_dir)

            # --- DONE Copy (save) the current full configuration file to this folder as configuration__hashvalue.py
            # --- DONE Save just the configuration to a minimalist python file
            # TODO do the same on other main_*.py scripts
            save_configuration_for_reproducibility(
                configuration=configuration,
                configuration_module=configuration_module,
                plot_dir=plot_dir,
                hashvalue="env{}-{}_{}".format(envId + 1, N, hashvalue),
                main_name="main.py",
            )
            # --- Save it to a pickle file
            if USE_PICKLE:
                with open(picklename, 'wb') as picklefile:
                    print("Saving the Evaluator 'evaluation' objet to", picklename, "...")
                    pickle.dump(evaluation, picklefile, pickle.HIGHEST_PROTOCOL)
            # --- Save it to a HD5 file
            if USE_HD5:
                evaluation.saveondisk(h5pyname)

        if not do_plots:
            continue  # XXX don't use break, it exit the loop on different environments

        # --- Also plotting the history of means
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_HistoryOfMeans')
            print(" - Plotting the history of means, and saving the plot to {} ...".format(savefig))
            evaluation.plotHistoryOfMeans(envId, savefig=savefig)  # XXX To save the figure
        else:
            evaluation.plotHistoryOfMeans(envId)  # XXX To plot without saving

        # --- Also plotting the boxplot of last regrets
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_BoxPlotRegret')
            evaluation.plotLastRegrets(envId, boxplot=True, savefig=savefig)
        else:
            evaluation.plotLastRegrets(envId, boxplot=True)  # XXX To plot without saving

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

        # --- Also plotting the number of detected change-points
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_NumberOfCPDetections')
            print(" - Plotting the memory consumption, and saving the plot to {} ...".format(savefig))
            evaluation.plotNumberOfCPDetections(envId, savefig=savefig)  # XXX To save the figure
        else:
            evaluation.plotNumberOfCPDetections(envId)  # XXX To plot without saving

        if meanReward:
            if saveallfigs:
                savefig = mainfig.replace('main', 'main_MeanRewards')
                print(" - Plotting the mean rewards, and saving the plot to {} ...".format(savefig))
                evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy, loglog=loglog, meanReward=True)  # XXX To save the figure
            else:
                evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, meanReward=True)  # XXX To plot without saving

        # --- Also plotting the regret
        if saveallfigs:
            print(" - Plotting the cumulative rewards, and saving the plot to {} ...".format(savefig))
            savefig = mainfig
            evaluation.plotRegrets(envId, savefig=savefig, moreAccurate=True)  # XXX To save the figure
            savefig = mainfig.replace('main', 'main_LessAccurate')
            evaluation.plotRegrets(envId, savefig=savefig, moreAccurate=False)  # XXX To save the figure
            savefig = mainfig.replace('main', 'main_BestArmPulls')
            print(" - Plotting the probability of picking the best arm, and saving the plot to {} ...".format(savefig))
            # --- Also plotting the probability of picking the best arm
            evaluation.plotBestArmPulls(envId, savefig=savefig)  # XXX To save the figure
            # if configuration['horizon'] >= 1000:
            #     savefig = mainfig.replace('main', 'main_semilogx')
            #     evaluation.plotRegrets(envId, savefig=savefig, semilogx=True)  # XXX To save the figure
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
            evaluation.plotRegrets(envId, moreAccurate=True)  # XXX To plot without saving
            evaluation.plotRegrets(envId, moreAccurate=False)  # XXX To plot without saving
            # --- Also plotting the probability of picking the best arm
            evaluation.plotBestArmPulls(envId)  # XXX To plot without saving
            # if configuration['horizon'] >= 1000:
            #     evaluation.plotRegrets(envId, semilogx=True)  # XXX To plot without saving
            evaluation.plotRegrets(envId, semilogy=True)  # XXX To plot without saving
            if configuration['horizon'] >= 1000:
                evaluation.plotRegrets(envId, loglog=True)
            if configuration['repetitions'] > 1:
                if plotSTD:
                    evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, plotSTD=True)  # XXX To plot without saving
                if plotMaxMin:
                    evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, plotMaxMin=True)  # XXX To plot without saving

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

        # --- Also plotting the histograms of regrets
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_HistogramsRegret')
            evaluation.plotLastRegrets(envId, subplots=False, savefig=savefig)
            print(" - Plotting the histograms of regrets, and saving the plot to {} ...".format(savefig))
            # for sharex, sharey in product([True, False], repeat=2):  # XXX 3 out of 4 were UGLY!
            for sharex, sharey in [(True, False)]:
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
            # for sharex, sharey in product([True, False], repeat=2):  # XXX 3 out of 4 were UGLY!
            for sharex, sharey in [(True, False)]:
                evaluation.plotLastRegrets(envId, sharex=sharex, sharey=sharey)  # XXX To plot without saving
            # evaluation.plotLastRegrets(envId, all_on_separate_figures=True)  # XXX To plot without saving

        if saveallfigs:
            print("\n\n==> To see the figures, do :\neog", os.path.join(plot_dir, "main*{}.png".format(hashvalue)))  # DEBUG
    # Done
    print("Done for simulations main.py ...")
    notify("Done for simulations main.py ...")
