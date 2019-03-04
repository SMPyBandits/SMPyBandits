#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them, for the multi-players case.
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
    from Environment import EvaluatorMultiPlayers, notify, start_tracemalloc, display_top_tracemalloc
    for arg in sys.argv:
        if arg.startswith('configuration'):
            module_name = arg.replace('.py', '')
            print("Reading argument from command line, importing the configuration from arg = {} (module = {})...".format(arg, module_name))
            configuration_module = importlib.import_module(module_name)
    if configuration_module is None:
        import configuration_multiplayers as configuration_module
except ImportError:
    from SMPyBandits.save_configuration_for_reproducibility import save_configuration_for_reproducibility
    from SMPyBandits.Environment import EvaluatorMultiPlayers, notify, start_tracemalloc, display_top_tracemalloc
    for arg in sys.argv:
        if arg.startswith('configuration'):
            module_name = arg.replace('.py', '')
            print("Reading argument from command line, importing the configuration from arg = {} (module = {})...".format(arg, module_name))
            configuration_module = importlib.import_module(module_name, 'SMPyBandits')
    if configuration_module is None:
        import SMPyBandits.configuration_multiplayers as configuration_module

# Get the configuration dictionnary
configuration = configuration_module.configuration

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
# else:
#     import matplotlib
#     matplotlib.use("TkAgg")


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

        if debug_memory: start_tracemalloc()  # DEBUG

        # --- Also plotting the history of means
        if interactive:
            evaluation.plotHistoryOfMeans(envId)  # XXX To plot without saving

        # Evaluate just that env
        evaluation.startOneEnv(envId, env)

        # Display the final rankings for that env
        evaluation.printFinalRanking(envId)
        evaluation.printLastRegrets(envId)
        evaluation.printRunningTimes(envId)
        evaluation.printMemoryConsumption(envId)
        if debug_memory: display_top_tracemalloc()  # DEBUG

        if not do_plots:
            break

        # Sub folder with a useful name
        subfolder = "MP__K{}_M{}_T{}_N{}".format(env.nbArms, len(configuration['players']), configuration['horizon'], configuration['repetitions'])
        # Get the name of the output file
        imagename = "main____env{}-{}_{}".format(envId + 1, N, hashvalue)
        plot_dir = os.path.join(PLOT_DIR, subfolder)

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

            # --- DONE Copy (save) the current full configuration file to this folder as configuration__hashvalue.py
            # --- DONE Save just the configuration to a minimalist python file
            # TODO do the same on other main_*.py scripts
            save_configuration_for_reproducibility(
                configuration=configuration,
                configuration_module=configuration_module,
                plot_dir=plot_dir,
                hashvalue=hashvalue,
                main_name="main_multiplayers.py",
            )

            # Save it to a pickle file
            if USE_PICKLE:
                with open(picklename, 'wb') as picklefile:
                    print("Saving the EvaluatorMultiPlayers 'evaluation' objet to", picklename, "...")
                    pickle.dump(evaluation, picklefile, pickle.HIGHEST_PROTOCOL)
            if USE_HD5:
                evaluation.saveondisk(h5pyname)

        # --- Also plotting the history of means
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_HistoryOfMeans')
            print(" - Plotting the history of means, and saving the plot to {} ...".format(savefig))
            evaluation.plotHistoryOfMeans(envId, savefig=savefig)  # XXX To save the figure
        else:
            evaluation.plotHistoryOfMeans(envId)  # XXX To plot without saving

        # --- Also plotting the boxplot of regrets
        print("\n- Plotting the boxplot of regrets")
        if saveallfigs:
            savefig = mainfig.replace('main', 'main_BoxPlotRegret')
            print("  and saving the plot to {} ...".format(savefig))
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

        # Plotting the decentralized rewards
        print("\n\n- Plotting the decentralized rewards")
        if saveallfigs:
            savefig = mainfig
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
