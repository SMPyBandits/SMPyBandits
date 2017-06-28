#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them.
"""
from __future__ import print_function, division

__author__ = "Lilian Besson"
__version__ = "0.6"

# Generic imports
import sys
from os import mkdir
import os.path
from os import getenv

# Backup evaluation object
import pickle
# import h5py

# Local imports
from Environment import Evaluator, notify

# Import a configuration file
if 'configuration_comparing_KLUCB_aggregation' in sys.argv:
    from configuration_comparing_KLUCB_aggregation import configuration
elif 'configuration_markovian' in sys.argv:
    from configuration_markovian import configuration
else:
    from configuration import configuration

# Solving https://github.com/Naereen/AlgoBandits/issues/15#issuecomment-292484493
# For instance, call SLEEP=12h to delay the simulation for 12hours
if getenv('SLEEP', False):
    from subprocess import call
    SLEEP = str(getenv('SLEEP'))
    print("\nSleeping for", SLEEP, "before starting the simulation...")  # DEBUG
    call(["sleep", SLEEP])  # more general
    print("Done sleeping for", SLEEP, "... Now I can start the simulation...")

USE_PICKLE = True   #: Should we save the Evaluator object to a .pickle file at the end of the simulation?

# Parameters for the plots (where to save them) and what to draw
PLOT_DIR = "plots"  #: Directory for the plots
semilogx = False  #: Plot in semilogx by default?
semilogy = False  #: Plot in semilogy by default?
loglog   = False  #: Plot in loglog   by default?
meanRegret = True  #: Plot mean regret ?
normalizedRegret = False  #: Plot instantaneous regret?

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

#: Whether to show plots, one by one, or not at all and just save them
interactive = True  # XXX dont keep it like this
interactive = False

if str(getenv('DEBUG', False)) == 'True' and __name__ == '__main__':
    print("====> TURNING DEBUG MODE ON <=====")
    saveallfigs, interactive = False, True

if getenv('XKCD', False) and interactive and not saveallfigs:
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

        # Evaluate just that env
        evaluation.startOneEnv(envId, env)

        # Display the final rankings for that env
        print("Giving the final ranks ...")
        evaluation.printFinalRanking(envId)

        # Sub folder with a useful name
        subfolder = "T{}_N{}__{}_algos".format(configuration['horizon'], configuration['repetitions'], len(configuration['policies']))
        plot_dir = os.path.join(PLOT_DIR, subfolder)

        # Get the name of the output file
        imagename = "main____env{}-{}_{}".format(envId + 1, N, hashvalue)
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
                    print("Saving the Evaluator 'evaluation' objet to", picklename, "...")
                    pickle.dump(evaluation, picklefile, pickle.HIGHEST_PROTOCOL)

            # h5pydb = h5pyfile.create_dataset("results", (XXX, XXX))
            # Save the internal vectorial memory of the evaluator object
            # rewards = np.zeros((self.nbPolicies, len(self.envs), self.duration))
            # rewardsSquared = np.zeros((self.nbPolicies, len(self.envs), self.duration))
            # BestArmPulls = np.zeros((self.nbPolicies, self.duration))
            # pulls = np.zeros((self.nbPolicies, env.nbArms))

        if not do_plots:
            break

        if saveallfigs:
            print(" - Plotting the cumulative rewards, and saving the plot to {} ...".format(savefig))
            evaluation.plotRegrets(envId, savefig=savefig)  # XXX To save the figure
            savefig = mainfig.replace('main', 'main_semilogx')
            evaluation.plotRegrets(envId, savefig=savefig, semilogx=True)  # XXX To save the figure
            savefig = mainfig.replace('main', 'main_semilogy')
            evaluation.plotRegrets(envId, savefig=savefig, semilogy=True)  # XXX To save the figure
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
            evaluation.plotRegrets(envId)
            evaluation.plotRegrets(envId, semilogx=True)
            evaluation.plotRegrets(envId, semilogy=True)
            # evaluation.plotRegrets(envId, loglog=True)
            if configuration['repetitions'] > 1:
                if plotSTD:
                    evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, plotSTD=True)
                if plotMaxMin:
                    evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, plotMaxMin=True)

        if meanRegret:
            if saveallfigs:
                savefig = mainfig.replace('main', 'main_MeanRewards')
                print(" - Plotting the mean rewards, and saving the plot to {} ...".format(savefig))
                evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy, loglog=loglog, meanRegret=True)  # XXX To save the figure
            else:
                evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, meanRegret=True)

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
                evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, normalizedRegret=True)
                if configuration['repetitions'] > 1:
                    if plotSTD:
                        evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, normalizedRegret=True, plotSTD=True)
                    if plotMaxMin:
                        evaluation.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, normalizedRegret=True, plotMaxMin=True)

        # --- Also plotting the probability of picking the best arm
        if evaluation.random_shuffle or evaluation.random_invert:
            print(" - Not plotting probability of picking the best arm as we used random events ...")
            print("   ==> FIXME correct this bug")
        else:
            if saveallfigs:
                savefig = mainfig.replace('main', 'main_BestArmPulls')
                print(" - Plotting the results, and saving the plot to {} ...".format(savefig))
                evaluation.plotBestArmPulls(envId, savefig=savefig)  # XXX To save the figure
            else:
                evaluation.plotBestArmPulls(envId)

        if saveallfigs:
            print("\n\n==> To see the figures, do :\neog", os.path.join(plot_dir, "main*{}.png".format(hashvalue)))  # DEBUG
    # Done
    print("Done for simulations main.py ...")
    notify("Done for simulations main.py ...")
