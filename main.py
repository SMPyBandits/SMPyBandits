#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them.
"""
from __future__ import print_function, division

__author__ = "Lilian Besson"
__version__ = "0.5"

# Generic imports
from os import mkdir
import os.path

# Backup evaluation object
# import pickle
# import h5py

import matplotlib.pyplot as plt

# Local imports
from Environment import Evaluator, notify

# Import a configuration file
from configuration import configuration
# from configuration_comparing_KLUCB_aggregation import configuration


# Parameters for the plots (where to save them) and what to draw
PLOT_DIR = "plots"
semilogx = False
meanRegret = True
normalizedRegret = False
plotSTD = False

saveallfigs = False
saveallfigs = True  # XXX dont keep it like this when experimenting

# if not saveallfigs:
#     plt.xkcd()  # XXX turn on XKCD-like style ?! cf. http://matplotlib.org/xkcd/ for more details

# Parameters for the Evaluator object
finalRanksOnAverage = True     # Use an average instead of the last value for the final ranking of the tested policies
averageOn = 1e-2               # Average the final rank on the 1% last time steps

# Whether to do the plots or not
do_plot = False
do_plot = True

# Whether to show plots, one by one, or not at all and just save them
interactive = False  # Seems to be the only mode which is working well
interactive = True

# Update configuration
configuration['showplot'] = not interactive


if __name__ == '__main__':
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
        # (almost) unique hash from the configuration
        hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))

        # Sub folder with a useful name
        subfolder = "T{}_N{}__{}_algos".format(configuration['horizon'], configuration['repetitions'], len(configuration['policies']))
        plot_dir = os.path.join(PLOT_DIR, subfolder)

        # Get the name of the output file
        imagename = "main____env{}-{}_{}.png".format(envId + 1, N, hashvalue)
        mainfig = os.path.join(plot_dir, imagename)
        savefig = mainfig

        # FIXME finish this to also save result in a HDF5 file!
        # h5pyname = mainfig.replace('.png', '.hdf5')
        # h5pyfile = h5py.File(h5pyname, 'w')
        picklename = mainfig.replace('.png', '.pickle')

        # Evaluate just that env
        evaluation.startOneEnv(envId, env)

        if saveallfigs:
            # Create the sub folder
            if os.path.isdir(plot_dir):
                print("{} is already a directory here...".format(plot_dir))
            elif os.path.isfile(plot_dir):
                raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
            else:
                mkdir(plot_dir)

            # # Save it to a pickle file
            # # TODO use numpy.savez_compressed instead ? https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed
            # with open(picklename, 'wb') as picklefile:
            #     print("Saving the 'evaluation' objet to", picklefile, "...")
            #     pickle.dump(evaluation, picklefile, pickle.HIGHEST_PROTOCOL)

        # h5pydb = h5pyfile.create_dataset("results", (XXX, XXX))
        # Save the internal vectorial memory of the evaluator object
        # rewards = np.zeros((self.nbPolicies, len(self.envs), self.duration))
        # rewardsSquared = np.zeros((self.nbPolicies, len(self.envs), self.duration))
        # BestArmPulls = np.zeros((self.nbPolicies, self.duration))
        # pulls = np.zeros((self.nbPolicies, env.nbArms))

        # Display the final rankings for that env
        print("Giving the final ranks ...")
        evaluation.printFinalRanking(envId)
        if not do_plot:
            break

        if saveallfigs:
            print(" - Plotting the cumulative rewards, and saving the plot to {} ...".format(savefig))
            evaluation.plotRegrets(envId, savefig=savefig, semilogx=False)  # XXX To save the figure
            savefig = mainfig.replace('main', 'main_semilogx')
            evaluation.plotRegrets(envId, savefig=savefig, semilogx=True)  # XXX To save the figure
        else:
            evaluation.plotRegrets(envId, semilogx=False, plotSTD=False)
            evaluation.plotRegrets(envId, semilogx=True, plotSTD=False)
            # if configuration['repetitions'] > 1: evaluation.plotRegrets(envId, semilogx=semilogx, plotSTD=True)

        if meanRegret:
            if saveallfigs:
                savefig = mainfig.replace('main', 'main_MeanRewards')
                print(" - Plotting the mean rewards, and saving the plot to {} ...".format(savefig))
                evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx, meanRegret=True)  # XXX To save the figure
            else:
                evaluation.plotRegrets(envId, semilogx=semilogx, meanRegret=True, plotSTD=False)
                # if configuration['repetitions'] > 1: evaluation.plotRegrets(envId, semilogx=semilogx, meanRegret=True, plotSTD=True)

        if normalizedRegret:
            if saveallfigs:
                savefig = mainfig.replace('main', 'main_Normalized')
                print(" - Plotting the mean rewards, and saving the plot to {} ...".format(savefig))
                evaluation.plotRegrets(envId, savefig=savefig, semilogx=semilogx, normalizedRegret=True)  # XXX To save the figure
            else:
                evaluation.plotRegrets(envId, semilogx=semilogx, normalizedRegret=True, plotSTD=False)
                # if configuration['repetitions'] > 1: evaluation.plotRegrets(envId, semilogx=semilogx, normalizedRegret=True, plotSTD=True)

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
    # Done
    print("Done for simulations main.py ...")
    notify("Done for simulations main.py ...")
