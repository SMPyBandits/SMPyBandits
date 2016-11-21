#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them.
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

from os import mkdir
import os.path
from Environment import Evaluator
from configuration import configuration

plot_dir = "plots"


if __name__ == '__main__':
    if os.path.isdir(plot_dir):
        print("{} is already a directory here...".format(plot_dir))
    elif os.path.isfile(plot_dir):
        raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
    else:
        mkdir(plot_dir)
    evaluation = Evaluator(configuration)
    N = len(evaluation.envs)
    evaluation.start()
    for i in range(N):
        # XXX be more explicit here
        hashvalue = hash((tuple(configuration.keys()), configuration.values()))  # almost unique hash from the configuration
        imagename = "main__{}_{}-{}.png".format(hashvalue, i + 1, N)
        evaluation.plotResults(i, savefig=os.path.join(plot_dir, imagename))
