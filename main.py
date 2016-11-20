#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them.
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from Environment import Evaluator
from configuration import configuration


if __name__ == '__main__':
    evaluation = Evaluator(configuration)
    N = len(evaluation.envs)
    evaluation.start()
    for i in range(N):
        # XXX be more explicit here
        hashvalue = hash((tuple(configuration.keys()), configuration.values()))  # almost unique hash from the configuration
        imagename = "main__{}_{}-{}.png".format(hashvalue, i + 1, N)
        evaluation.plotResults(i, savefig=imagename)
