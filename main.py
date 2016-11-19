#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load the config, run the simulations, and plot them.
"""

__author__ = "Lilian Besson, Emilie Kaufmann"
__version__ = "0.1"

from Environment import Evaluator
# XXX could this be read from a JSON file instead?
from configuration import configuration


if __name__ == '__main__':
    evaluation = Evaluator(configuration)
    N = len(evaluation.envs)
    evaluation.start()
    for i in range(N):
        # XXX be more parametric here
        imagename = "main__{}-{}.png".format(i + 1, N)
        evaluation.plotResults(i, savefig=imagename)
