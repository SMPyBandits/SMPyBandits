# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the multi-players case.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.1"

# Import arms
from Arms.Bernoulli import Bernoulli
# from Arms.Exponential import Exponential
# from Arms.Gaussian import Gaussian
# from Arms.Poisson import Poisson

# Import algorithms
from PoliciesMultiPlayers import *


# HORIZON : number of time steps of the experiments
# XXX Should be >= 10000 to be interesting "asymptotically"
HORIZON = 2000
HORIZON = 3000
HORIZON = 20000
HORIZON = 30000
HORIZON = 10000
HORIZON = 500

# REPETITIONS : number of repetitions of the experiments
# XXX Should be >= 10 to be stastically trustworthy
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
REPETITIONS = 50
REPETITIONS = 500
REPETITIONS = 200
REPETITIONS = 20
REPETITIONS = 100
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing

DO_PARALLEL = True
DO_PARALLEL = False  # XXX do not let this = False  # To profile the code, turn down parallel computing
N_JOBS = -1 if DO_PARALLEL else 1


# NB_PLAYERS : number of player.
NB_PLAYERS = 2
NB_PLAYERS = 1   # FIXME I should first check that the framework works well for 1 player


configuration = {
    # Duration of the experiment
    "horizon": HORIZON,
    # Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 5,  # Max joblib verbosity
    # Arms
    "environment": [
        # FIXME try with other arms distribution: Exponential, Gaussian, Poisson, etc!
        {   # A very very easy problem: 3 arms, one bad, one average, one good
            "arm_type": Bernoulli,
            "params": [0.1, 0.5, 0.9]
        },
        # {   # A very easy problem, but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # },
    ],
    # XXX Parameters for the multi-players setting
    # TODO first try with 1, then M stupid players, to check
    "players": [
        # --- Stupid algorithms
        {
            "archtype": Dummy,   # The stupidest policy
            "params": {}
        },
    ] * NB_PLAYERS
}

print("Loaded experiments configuration from 'configuration.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
