# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for multi-players with aggregation.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

# Tries to know number of CPU
try:
    from multiprocessing import cpu_count
    CPU_COUNT = cpu_count()
except ImportError:
    CPU_COUNT = 1

from os import getenv
import numpy as np

if __name__ == '__main__':
    print("Warning: this script 'configuration_multiplayers_with_aggregation.py' is NOT executable. Use 'main_multiplayers.py configuration_multiplayers_with_aggregation' or 'make moremultiplayers_with_aggregation' or 'make moremultiplayers' ...")  # DEBUG
    exit(0)

try:
    # Import arms
    from Arms import *
    # Import contained classes
    from Environment import MAB
    # Collision Models
    from Environment.CollisionModels import *
    # Import algorithms, both single-player and multi-player
    from Policies import *
    from PoliciesMultiPlayers import *
except ImportError:
    from SMPyBandits.Arms import *
    from SMPyBandits.Environment import MAB
    from SMPyBandits.Environment.CollisionModels import *
    from SMPyBandits.Policies import *
    from SMPyBandits.PoliciesMultiPlayers import *


#: HORIZON : number of time steps of the experiments.
#: Warning Should be >= 10000 to be interesting "asymptotically".
HORIZON = 10000
HORIZON = int(getenv('T', HORIZON))

#: REPETITIONS : number of repetitions of the experiments.
#: Warning: Should be >= 10 to be statistically trustworthy.
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores, to h    ave exactly one repetition process by cores
REPETITIONS = 200
REPETITIONS = int(getenv('N', REPETITIONS))

#: To profile the code, turn down parallel computing
DO_PARALLEL = False  # XXX do not let this = False  # To profile the code, turn down parallel computing
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1) and DO_PARALLEL

#: Number of jobs to use for the parallel computations. -1 means all the CPU cores, 1 means no parallelization.
N_JOBS = -1 if DO_PARALLEL else 1
if CPU_COUNT > 4:  # We are on a server, let's be nice and not use all cores
    N_JOBS = min(CPU_COUNT, max(int(CPU_COUNT / 3), CPU_COUNT - 8))
N_JOBS = int(getenv('N_JOBS', N_JOBS))

#: NB_PLAYERS : number of players for the game. Should be >= 2 and <= number of arms.
NB_PLAYERS = 3    # Less that the number of arms
NB_PLAYERS = 6    # Less that the number of arms
NB_PLAYERS = int(getenv('M', NB_PLAYERS))
NB_PLAYERS = int(getenv('NB_PLAYERS', NB_PLAYERS))

#: The best collision model: none of the colliding users get any reward
collisionModel = onlyUniqUserGetsReward    # XXX this is the best one

# Parameters for the arms
VARIANCE = 0.05   #: Variance of Gaussian arms


#: Should we cache rewards? The random rewards will be the same for all the REPETITIONS simulations for each algorithms.
CACHE_REWARDS = False  # XXX to disable manually this feature


#: Number of arms for non-hard-coded problems (Bayesian problems)
NB_ARMS = 2 * NB_PLAYERS
NB_ARMS = int(getenv('K', NB_ARMS))
NB_ARMS = int(getenv('NB_ARMS', NB_ARMS))

#: Default value for the lower value of means
LOWER = 0.
#: Default value for the amplitude value of means
AMPLITUDE = 1.

#: Type of arms for non-hard-coded problems (Bayesian problems)
ARM_TYPE = "Bernoulli"
ARM_TYPE = str(getenv('ARM_TYPE', ARM_TYPE))
if ARM_TYPE in ["UnboundedGaussian"]:
    LOWER = -5
    AMPLITUDE = 10

LOWER = float(getenv('LOWER', LOWER))
AMPLITUDE = float(getenv('AMPLITUDE', AMPLITUDE))
assert AMPLITUDE > 0, "Error: invalid amplitude = {:.3g} but has to be > 0."  # DEBUG

ARM_TYPE_str = str(ARM_TYPE)
ARM_TYPE = mapping_ARM_TYPE[ARM_TYPE]

#: True to use bayesian problem
ENVIRONMENT_BAYESIAN = False
ENVIRONMENT_BAYESIAN = getenv('BAYES', str(ENVIRONMENT_BAYESIAN)) == 'True'

#: Means of arms for non-hard-coded problems (non Bayesian)
MEANS = uniformMeans(nbArms=NB_ARMS, delta=0.05, lower=LOWER, amplitude=AMPLITUDE, isSorted=True)

import numpy as np
# more parametric? Read from cli?
MEANS_STR = getenv('MEANS', '')
if MEANS_STR != '':
    MEANS = [ float(m) for m in MEANS_STR.replace('[', '').replace(']', '').split(',') ]
    print("Using cli env variable to use MEANS = {}.".format(MEANS))  # DEBUG


#: This dictionary configures the experiments
configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Collision model
    "collisionModel": collisionModel,
    # --- Other parameters for the Evaluator
    "finalRanksOnAverage": True,  # Use an average instead of the last value for the final ranking of the tested players
    "averageOn": 1e-3,  # Average the final rank on the 1.% last time steps
    # --- Should we plot the lower-bounds or not?
    "plot_lowerbounds": False,
    # --- Arms
    "environment": [
        {   # Use vector from command line
            "arm_type": ARM_TYPE,
            "params": MEANS
        },
    ],
}

if ENVIRONMENT_BAYESIAN:
    configuration["environment"] = [  # XXX Bernoulli arms
        {   # A Bayesian problem: every repetition use a different mean vectors!
            "arm_type": ARM_TYPE,
            "params": {
                "function": randomMeans,
                "args": {
                    "nbArms": NB_ARMS,
                    "mingap": 1. / (3 * NB_ARMS),
                    "lower": LOWER,
                    "amplitude": AMPLITUDE,
                    "isSorted": True,
                }
            }
        },
    ]


try:
    #: Number of arms *in the first environment*
    nbArms = int(configuration['environment'][0]['params']['args']['nbArms'])
except (TypeError, KeyError):
    nbArms = len(configuration['environment'][0]['params'])

if len(configuration['environment']) > 1:
    print("WARNING do not use this hack if you try to use more than one environment.")

#: Compute the gap of the first problem.
#: (for d in MEGA's parameters, and epsilon for MusicalChair's parameters)
try:
    GAP = np.min(np.diff(np.sort(configuration['environment'][0]['params'])))
except (ValueError, np.AxisError):
    print("Warning: using the default value for the GAP (Bayesian environment maybe?)")  # DEBUG
    GAP = 1. / (3 * NB_ARMS)


# FIXME very simulation dependent! Change manually...
list_of_change_labels = [
    # "Aggr(rhoRand, RandTopM, MCTopM, Selfish) UCB",
    "Aggr(rhoRand, RandTopM, MCTopM, Selfish) kl-UCB",
    # "Aggr(rhoRand, RandTopM, MCTopM, Selfish) MOSS",
    # "rhoRand UCB",
    "rhoRand kl-UCB",
    # "rhoRand MOSS",
    # "rhoRand Aggr(UCB, kl-UCB, MOSS)",
    # "RandTopM UCB",
    "RandTopM kl-UCB",
    # "RandTopM MOSS",
    # "RandTopM Aggr(UCB, kl-UCB, MOSS)",
    # "MCTopM UCB",
    "MCTopM kl-UCB",
    # "MCTopM MOSS",
    # "MCTopM Aggr(UCB, kl-UCB, MOSS)",
    # "Selfish UCB",
    "Selfish kl-UCB",
    # "Selfish MOSS",
    # "Selfish Aggr(UCB, kl-UCB, MOSS)",
    # "Centralized UCB",
    "Centralized kl-UCB",
    # "Centralized MOSS",
    # "Centralized Aggr(UCB, kl-UCB, MOSS)",
]

configuration["change_labels"] = {
    i: label for i, label in enumerate(list_of_change_labels)
}


configuration["successive_players"] = [

    # ---- Aggregating for the multi-user parts
    # [
    #     Aggregator(nbArms, children=[
    #         rhoRand(NB_PLAYERS, nbArms, UCB).children[j],
    #         RandTopM(NB_PLAYERS, nbArms, UCB).children[j],
    #         MCTopM(NB_PLAYERS, nbArms, UCB).children[j],
    #         Selfish(NB_PLAYERS, nbArms, UCB).children[j],
    #     ])
    #     for j in range(NB_PLAYERS)
    # ],
    # [
    #     Aggregator(nbArms, children=[
    #         rhoRand(NB_PLAYERS, nbArms, klUCB).children[j],
    #         RandTopM(NB_PLAYERS, nbArms, klUCB).children[j],
    #         MCTopM(NB_PLAYERS, nbArms, klUCB).children[j],
    #         # Selfish(NB_PLAYERS, nbArms, klUCB).children[j],
    #     ])
    #     for j in range(NB_PLAYERS)
    # ],
    # [
    #     Aggregator(nbArms, children=[
    #         rhoRand(NB_PLAYERS, nbArms, EmpiricalMeans).children[j],
    #         RandTopM(NB_PLAYERS, nbArms, EmpiricalMeans).children[j],
    #         MCTopM(NB_PLAYERS, nbArms, EmpiricalMeans).children[j],
    #         # Selfish(NB_PLAYERS, nbArms, EmpiricalMeans).children[j],
    #     ])
    #     for j in range(NB_PLAYERS)
    # ],
    [
        Aggregator(nbArms, children=[
            rhoRand(NB_PLAYERS, nbArms, Aggregator, children=[UCB, klUCB, EmpiricalMeans]).children[j],
            RandTopM(NB_PLAYERS, nbArms, Aggregator, children=[UCB, klUCB, EmpiricalMeans]).children[j],
            MCTopM(NB_PLAYERS, nbArms, Aggregator, children=[UCB, klUCB, EmpiricalMeans]).children[j],
            # Selfish(NB_PLAYERS, nbArms, Aggregator, children=[UCB, klUCB, EmpiricalMeans]).children[j],
        ])
        for j in range(NB_PLAYERS)
    ],
    # ---- rhoRand etc
    # rhoRand(NB_PLAYERS, nbArms, UCB).children,
    # rhoRand(NB_PLAYERS, nbArms, klUCB).children,
    # rhoRand(NB_PLAYERS, nbArms, EmpiricalMeans).children,
    rhoRand(NB_PLAYERS, nbArms, Aggregator, children=[UCB, klUCB, EmpiricalMeans]).children,

    # # ---- RandTopM
    # RandTopM(NB_PLAYERS, nbArms, UCB).children,
    # RandTopM(NB_PLAYERS, nbArms, klUCB).children,
    # RandTopM(NB_PLAYERS, nbArms, EmpiricalMeans).children,
    RandTopM(NB_PLAYERS, nbArms, Aggregator, children=[UCB, klUCB, EmpiricalMeans]).children,

    # ---- MCTopM
    # MCTopM(NB_PLAYERS, nbArms, UCB).children,
    # MCTopM(NB_PLAYERS, nbArms, klUCB).children,
    # MCTopM(NB_PLAYERS, nbArms, EmpiricalMeans).children,
    MCTopM(NB_PLAYERS, nbArms, Aggregator, children=[UCB, klUCB, EmpiricalMeans]).children,

    # ---- Selfish
    # Selfish(NB_PLAYERS, nbArms, UCB).children,
    # Selfish(NB_PLAYERS, nbArms, klUCB).children,
    # Selfish(NB_PLAYERS, nbArms, EmpiricalMeans).children,
    Selfish(NB_PLAYERS, nbArms, Aggregator, children=[UCB, klUCB, EmpiricalMeans]).children,

    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, UCB).children,
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, klUCB).children,
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, EmpiricalMeans).children,
    CentralizedMultiplePlay(NB_PLAYERS, nbArms, Aggregator, children=[UCB, klUCB, EmpiricalMeans]).children,
]


configuration.update({
    "players": Selfish(NB_PLAYERS, nbArms, UCB).children
})


# DONE
print("Loaded experiments configuration from 'configuration_multiplayers_with_aggregation.py' :")
print("configuration =", configuration)  # DEBUG
