# -*- coding: utf-8 -*-
"""
An example of a configuration file to launch some the simulations, for the single-player case.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


CPU_COUNT = 4

from os import getenv
import numpy as np

# Import arms
from Arms import *
# Import contained classes
from Environment import MAB
# Collision Models
from Environment.CollisionModels import *
# Import algorithms, both single-player and multi-player
from Policies import *
from PoliciesMultiPlayers import *


#: HORIZON : number of time steps of the experiments.
#: Warning Should be >= 10000 to be interesting "asymptotically".
HORIZON = 10000
HORIZON = int(getenv('T', HORIZON))

#: REPETITIONS : number of repetitions of the experiments.
#: Warning: Should be >= 10 to be statistically trustworthy.
REPETITIONS = 10
REPETITIONS = int(getenv('N', REPETITIONS))

#: To profile the code, turn down parallel computing
DO_PARALLEL = True

#: Number of jobs to use for the parallel computations. -1 means all the CPU cores, 1 means no parallelization.
N_JOBS = -1 if DO_PARALLEL else 1
N_JOBS = int(getenv('N_JOBS', N_JOBS))

#: NB_PLAYERS : number of players for the game. Should be >= 2 and <= number of arms.
NB_PLAYERS = 3    # Less that the number of arms
NB_PLAYERS = int(getenv('M', NB_PLAYERS))
NB_PLAYERS = int(getenv('NB_PLAYERS', NB_PLAYERS))

#: The best collision model: none of the colliding users get any reward
collisionModel = onlyUniqUserGetsReward    # XXX this is the best one

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
    "plot_lowerbounds": True,  # XXX Default
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


configuration["successive_players"] = [
    # XXX This new SIC_MMAB algorithm
    [ SIC_MMAB(nbArms, HORIZON) for _ in range(NB_PLAYERS) ],
    [ SIC_MMAB_UCB(nbArms, HORIZON) for _ in range(NB_PLAYERS) ],
    [ SIC_MMAB_klUCB(nbArms, HORIZON) for _ in range(NB_PLAYERS) ],

    # ---- rhoRand etc
    rhoRand(NB_PLAYERS, nbArms, UCB).children,
    rhoRand(NB_PLAYERS, nbArms, klUCB).children,

    # # ---- RandTopM
    RandTopM(NB_PLAYERS, nbArms, UCB).children,
    RandTopM(NB_PLAYERS, nbArms, klUCB).children,

    # ---- MCTopM
    MCTopM(NB_PLAYERS, nbArms, UCB).children,
    MCTopM(NB_PLAYERS, nbArms, klUCB).children,

    # # # ---- Selfish
    Selfish(NB_PLAYERS, nbArms, UCB).children,
    Selfish(NB_PLAYERS, nbArms, klUCB).children,

    # # --- XXX MusicalChairNoSensing (selfish), a better Musical Chair
    # [ MusicalChairNoSensing(nbPlayers=NB_PLAYERS, nbArms=nbArms, horizon=HORIZON) for _ in range(NB_PLAYERS) ],

    # # --- Centralized multiple play
    CentralizedMultiplePlay(NB_PLAYERS, nbArms, UCB).children,
    CentralizedMultiplePlay(NB_PLAYERS, nbArms, klUCB).children,

    # # XXX stupid version with fixed T0 : cannot adapt to any problem
    [ MusicalChair(nbArms, Time0=50*NB_ARMS) for _ in range(NB_PLAYERS) ],
    [ MusicalChair(nbArms, Time0=100*NB_ARMS) for _ in range(NB_PLAYERS) ],
    [ MusicalChair(nbArms, Time0=150*NB_ARMS) for _ in range(NB_PLAYERS) ],

    # # XXX cheated version, with known gap (epsilon < Delta) and proba of success 5% !
    # [ MusicalChair(nbArms, Time0=optimalT0(nbArms=NB_ARMS, epsilon=0.99*GAP, delta=0.5)) for _ in range(NB_PLAYERS) ],
    # [ MusicalChair(nbArms, Time0=optimalT0(nbArms=NB_ARMS, epsilon=0.99*GAP, delta=0.1)) for _ in range(NB_PLAYERS) ],

    # # XXX cheated version, with known gap and known horizon (proba of success delta < 1 / T) !
    # [ MusicalChair(nbArms, Time0=optimalT0(nbArms=NB_ARMS, epsilon=0.99*GAP, delta=1./(1+HORIZON))) for _ in range(NB_PLAYERS) ],
]

configuration.update({
    # --- DONE Using multi-player Selfish policy
    "players": Selfish(NB_PLAYERS, nbArms, UCB).children
})

# DONE
print("Loaded experiments configuration from 'example_of_configuration_multiplayers.py' :")
print("configuration =", configuration)  # DEBUG
