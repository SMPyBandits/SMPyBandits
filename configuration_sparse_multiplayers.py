# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the multi-players case with sparse activated players.
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
    print("Warning: this script 'configuration_sparse_multiplayers.py' is NOT executable. Use 'main_sparse_multiplayers.py' or 'make sparsemulti' ...")  # DEBUG
    exit(0)

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
#: Warning: Should be >= 10 to be stastically trustworthy.
REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
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
NB_PLAYERS = 4    # Less that the number of arms
NB_PLAYERS = int(getenv('M', NB_PLAYERS))
NB_PLAYERS = int(getenv('NB_PLAYERS', NB_PLAYERS))

#: ACTIVATION : probability of activation of each player.
ACTIVATION = 0.5
ACTIVATION = float(getenv('P', ACTIVATION))


# Parameters for the arms
VARIANCE = 0.05   #: Variance of Gaussian arms

#: Number of arms for non-hard-coded problems (Bayesian problems)
NB_ARMS = 2
NB_ARMS = int(getenv('K', NB_ARMS))
NB_ARMS = int(getenv('NB_ARMS', NB_ARMS))

#: Type of arms for non-hard-coded problems (Bayesian problems)
ARM_TYPE = "Bernoulli"
ARM_TYPE = str(getenv('ARM_TYPE', ARM_TYPE))
mapping_ARM_TYPE = {
    "Constant": Constant,
    "Uniform": Uniform,
    "Bernoulli": Bernoulli, "B": Bernoulli,
    "Gaussian": Gaussian, "Gauss": Gaussian, "G": Gaussian,
    "Poisson": Poisson, "P": Poisson,
    "Exponential": ExponentialFromMean, "Exp": ExponentialFromMean, "E": ExponentialFromMean,
    "Gamma": GammaFromMean,
}
ARM_TYPE = mapping_ARM_TYPE[ARM_TYPE]


#: This dictionary configures the experiments
configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # --- Probability of activation of each player
    "activation": ACTIVATION,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Other parameters for the Evaluator
    "finalRanksOnAverage": True,  # Use an average instead of the last value for the final ranking of the tested players
    "averageOn": 1e-3,  # Average the final rank on the 1.% last time steps
    # --- Arms
    "environment": [
        # {   # A Bayesian problem: every repetition use a different mean vectors!
        #     "arm_type": ARM_TYPE,
        #     "params": {
        #         "function": randomMeans,
        #         "args": {
        #             "nbArms": NB_ARMS,
        #             "mingap": None,
        #             # "mingap": 0.05,
        #             # "mingap": 1. / (3. * NB_ARMS),
        #             "lower": 0.,
        #             "amplitude": 1.,
        #             "isSorted": True,
        #         }
        #     }
        # },
        # XXX Default!
        {   # A very easy problem (X arms), but it is used in a lot of articles
            "arm_type": ARM_TYPE,
            "params": uniformMeans(NB_ARMS, 1 / (1. + NB_ARMS))
        }
        # {   # A Bayesian problem: every repetition use a different mean vectors!
        #     "arm_type": ARM_TYPE,
        #     "params": {
        #         "function": randomMeansWithGapBetweenMbestMworst,
        #         "args": {
        #             "nbArms": NB_ARMS,
        #             "nbPlayers": NB_PLAYERS,
        #             "mingap": 0.1,
        #             "lower": 0.,
        #             "amplitude": 1.,
        #             "isSorted": True,
        #         }
        #     }
        # },
    ],
}


try:
    #: Number of arms *in the first environment*
    nbArms = int(configuration['environment'][0]['params']['args']['nbArms'])
except (TypeError, KeyError):
    nbArms = len(configuration['environment'][0]['params'])

if len(configuration['environment']) > 1:
    print("WARNING do not use this hack if you try to use more than one environment.")
# XXX compute optimal values for d (MEGA's parameter)
# D = max(0.01, np.min(np.diff(np.sort(configuration['environment'][0]['params']))) / 2)


configuration["successive_players"] = [
    # --- Comparing Selfish, rhoRand, rhoLearn, RandTopM for klUCB, and estimating M
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, klUCB).children,

    # # ---- RandTopM
    # RandTopM(NB_PLAYERS, nbArms, klUCB).children,
    # # # EstimateM(NB_PLAYERS, nbArms, RandTopM, klUCB).children,
    # # RandTopMEst(NB_PLAYERS, nbArms, klUCB).children,  # = EstimateM(... RandTopM, klUCB)
    # # RandTopMEstPlus(NB_PLAYERS, nbArms, klUCB, HORIZON).children,

    # # ---- MCTopM
    # MCTopM(NB_PLAYERS, nbArms, klUCB).children,
    # # # EstimateM(NB_PLAYERS, nbArms, MCTopM, klUCB).children,
    # # MCTopMEst(NB_PLAYERS, nbArms, klUCB).children,  # = EstimateM(... MCTopM, klUCB)
    # # MCTopMEstPlus(NB_PLAYERS, nbArms, klUCB, HORIZON).children,

    # ---- Selfish
    Selfish(NB_PLAYERS, nbArms, Exp3Decreasing).children,
    Selfish(NB_PLAYERS, nbArms, Exp3PlusPlus).children,
    Selfish(NB_PLAYERS, nbArms, UCB).children,
    Selfish(NB_PLAYERS, nbArms, klUCB).children,
    Selfish(NB_PLAYERS, nbArms, Thompson).children,

    # # ---- rhoRand etc
    # rhoRand(NB_PLAYERS, nbArms, klUCB).children,
    # # EstimateM(NB_PLAYERS, nbArms, rhoRand, klUCB).children,
    # rhoEst(NB_PLAYERS, nbArms, klUCB).children,  # = EstimateM(... rhoRand, klUCB)
    # # rhoEst(NB_PLAYERS, nbArms, klUCB, threshold=threshold_on_t).children,  # = EstimateM(... rhoRand, klUCB)
    # # EstimateM(NB_PLAYERS, nbArms, rhoRand, klUCB, horizon=HORIZON, threshold=threshold_on_t_with_horizon).children,  # = rhoEstPlus(...)
    # rhoEstPlus(NB_PLAYERS, nbArms, klUCB, HORIZON).children,
]


configuration.update({
    # --- DONE Using multi-player Selfish policy
    "players": Selfish(NB_PLAYERS, nbArms, Uniform).children
    # "players": Selfish(NB_PLAYERS, nbArms, TakeRandomFixedArm).children
    # "players": Selfish(NB_PLAYERS, nbArms, Exp3Decreasing).children
    # "players": Selfish(NB_PLAYERS, nbArms, Exp3WithHorizon, horizon=HORIZON).children
    # "players": Selfish(NB_PLAYERS, nbArms, UCB).children
    # "players": Selfish(NB_PLAYERS, nbArms, UCBalpha, alpha=0.25).children  # This one is efficient!
    # "players": Selfish(NB_PLAYERS, nbArms, MOSS).children
    # "players": Selfish(NB_PLAYERS, nbArms, klUCB).children
    # "players": Selfish(NB_PLAYERS, nbArms, klUCBPlus).children
    # "players": Selfish(NB_PLAYERS, nbArms, klUCBHPlus, horizon=HORIZON).children  # Worse than simple klUCB and klUCBPlus
    # "players": Selfish(NB_PLAYERS, nbArms, Thompson).children
    # "players": Selfish(NB_PLAYERS, nbArms, SoftmaxDecreasing).children
    # "players": Selfish(NB_PLAYERS, nbArms, BayesUCB).children
})

# DONE
print("Loaded experiments configuration from 'configuration.py' :")
print("configuration =", configuration)  # DEBUG
