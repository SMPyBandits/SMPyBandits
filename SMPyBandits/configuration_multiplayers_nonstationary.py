# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the piecewise stationary multi-players case.
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
    print("Warning: this script 'configuration_multiplayers.py' is NOT executable. Use 'main_multiplayers.py' or 'make multiplayers' or 'make moremultiplayers' ...")  # DEBUG
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
HORIZON = 1000
HORIZON = int(getenv('T', HORIZON))

#: REPETITIONS : number of repetitions of the experiments.
#: Warning: Should be >= 10 to be statistically trustworthy.
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
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
NB_PLAYERS = 1    # Less that the number of arms
NB_PLAYERS = 2    # Less that the number of arms
NB_PLAYERS = 3    # Less that the number of arms
# NB_PLAYERS = 6    # Less that the number of arms
# NB_PLAYERS = 25   # XXX More than the number of arms !!
# NB_PLAYERS = 30   # XXX More than the number of arms !!
NB_PLAYERS = int(getenv('M', NB_PLAYERS))
NB_PLAYERS = int(getenv('NB_PLAYERS', NB_PLAYERS))

# #: Different Collision models
# collisionModel = noCollision  #: Like single player.
# collisionModel = rewardIsSharedUniformly  #: Weird collision model.

# # Based on a distance of each user with the base station: the closer one wins if collision
# distances = 'uniform'  # Uniformly spaced objects
# distances = 'random'  # Let it compute the random distances, ONCE by thread, and then cache it? XXX
# distances = np.random.random_sample(NB_PLAYERS)  # Distance between 0 and 1, randomly affected!
# print("Each player is at the base station with a certain distance (the lower, the more chance it has to be selected)")  # DEBUG
# for i in range(NB_PLAYERS):
#     print("  - Player nb #{}\tis at distance {:.3g} to the Base Station ...".format(i + 1, distances[i]))  # DEBUG


# def onlyCloserUserGetsReward(t, arms, players, choices, rewards, pulls, collisions, distances=distances):
#     return closerUserGetsReward(t, arms, players, choices, rewards, pulls, collisions, distances=distances)


# collisionModel = onlyCloserUserGetsReward
# collisionModel.__doc__ = closerUserGetsReward.__doc__

#: The best collision model: none of the colliding users get any reward
collisionModel = onlyUniqUserGetsReward    # XXX this is the best one

# collisionModel = allGetRewardsAndUseCollision  #: DONE this is a bad collision model

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


NB_BREAK_POINTS = 1  #: Number of true breakpoints. They are uniformly spaced in time steps (and the first one at t=0 does not count).
NB_BREAK_POINTS = int(getenv('NB_BREAK_POINTS', NB_BREAK_POINTS))


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
    # --- Random events
    "nb_break_points": NB_BREAK_POINTS,
    # --- Should we plot the lower-bounds or not?
    "plot_lowerbounds": True,  # XXX Default
    # --- Arms
    "environment": [
        # # {   # A damn simple problem: 2 arms, one bad, one good
        # #     "arm_type": Bernoulli,
        # #     "params": [0.1, 0.9]  # uniformMeans(2, 0.1)
        # #     # "params": [0.9, 0.9]
        # #     # "params": [0.85, 0.9]
        # # }
        # # {   # XXX to test with 1 suboptimal arm only
        # #     "arm_type": Bernoulli,
        # #     "params": uniformMeans((NB_PLAYERS + 1), 1 / (1. + (NB_PLAYERS + 1)))
        # # }
        # # {   # XXX to test with half very bad arms, half perfect arms
        # #     "arm_type": Bernoulli,
        # #     "params": shuffled([0] * NB_PLAYERS) + ([1] * NB_PLAYERS)
        # # }
        # # {   # XXX To only test the orthogonalization (collision avoidance) protocol
        # #     "arm_type": Bernoulli,
        # #     "params": [1] * NB_PLAYERS
        # # }
        # # {   # XXX To only test the orthogonalization (collision avoidance) protocol
        # #     "arm_type": Bernoulli,
        # #     "params": [1] * NB_ARMS
        # # }
        # # {   # XXX To only test the orthogonalization (collision avoidance) protocol
        # #     "arm_type": Bernoulli,
        # #     "params": ([0] * (NB_ARMS - NB_PLAYERS)) + ([1] * NB_PLAYERS)
        # # }
        # # {   # An easy problem, but with a LOT of arms! (50 arms)
        # #     "arm_type": Bernoulli,
        # #     "params": uniformMeans(50, 1 / (1. + 50))
        # # }
        # # # XXX Default!
        # # {   # A very easy problem (X arms), but it is used in a lot of articles
        # #     "arm_type": ARM_TYPE,
        # #     "params": uniformMeans(NB_ARMS, 1 / (1. + NB_ARMS))
        # # },
        # {   # Use vector from command line
        #     "arm_type": ARM_TYPE,
        #     "params": MEANS
        # },
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
                    # "mingap": None,
                    # "mingap": 0.0000001,
                    # "mingap": 0.1,
                    "mingap": 1. / (3 * NB_ARMS),
                    "lower": LOWER,
                    "amplitude": AMPLITUDE,
                    "isSorted": True,
                }
            }
        },
    ]


# FIXME we cannot launch simulations on many problems in just one launch, because the oracle needs to know the change-point locations (and they change for some problems), and some algorithms need to know the number of arms for parameter selections?

PROBLEMS = [1]
STR_PROBLEMS = str(getenv('PROBLEMS', '1, 2')).replace(' ', '')
PROBLEMS = [int(p) for p in STR_PROBLEMS.split(',')]

# XXX Pb 0 changes are only on one arm at a time, only 2 arms
if 0 in PROBLEMS:  # WARNING remove this "False and" to use this problem
    configuration["environment"] += [
        {   # A simple piece-wise stationary problem
            "arm_type": ARM_TYPE,
            "params": {
                "listOfMeans": [
                    [0.1, 0.2],  # 0    to 399
                    [0.1, 0.3],  # 400  to 799
                    [0.5, 0.3],  # 800  to 1199
                    [0.4, 0.3],  # 1200 to 1599
                    [0.3, 0.9],  # 1600 to end
                ],
                "changePoints": [
                    0,
                    int((1 * HORIZON) / 5.0),
                    int((2 * HORIZON) / 5.0),
                    int((3 * HORIZON) / 5.0),
                    int((4 * HORIZON) / 5.0),
                    # 20000,  # XXX larger than horizon, just to see if it is a problem?
                ],
            }
        },
    ]

# XXX Pb 1 changes are only on one arm at a time
if 1 in PROBLEMS:  # WARNING remove this "False and" to use this problem
    configuration["environment"] += [
        {   # A simple piece-wise stationary problem
            "arm_type": ARM_TYPE,
            "params": {
                "listOfMeans": [
                    [0.3, 0.5, 0.9],  # 0    to 399
                    [0.3, 0.2, 0.9],  # 400  to 799
                    [0.3, 0.2, 0.1],  # 800  to 1199
                    [0.7, 0.2, 0.1],  # 1200 to 1599
                    [0.7, 0.5, 0.1],  # 1600 to end
                ],
                "changePoints": [
                    0,
                    int((1 * HORIZON) / 5.0),
                    int((2 * HORIZON) / 5.0),
                    int((3 * HORIZON) / 5.0),
                    int((4 * HORIZON) / 5.0),
                    # 20000,  # XXX larger than horizon, just to see if it is a problem?
                ],
            }
        },
    ]

# XXX Pb 2 changes are on all or almost arms at a time
if 2 in PROBLEMS:  # WARNING remove this "False and" to use this problem
    configuration["environment"] += [
        {   # A simple piece-wise stationary problem
            "arm_type": ARM_TYPE,
            "params": {
                "listOfMeans": [
                    [0.4, 0.5, 0.9],  # 0    to 399
                    [0.5, 0.4, 0.7],  # 400  to 799
                    [0.6, 0.3, 0.5],  # 800  to 1199
                    [0.7, 0.2, 0.3],  # 1200 to 1599
                    [0.8, 0.1, 0.1],  # 1600 to end
                ],
                "changePoints": [
                    0,
                    int((1 * HORIZON) / 5.0),
                    int((2 * HORIZON) / 5.0),
                    int((3 * HORIZON) / 5.0),
                    int((4 * HORIZON) / 5.0),
                    # 20000,  # XXX larger than horizon, just to see if it is a problem?
                ],
            }
        },
    ]

# XXX Pb 3 changes are on all or almost arms at a time, from https://subhojyoti.github.io/pdf/aistats_2019.pdf
if 3 in PROBLEMS:  # WARNING remove this "False and" to use this problem
    configuration["environment"] += [
        {   # A simple piece-wise stationary problem
            "arm_type": ARM_TYPE,
            "params": {
                "listOfMeans": [
                    [0.1, 0.2, 0.9],  # 0    to 999
                    [0.4, 0.9, 0.1],  # 1000 to 1999
                    [0.5, 0.1, 0.2],  # 2000 to 2999
                    [0.2, 0.2, 0.3],  # 3000 to end
                ],
                "changePoints": [
                    0,
                    int((1 * HORIZON) / 4.0),
                    int((2 * HORIZON) / 4.0),
                    int((3 * HORIZON) / 4.0),
                ],
            }
        },
    ]

# XXX Pb 4 changes are on all or almost arms at a time, but sequences don't have same length
if 4 in PROBLEMS:
    configuration["environment"] += [
        {   # A simple piece-wise stationary problem
            "arm_type": ARM_TYPE,
            "params": {
                "listOfMeans": [
                    [0.1, 0.5, 0.9],  # 1th sequence, best=3rd
                    [0.3, 0.4, 0.1],  # 2th sequence, best=2nd, DeltaMin=0.1
                    [0.5, 0.3, 0.2],  # 3th sequence, best=1st, DeltaMin=0.1
                    [0.7, 0.4, 0.3],  # 4th sequence, best=1st, DeltaMin=0.1
                    [0.1, 0.5, 0.2],  # 5th sequence, best=2nd, DeltaMin=0.1
                ],
                "changePoints": [
                    0,
                    int((4 * HORIZON) / 8.0),
                    int((5 * HORIZON) / 8.0),
                    int((6 * HORIZON) / 8.0),
                    int((7 * HORIZON) / 8.0),
                    # 20000,  # XXX larger than horizon, just to see if it is a problem?
                ],
            }
        },
    ]

# XXX Pb 5 Example from the Yahoo! dataset, from article "Nearly Optimal Adaptive Procedure with Change Detection for Piecewise-Stationary Bandit" (M-UCB) https://arxiv.org/abs/1802.03692
if 5 in PROBLEMS:  # WARNING remove this "False and" to use this problem
    configuration["environment"] = [
        {   # A very hard piece-wise stationary problem, with 6 arms and 9 change points
            "arm_type": ARM_TYPE,
            "params": {
                "listOfMeans": [
                    # red, green, blue, yellow, cyan, red dotted
                    [0.071, 0.041, 0.032, 0.030, 0.020, 0.011],  # 1st segment
                    [0.055, 0.053, 0.032, 0.030, 0.008, 0.011],  # 2nd segment
                    [0.040, 0.063, 0.032, 0.030, 0.008, 0.011],  # 3th segment
                    [0.040, 0.042, 0.043, 0.030, 0.008, 0.011],  # 4th segment
                    [0.030, 0.032, 0.055, 0.030, 0.008, 0.011],  # 5th segment
                    [0.030, 0.032, 0.020, 0.030, 0.008, 0.021],  # 6th segment
                    [0.020, 0.022, 0.020, 0.045, 0.008, 0.021],  # 7th segment
                    [0.020, 0.022, 0.020, 0.057, 0.008, 0.011],  # 8th segment
                    [0.020, 0.022, 0.034, 0.057, 0.022, 0.011],  # 9th segment
                ],
                "changePoints": np.linspace(0, HORIZON, num=9, endpoint=False, dtype=int),
            }
        },
    ]

# XXX Pb 6 Another example from the Yahoo! dataset, from article "On Abruptly-Changing and Slowly-Varying Multiarmed Bandit Problems" (SW-UCB#) https://arxiv.org/abs/1802.08380
if 6 in PROBLEMS:  # WARNING remove this "False and" to use this problem
    configuration["environment"] = [
        {   # A very hard piece-wise stationary problem, with 5 arms and 9 change points
            "arm_type": ARM_TYPE,
            "params": {
                "listOfMeans": [
                    # blue, red, golden, purple, green
                    [0.070, 0.044, 0.043, 0.029, 0.039],
                    [0.063, 0.044, 0.044, 0.029, 0.040],
                    [0.063, 0.045, 0.044, 0.028, 0.040],
                    [0.063, 0.045, 0.046, 0.028, 0.034],
                    [0.055, 0.045, 0.046, 0.028, 0.034],
                    [0.055, 0.049, 0.045, 0.024, 0.035],
                    [0.052, 0.049, 0.041, 0.024, 0.035],
                    [0.052, 0.048, 0.041, 0.020, 0.037],
                    [0.052, 0.048, 0.037, 0.020, 0.037],
                    [0.045, 0.050, 0.037, 0.020, 0.035],
                    [0.045, 0.050, 0.033, 0.018, 0.035],
                    [0.0455, 0.047, 0.033, 0.018, 0.035],
                    [0.0455, 0.047, 0.033, 0.018, 0.034],
                    [0.037, 0.042, 0.030, 0.020, 0.034],
                    [0.029, 0.032, 0.030, 0.020, 0.034],
                    [0.031, 0.026, 0.032, 0.020, 0.033],
                    [0.033, 0.026, 0.025, 0.020, 0.033],
                    [0.033, 0.035, 0.023, 0.020, 0.030],
                    [0.045, 0.038, 0.015, 0.020, 0.023],
                    [0.045, 0.038, 0.020, 0.014, 0.023],
                    [0.045, 0.038, 0.021, 0.014, 0.023],
                    [0.049, 0.042, 0.029, 0.014, 0.016],
                    [0.049, 0.042, 0.029, 0.016, 0.016],
                    [0.049, 0.042, 0.030, 0.014, 0.016],
                    [0.046, 0.040, 0.035, 0.020, 0.019],
                    [0.046, 0.040, 0.035, 0.020, 0.029],
                    [0.046, 0.040, 0.035, 0.023, 0.029],
                    [0.046, 0.037, 0.034, 0.023, 0.033],
                    [0.050, 0.037, 0.034, 0.024, 0.033],
                    [0.050, 0.040, 0.034, 0.024, 0.033],
                    [0.050, 0.040, 0.032, 0.024, 0.035],
                    [0.049, 0.040, 0.029, 0.0235, 0.035],
                    [0.049, 0.0405, 0.029, 0.0235, 0.037],
                    [0.047, 0.038, 0.0295, 0.025, 0.037],
                    [0.047, 0.038, 0.034, 0.025, 0.037],
                    [0.047, 0.041, 0.034, 0.025, 0.038],
                    [0.051, 0.041, 0.035, 0.025, 0.038],
                    [0.051, 0.040, 0.035, 0.025, 0.038],
                    [0.051, 0.038, 0.033, 0.025, 0.039],
                    [0.047, 0.038, 0.033, 0.026, 0.039],
                    [0.047, 0.035, 0.032, 0.026, 0.039],
                    [0.045, 0.033, 0.032, 0.024, 0.038],
                    [0.045, 0.030, 0.031, 0.024, 0.038],
                    [0.045, 0.027, 0.031, 0.024, 0.038],
                    [0.043, 0.027, 0.026, 0.021, 0.0375],
                    [0.043, 0.030, 0.026, 0.021, 0.0375],
                    [0.043, 0.030, 0.026, 0.021, 0.0375],
                    [0.043, 0.034, 0.025, 0.021, 0.0375],
                    [0.045, 0.034, 0.015, 0.020, 0.0375],
                    [0.045, 0.033, 0.016, 0.020, 0.036],
                    [0.043, 0.033, 0.020, 0.018, 0.036],
                    [0.043, 0.035, 0.020, 0.018, 0.032],
                    [0.043, 0.035, 0.027, 0.018, 0.032],
                    [0.040, 0.035, 0.027, 0.018, 0.032],
                    [0.033, 0.036, 0.029, 0.019, 0.033],
                    [0.028, 0.036, 0.029, 0.019, 0.033],
                    [0.028, 0.038, 0.029, 0.017, 0.033],
                    [0.032, 0.038, 0.034, 0.017, 0.030],
                    [0.031, 0.038, 0.034, 0.015, 0.030],
                    [0.031, 0.040, 0.034, 0.015, 0.030],
                    [0.038, 0.040, 0.034, 0.014, 0.029],
                    [0.038, 0.038, 0.034, 0.012, 0.026],
                    [0.042, 0.038, 0.034, 0.018, 0.026],
                    [0.042, 0.037, 0.034, 0.018, 0.019],
                    [0.042, 0.037, 0.034, 0.018, 0.0185],
                    [0.043, 0.037, 0.034, 0.023, 0.017],
                    [0.044, 0.038, 0.036, 0.023, 0.024],
                    [0.044, 0.038, 0.036, 0.023, 0.029],
                    [0.044, 0.038, 0.036, 0.025, 0.029],
                    [0.044, 0.037, 0.034, 0.025, 0.034],
                    [0.044, 0.035, 0.034, 0.028, 0.034],
                    [0.044, 0.035, 0.034, 0.028, 0.037],
                    [0.049, 0.035, 0.034, 0.028, 0.037],
                    [0.048, 0.032, 0.037, 0.028, 0.037],
                    [0.048, 0.032, 0.037, 0.027, 0.037],
                    [0.047, 0.029, 0.037, 0.027, 0.038],
                    [0.047, 0.027, 0.039, 0.027, 0.038],
                    [0.047, 0.023, 0.039, 0.030, 0.039],
                    [0.049, 0.022, 0.035, 0.030, 0.039],
                    [0.049, 0.031, 0.035, 0.030, 0.039],
                    [0.049, 0.031, 0.035, 0.027, 0.039],
                    [0.049, 0.032, 0.033, 0.027, 0.039],
                ],
                "changePoints": np.linspace(0, HORIZON, num=82, endpoint=False, dtype=int),
            }
        },
    ]


CHANGE_POINTS = configuration["environment"][0]["params"]["changePoints"]
LIST_OF_MEANS = configuration["environment"][0]["params"]["listOfMeans"]
# CHANGE_POINTS = np.unique(np.array(list(set.union(*(set(env["params"]["changePoints"]) for env in ENVIRONMENT)))))

NB_BREAK_POINTS = max([len(env["params"]["changePoints"]) - (1 if 0 in env["params"]["changePoints"] else 0) for env in configuration["environment"]])
configuration["nb_break_points"] = NB_BREAK_POINTS


try:
    #: Number of arms *in the first environment*
    nbArms = int(configuration["environment"][0]["params"]["args"]["nbArms"])
except (TypeError, KeyError):
    try:
        nbArms = len(configuration["environment"][0]["params"]["listOfMeans"][0])
    except (TypeError, KeyError):
        nbArms = len(configuration["environment"][0]["params"])


if len(configuration['environment']) > 1:
    print("WARNING do not use this hack if you try to use more than one environment.")


#: Warning: if using Exponential or Gaussian arms, gives klExp or klGauss to KL-UCB-like policies!
klucb = klucb_mapping.get(str(configuration["environment"][0]["arm_type"]), klucbBern)


# XXX compare different values of the experimental sliding window algorithm
EPSS   = [0.05]  #+ [0.1]
ALPHAS = [1]
TAUS   = [
        # 500, 1000, 2000,
        int(2 * np.sqrt(HORIZON * np.log(HORIZON) / max(1, NB_BREAK_POINTS))),  # "optimal" value according to [Garivier & Moulines, 2008]
    ]
GAMMAS = [0.75]  #+ [0.9999, 0.99, 0.75, 0.5]
GAMMA_T_UpsilonT = 1 - np.sqrt(NB_BREAK_POINTS / HORIZON) / 4.
# GAMMAS = [GAMMA_T_UpsilonT]

WINDOW_SIZE = NB_ARMS * int(np.ceil(HORIZON / 100))  #: Default window size :math:`w` for the M-UCB and SW-UCB algorithm.
# WINDOW_SIZE = 400  # FIXME manually set...

PER_ARM_RESTART = [
    True,  # Per-arm restart XXX comment to only test global arm
    # False, # Global restart XXX seems more efficient? (at least more memory efficient!)
]

MIN_NUMBER_OF_OBSERVATION_BETWEEN_CHANGE_POINT = np.min(np.diff(CHANGE_POINTS)) // (2 * NB_ARMS)

UPSILON_T = max(1, NB_BREAK_POINTS)

NUMBER_OF_CHANGE_POINTS = NB_ARMS * UPSILON_T
if len(PROBLEMS) == 1 and set(PROBLEMS) <= {1,2,3,4,5,6}:
    CT = sum([sum(np.diff(np.array(LIST_OF_MEANS)[:, i]) != 0) for i in range(np.shape(LIST_OF_MEANS)[1])])
    NUMBER_OF_CHANGE_POINTS = CT
print("\nUsing Upsilon_T = {} break-points (time when at least one arm changes), and C_T = {} change-points (number of changes of all arms).".format(UPSILON_T, NUMBER_OF_CHANGE_POINTS))  # DEBUG

DELTA_for_MUCB = 0.1
EPSILON_for_CUSUM = 0.1
if len(PROBLEMS) == 1: # and set(PROBLEMS) <= {1,2,3,4,5,6}:
    print("For this problem, we compute the Delta^change and Delta^opt...")  # DEBUG
    min_change_on_mean = min(delta for delta in [min([delta for delta in np.abs(np.diff(np.array(LIST_OF_MEANS)[:, i])) if delta > 0 ]) for i in range(np.shape(LIST_OF_MEANS)[1])] if delta > 0)
    print("min_change_on_mean =", min_change_on_mean)  # DEBUG
    min_optimality_gap = min(delta for delta in [min([delta for delta in np.abs(np.diff(np.array(LIST_OF_MEANS)[j, :])) if delta > 0 ]) for j in range(np.shape(LIST_OF_MEANS)[0])] if delta > 0)
    print("min_optimality_gap =", min_optimality_gap)  # DEBUG
    # DELTA_for_MUCB = min_change_on_mean
    # EPSILON_for_CUSUM = min_change_on_mean
print("DELTA_for_MUCB =", DELTA_for_MUCB)  # DEBUG
print("EPSILON_for_CUSUM =", EPSILON_for_CUSUM)  # DEBUG

DELTA_T = 1.0 / np.sqrt(HORIZON)  # XXX tune the delta as a function of T
DELTA_T_UpsilonT = 1.0 / np.sqrt(UPSILON_T * HORIZON)  # XXX tune the delta as just a function of T and Upsilon_T
DELTA_T_UpsilonT_K = 1.0 / np.sqrt(NB_ARMS * UPSILON_T * HORIZON)  # XXX tune the delta as just a function of T and Upsilon_T
DELTA_T_CT = 1.0 / np.sqrt(NUMBER_OF_CHANGE_POINTS * HORIZON)  # XXX tune the delta as just a function of T and Upsilon_T

DELTA_GLOBAL = DELTA_T_UpsilonT
DELTA_LOCAL = DELTA_T_UpsilonT_K

# ALPHA_0 = 1
ALPHA_0 = 0.05
# ALPHA_0 = 0

ALPHA_T = ALPHA_0 * np.sqrt(np.log(HORIZON) / HORIZON)  # XXX tune the alpha as a function of T
ALPHA_T_UpsilonT = ALPHA_0 * np.sqrt(UPSILON_T * np.log(HORIZON) / HORIZON)  # XXX tune the alpha as just a function of T and Upsilon_T
ALPHA_T_UpsilonT_K = ALPHA_0 * np.sqrt(NB_ARMS * UPSILON_T * np.log(HORIZON) / HORIZON)  # XXX tune the alpha as just a function of T and Upsilon_T
ALPHA_T_CT = ALPHA_0 * np.sqrt(NUMBER_OF_CHANGE_POINTS * np.log(HORIZON) / HORIZON)  # XXX tune the alpha as just a function of T and Upsilon_T

ALPHA_GLOBAL = ALPHA_T_UpsilonT
ALPHA_LOCAL = ALPHA_T_UpsilonT_K

#: Compute the gap of the first problem.
#: (for d in MEGA's parameters, and epsilon for MusicalChair's parameters)
try:
    GAP = np.min(np.diff(np.sort(configuration['environment'][0]['params'])))
except (ValueError, np.AxisError):
    print("Warning: using the default value for the GAP (Bayesian environment maybe?)")  # DEBUG
    GAP = 1. / (3 * NB_ARMS)


configuration["successive_players"] = [
    # ---- rhoRand etc
    # rhoRand(NB_PLAYERS, nbArms, UCB).children,
    rhoRand(NB_PLAYERS, nbArms, klUCB).children,
    rhoRand(NB_PLAYERS, nbArms, BernoulliGLR_IndexPolicy_WithDeterministicExploration, policy=klUCB_forGLR, per_arm_restart=True, delta=DELTA_LOCAL, alpha0=ALPHA_LOCAL, lazy_detect_change_only_x_steps=20, lazy_try_value_s_only_x_steps=20).children,
    # rhoRand(NB_PLAYERS, nbArms, BernoulliGLR_IndexPolicy_WithDeterministicExploration, policy=klUCB_forGLR, per_arm_restart=False, delta=DELTA_GLOBAL, alpha0=ALPHA_GLOBAL, lazy_detect_change_only_x_steps=20, lazy_try_value_s_only_x_steps=20).children,

    # # ---- RandTopM
    # RandTopM(NB_PLAYERS, nbArms, UCB).children,
    RandTopM(NB_PLAYERS, nbArms, klUCB).children,
    RandTopM(NB_PLAYERS, nbArms, BernoulliGLR_IndexPolicy_WithDeterministicExploration, policy=klUCB_forGLR, per_arm_restart=True, delta=DELTA_LOCAL, alpha0=ALPHA_LOCAL, lazy_detect_change_only_x_steps=20, lazy_try_value_s_only_x_steps=20).children,
    # RandTopM(NB_PLAYERS, nbArms, BernoulliGLR_IndexPolicy_WithDeterministicExploration, policy=klUCB_forGLR, per_arm_restart=False, delta=DELTA_GLOBAL, alpha0=ALPHA_GLOBAL, lazy_detect_change_only_x_steps=20, lazy_try_value_s_only_x_steps=20).children,

    # ---- MCTopM
    # MCTopM(NB_PLAYERS, nbArms, UCB).children,
    MCTopM(NB_PLAYERS, nbArms, klUCB).children,
    MCTopM(NB_PLAYERS, nbArms, BernoulliGLR_IndexPolicy_WithDeterministicExploration, policy=klUCB_forGLR, per_arm_restart=True, delta=DELTA_LOCAL, alpha0=ALPHA_LOCAL, lazy_detect_change_only_x_steps=20, lazy_try_value_s_only_x_steps=20).children,
    # MCTopM(NB_PLAYERS, nbArms, BernoulliGLR_IndexPolicy_WithDeterministicExploration, policy=klUCB_forGLR, per_arm_restart=False, delta=DELTA_GLOBAL, alpha0=ALPHA_GLOBAL, lazy_detect_change_only_x_steps=20, lazy_try_value_s_only_x_steps=20).children,

    # ---- Selfish
    Selfish(NB_PLAYERS, nbArms, Thompson).children,
    # Selfish(NB_PLAYERS, nbArms, UCB).children,
    Selfish(NB_PLAYERS, nbArms, klUCB).children,

    # ---- TODO Selfish for algorithms specialized for non-stationary settings
    Selfish(NB_PLAYERS, nbArms, OracleSequentiallyRestartPolicy, changePoints=CHANGE_POINTS, listOfMeans=LIST_OF_MEANS, policy=klUCB, reset_for_all_change=True, reset_for_suboptimal_change=False).children,
    Selfish(NB_PLAYERS, nbArms, DiscountedThompson, gamma=0.99).children,
    Selfish(NB_PLAYERS, nbArms, Monitored_IndexPolicy, horizon=HORIZON, w=WINDOW_SIZE, delta=DELTA_for_MUCB, policy=klUCB).children,
    Selfish(NB_PLAYERS, nbArms, CUSUM_IndexPolicy, horizon=HORIZON, max_nb_random_events=NB_BREAK_POINTS, epsilon=EPSILON_for_CUSUM, policy=klUCB, lazy_detect_change_only_x_steps=20).children,
    Selfish(NB_PLAYERS, nbArms, BernoulliGLR_IndexPolicy_WithDeterministicExploration, policy=klUCB_forGLR, per_arm_restart=True, delta=DELTA_LOCAL, alpha0=ALPHA_LOCAL, lazy_detect_change_only_x_steps=20, lazy_try_value_s_only_x_steps=20).children,
    # Selfish(NB_PLAYERS, nbArms, BernoulliGLR_IndexPolicy_WithDeterministicExploration, policy=klUCB_forGLR, per_arm_restart=False, delta=DELTA_GLOBAL, alpha0=ALPHA_GLOBAL, lazy_detect_change_only_x_steps=20, lazy_try_value_s_only_x_steps=20).children,

    # --- FIXME MusicalChairNoSensing (selfish), a better Musical Chair
    # [ MusicalChairNoSensing(nbPlayers=NB_PLAYERS, nbArms=nbArms, horizon=HORIZON) for _ in range(NB_PLAYERS) ],

    # ---- Centralized multiple play
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, UCB).children,
    CentralizedMultiplePlay(NB_PLAYERS, nbArms, klUCB).children,

    # ---- TODO Selfish for algorithms specialized for non-stationary settings
    CentralizedMultiplePlay(NB_PLAYERS, nbArms, OracleSequentiallyRestartPolicy, changePoints=CHANGE_POINTS, listOfMeans=LIST_OF_MEANS, policy=klUCB, reset_for_all_change=True, reset_for_suboptimal_change=False).children,
    CentralizedMultiplePlay(NB_PLAYERS, nbArms, DiscountedThompson, gamma=0.99).children,
    CentralizedMultiplePlay(NB_PLAYERS, nbArms, Monitored_IndexPolicy, horizon=HORIZON, w=WINDOW_SIZE, delta=DELTA_for_MUCB, policy=klUCB).children,
    CentralizedMultiplePlay(NB_PLAYERS, nbArms, CUSUM_IndexPolicy, horizon=HORIZON, max_nb_random_events=NB_BREAK_POINTS, epsilon=EPSILON_for_CUSUM, policy=klUCB, lazy_detect_change_only_x_steps=20).children,
    CentralizedMultiplePlay(NB_PLAYERS, nbArms, BernoulliGLR_IndexPolicy_WithDeterministicExploration, policy=klUCB_forGLR, per_arm_restart=True, delta=DELTA_LOCAL, alpha0=ALPHA_LOCAL, lazy_detect_change_only_x_steps=20, lazy_try_value_s_only_x_steps=20).children,
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, BernoulliGLR_IndexPolicy_WithDeterministicExploration, policy=klUCB_forGLR, per_arm_restart=False, delta=DELTA_GLOBAL, alpha0=ALPHA_GLOBAL, lazy_detect_change_only_x_steps=20, lazy_try_value_s_only_x_steps=20).children,

    # # FIXME how to chose the 5 parameters for MEGA policy ?
    # # XXX By trial and error??
    # # d should be smaller than the gap Delta = mu_M* - mu_(M-1)* (gap between Mbest and Mworst)
    # [ MEGA(nbArms, p0=0.1, alpha=0.1, beta=0.5, c=0.1, d=0.99*GAP) for _ in range(NB_PLAYERS) ],  # XXX always linear regret!

    # # XXX stupid version with fixed T0 : cannot adapt to any problem
    # [ MusicalChair(nbArms, Time0=1000) for _ in range(NB_PLAYERS) ],
    [ MusicalChair(nbArms, Time0=100*NB_ARMS) for _ in range(NB_PLAYERS) ],
    [ MusicalChair(nbArms, Time0=150*NB_ARMS) for _ in range(NB_PLAYERS) ],
    [ MusicalChair(nbArms, Time0=250*NB_ARMS) for _ in range(NB_PLAYERS) ],
    # # # XXX cheated version, with known gap (epsilon < Delta) and proba of success 5% !
    # [ MusicalChair(nbArms, Time0=optimalT0(nbArms=NB_ARMS, epsilon=0.99*GAP, delta=0.5)) for _ in range(NB_PLAYERS) ],
    # [ MusicalChair(nbArms, Time0=optimalT0(nbArms=NB_ARMS, epsilon=0.99*GAP, delta=0.1)) for _ in range(NB_PLAYERS) ],
    # # # XXX cheated version, with known gap and known horizon (proba of success delta < 1 / T) !
    # [ MusicalChair(nbArms, Time0=optimalT0(nbArms=NB_ARMS, epsilon=0.99*GAP, delta=1./(1+HORIZON))) for _ in range(NB_PLAYERS) ],

    # DONE test this new SIC_MMAB algorithm
    # [ SIC_MMAB(nbArms, HORIZON) for _ in range(NB_PLAYERS) ],
    # [ SIC_MMAB_UCB(nbArms, HORIZON) for _ in range(NB_PLAYERS) ],
    [ SIC_MMAB_klUCB(nbArms, HORIZON) for _ in range(NB_PLAYERS) ],

    # # XXX stupid version with fixed T0 : cannot adapt to any problem
    # [ TrekkingTSN(nbArms, theta=0.1, epsilon=0.1, delta=0.1) for _ in range(NB_PLAYERS) ],
    # # DONE test this new TrekkingTSN algorithm!
]


configuration.update({
    # --- DONE Defining manually each child
    # "players": [TakeFixedArm(nbArms, nbArms - 1) for _ in range(NB_PLAYERS)]
    # "players": [TakeRandomFixedArm(nbArms) for _ in range(NB_PLAYERS)]

    # --- Defining each player as one child of a multi-player policy

    # --- DONE Using multi-player dummy Centralized policy
    # XXX each player needs to now the number of players
    # "players": CentralizedFixed(NB_PLAYERS, nbArms).children
    # "players": CentralizedCycling(NB_PLAYERS, nbArms).children
    # --- DONE Using a smart Centralized policy, based on choiceMultiple()
    # "players": CentralizedMultiplePlay(NB_PLAYERS, nbArms, UCB, uniformAllocation=False).children
    # "players": CentralizedMultiplePlay(NB_PLAYERS, nbArms, UCB, uniformAllocation=True).children
    # "players": CentralizedMultiplePlay(NB_PLAYERS, nbArms, Thompson, uniformAllocation=False).children
    # "players": CentralizedMultiplePlay(NB_PLAYERS, nbArms, Thompson, uniformAllocation=True).children

    # --- DONE Using a smart Centralized policy, based on choiceIMP() -- It's not better, in fact
    # "players": CentralizedIMP(NB_PLAYERS, nbArms, UCB, uniformAllocation=False).children
    # "players": CentralizedIMP(NB_PLAYERS, nbArms, UCB, uniformAllocation=True).children
    # "players": CentralizedIMP(NB_PLAYERS, nbArms, Thompson, uniformAllocation=False).children
    # "players": CentralizedIMP(NB_PLAYERS, nbArms, Thompson, uniformAllocation=True).children

    # --- DONE Using multi-player Selfish policy
    # "players": Selfish(NB_PLAYERS, nbArms, Uniform).children
    # "players": Selfish(NB_PLAYERS, nbArms, TakeRandomFixedArm).children
    # "players": Selfish(NB_PLAYERS, nbArms, Exp3Decreasing).children
    # "players": Selfish(NB_PLAYERS, nbArms, Exp3WithHorizon, horizon=HORIZON).children
    # "players": Selfish(NB_PLAYERS, nbArms, UCB).children
    # "players": Selfish(NB_PLAYERS, nbArms, DiscountedUCB).children
    # "players": Selfish(NB_PLAYERS, nbArms, Thompson).children
    # "players": Selfish(NB_PLAYERS, nbArms, DiscountedThompson, gamma=0.99).children
    "players": Selfish(NB_PLAYERS, nbArms, BernoulliGLR_IndexPolicy_WithDeterministicExploration).children

    # --- XXX play with SIC_MMAB
    # "players": [ SIC_MMAB(nbArms, HORIZON) for _ in range(NB_PLAYERS) ]
})
# TODO the EvaluatorMultiPlayers should regenerate the list of players in every repetitions, to have at the end results on the average behavior of these randomized multi-players policies


# DONE
print("Loaded experiments configuration from 'configuration_multiplayers_nonstationary.py' :")
print("configuration =", configuration)  # DEBUG
