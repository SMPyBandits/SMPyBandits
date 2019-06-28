# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the piecewise stationary single-player case.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

# Tries to know number of CPU
try:
    from multiprocessing import cpu_count
    CPU_COUNT = cpu_count()  #: Number of CPU on the local machine
except ImportError:
    CPU_COUNT = 1

from os import getenv

if __name__ == '__main__':
    print("Warning: this script 'configuration_nonstationary.py' is NOT executable. Use 'main.py' or 'make single' ...")  # DEBUG
    exit(0)

# Import arms and algorithms
try:
    from Arms import *
    from Policies import *
except ImportError:
    from SMPyBandits.Arms import *
    from SMPyBandits.Policies import *

#: HORIZON : number of time steps of the experiments.
#: Warning Should be >= 10000 to be interesting "asymptotically".
HORIZON = 5000
HORIZON = int(getenv('T', HORIZON))

#: REPETITIONS : number of repetitions of the experiments.
#: Warning: Should be >= 10 to be statistically trustworthy.
REPETITIONS = 100
REPETITIONS = int(getenv('N', REPETITIONS))

#: To profile the code, turn down parallel computing
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1 or REPETITIONS == -1) and DO_PARALLEL

#: Number of jobs to use for the parallel computations. -1 means all the CPU cores, 1 means no parallelization.
N_JOBS = -1 if DO_PARALLEL else 1
if CPU_COUNT > 4:  # We are on a server, let's be nice and not use all cores
    N_JOBS = min(CPU_COUNT, max(int(CPU_COUNT / 3), CPU_COUNT - 8))
N_JOBS = int(getenv('N_JOBS', N_JOBS))
if REPETITIONS == -1:
    REPETITIONS = max(N_JOBS, CPU_COUNT)


NB_BREAK_POINTS = 5  #: Number of true breakpoints. They are uniformly spaced in time steps (and the first one at t=0 does not count).
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
    # --- Random events
    "nb_break_points": NB_BREAK_POINTS,
    # --- Should we plot the lower-bounds or not?
    "plot_lowerbound": False,  # XXX Default for non stationary: we do not have a better lower bound than Lai & Robbins's.
    # --- Arms
    "environment": [],
}


#: Number of arms for non-hard-coded problems (Bayesian problems)
NB_ARMS = 3
NB_ARMS = int(getenv('K', NB_ARMS))
NB_ARMS = int(getenv('NB_ARMS', NB_ARMS))

#: Default value for the lower value of means
LOWER = 0.
#: Default value for the amplitude value of means
AMPLITUDE = 1.
#: Variance of Gaussian arms, if needed
VARIANCE = 0.25

#: Type of arms for non-hard-coded problems (Bayesian problems)
ARM_TYPE = "Bernoulli"
ARM_TYPE = str(getenv('ARM_TYPE', ARM_TYPE))
ARM_TYPE_str = str(ARM_TYPE)
ARM_TYPE = mapping_ARM_TYPE[ARM_TYPE]

#: Means of arms for non-hard-coded problems (non Bayesian)
MEANS = uniformMeans(nbArms=NB_ARMS, delta=0.05, lower=LOWER, amplitude=AMPLITUDE, isSorted=True)

# FIXME we cannot launch simulations on many problems in just one launch, because the oracle needs to know the change-point locations (and they change for some problems), and some algorithms need to know the number of arms for parameter selections?

PROBLEMS = [1, 2]
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
                    int(0    * HORIZON / 2000.0),
                    int(400  * HORIZON / 2000.0),
                    int(800  * HORIZON / 2000.0),
                    int(1200 * HORIZON / 2000.0),
                    int(1600 * HORIZON / 2000.0),
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
                    int(0    * HORIZON / 2000.0),
                    int(400  * HORIZON / 2000.0),
                    int(800  * HORIZON / 2000.0),
                    int(1200 * HORIZON / 2000.0),
                    int(1600 * HORIZON / 2000.0),
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
                    int(0    * HORIZON / 2000.0),
                    int(400  * HORIZON / 2000.0),
                    int(800  * HORIZON / 2000.0),
                    int(1200 * HORIZON / 2000.0),
                    int(1600 * HORIZON / 2000.0),
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
                    int(0    * HORIZON / 4000.0),
                    int(1000 * HORIZON / 4000.0),
                    int(2000 * HORIZON / 4000.0),
                    int(3000 * HORIZON / 4000.0),
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
                    int(0    * HORIZON / 2000.0),
                    int(1000 * HORIZON / 2000.0),
                    int(1250 * HORIZON / 2000.0),
                    int(1500 * HORIZON / 2000.0),
                    int(1750 * HORIZON / 2000.0),
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
                "listOfMeans": np.array([
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
                ]) * 1,   # FIXME change *10 if you want to increase gaps?
                "changePoints": np.linspace(0, HORIZON, num=82, endpoint=False, dtype=int),
            }
        },
    ]


# FIXME experimental code to check some condition on the problems

def lowerbound_on_sequence_length(horizon, gap):
    r""" A function that computes the lower-bound (we will find) on the sequence length to have a reasonable bound on the delay of our change-detection algorithm.

    - It returns the smallest possible sequence length :math:`L = \tau_{m+1} - \tau_m` satisfying:

    .. math:: L \geq \frac{8}{\Delta^2} \log(T).
    """
    if np.isclose(gap, 0): return 0
    condition = lambda length: length >= (8/gap**2) * np.log(horizon)
    length = 1
    while not condition(length):
        length += 1
    return length


def check_condition_on_piecewise_stationary_problems(horizon, listOfMeans, changePoints):
    """ Check some conditions on the piecewise stationary problem."""
    M = len(listOfMeans)
    print("For a piecewise stationary problem with M = {} sequences...".format(M))  # DEBUG
    for m in range(M - 1):
        mus_m = listOfMeans[m]
        tau_m = changePoints[m]
        mus_mp1 = listOfMeans[m+1]
        tau_mp1 = changePoints[m+1]
        print("\nChecking m-th (m = {}) sequence, µ_m = {}, µ_m+1 = {} and tau_m = {} and tau_m+1 = {}".format(m, mus_m, mus_mp1, tau_m, tau_mp1))  # DEBUG
        for i, (mu_i_m, mu_i_mp1) in enumerate(zip(mus_m, mus_mp1)):
            gap = abs(mu_i_m - mu_i_mp1)
            length = tau_mp1 - tau_m
            lowerbound = lowerbound_on_sequence_length(horizon, gap)
            print("   - For arm i = {}, gap = {:.3g} and length = {} with lowerbound on length = {}...".format(i, gap, length, lowerbound))  # DEBUG
            if length < lowerbound:
                print("WARNING For arm i = {}, gap = {:.3g} and length = {} < lowerbound on length = {} !!".format(i, gap, length, lowerbound))  # DEBUG


# for envId, env in enumerate(configuration["environment"]):
#     print("\n\n\nChecking environment number {}".format(envId))  # DEBUG
#     listOfMeans = env["params"]["listOfMeans"]
#     changePoints = env["params"]["changePoints"]
#     check_condition_on_piecewise_stationary_problems(HORIZON, listOfMeans, changePoints)


CHANGE_POINTS = configuration["environment"][0]["params"]["changePoints"]
LIST_OF_MEANS = configuration["environment"][0]["params"]["listOfMeans"]
# CHANGE_POINTS = np.unique(np.array(list(set.union(*(set(env["params"]["changePoints"]) for env in ENVIRONMENT)))))

NB_BREAK_POINTS = max([len(env["params"]["changePoints"]) - (1 if 0 in env["params"]["changePoints"] else 0) for env in configuration["environment"]])
configuration["nb_break_points"] = NB_BREAK_POINTS


# if False:  # WARNING remove this "False and" to use this problem
#     configuration["environment"] = [
#         {   # A non stationary problem: every step of the same repetition use a different mean vector!
#             "arm_type": ARM_TYPE,
#             "params": {
#                 "newMeans": randomMeans,
#                 # XXX Note that even using geometricChangePoints does not mean random change points *at each repetitions*
#                 # "changePoints": geometricChangePoints(horizon=HORIZON, proba=NB_BREAK_POINTS/HORIZON),
#                 "changePoints": np.linspace(0, HORIZON, num=NB_BREAK_POINTS, dtype=int, endpoint=False),
#                 "args": {
#                     "nbArms": NB_ARMS,
#                     "lower": LOWER, "amplitude": AMPLITUDE,
#                     "mingap": None, "isSorted": False,
#                 },
#                 # XXX onlyOneArm is None by default,
#                 "onlyOneArm": None,
#                 # XXX but onlyOneArm can be "uniform" to only change *one* arm at each change point,
#                 # "onlyOneArm": "uniform",
#                 # XXX onlyOneArm can also be an integer to only change n arms at each change point,
#                 # "onlyOneArm": 3,
#             }
#         },
#     ]

# if False:  # WARNING remove this "False and" to use this problem
#     configuration["environment"] = [  # XXX Bernoulli arms
#         {   # A non stationary problem: every step of the same repetition use a different mean vector!
#             "arm_type": ARM_TYPE,
#             "params": {
#                 "newMeans": continuouslyVaryingMeans,
#                 "changePoints": np.linspace(0, HORIZON, num=NB_BREAK_POINTS, dtype=int),
#                 "args": {
#                    "nbArms": NB_ARMS,
#                    "maxSlowChange": 0.1, "sign": +1,
#                    "mingap": None, "isSorted": False,
#                    "lower": LOWER, "amplitude": AMPLITUDE,
#                 }
#             }
#         },
#     ]


# if False:  # WARNING remove this "False and" to use this problem
#     configuration["environment"] = [  # XXX Bernoulli arms
#         {   # A non stationary problem: every step of the same repetition use a different mean vector!
#             "arm_type": ARM_TYPE,
#             "params": {
#                 "newMeans": randomContinuouslyVaryingMeans,
#                 "changePoints": np.linspace(0, HORIZON, num=NB_BREAK_POINTS, dtype=int),
#                 "args": {
#                     "nbArms": NB_ARMS,
#                     "maxSlowChange": 0.1, "horizon": HORIZON,
#                     "mingap": None, "isSorted": False,
#                     "lower": LOWER, "amplitude": AMPLITUDE,
#                 }
#             }
#         },
#     ]

try:
    #: Number of arms *in the first environment*
    nbArms = int(configuration["environment"][0]["params"]["args"]["nbArms"])
except (TypeError, KeyError):
    try:
        nbArms = len(configuration["environment"][0]["params"]["listOfMeans"][0])
    except (TypeError, KeyError):
        nbArms = len(configuration["environment"][0]["params"])

#: Warning: if using Exponential or Gaussian arms, gives klExp or klGauss to KL-UCB-like policies!
klucb = klucb_mapping.get(str(configuration["environment"][0]["arm_type"]), klucbBern)


# XXX compare different values of the experimental sliding window algorithm
EPSS   = [0.05]  #+ [0.1]
ALPHAS = [1]
TAUS   = [
        # 500, 1000, 2000,
        int(2 * np.sqrt(HORIZON * np.log(HORIZON) / max(1, NB_BREAK_POINTS))),  # "optimal" value according to [Garivier & Moulines, 2008]
    ]
GAMMAS = [0.95]  #+ [0.9999, 0.99, 0.75, 0.5]
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

configuration.update({
    "policies":
    # # [  # XXX Regular adversarial bandits algorithms!
    # #     # { "archtype": Exp3WithHorizon, "params": { "horizon": HORIZON, } },
    # #     { "archtype": Exp3PlusPlus, "params": {} },
    # # ] +
    # # [
    # #     # --- TsallisInf algorithm
    # #     {
    # #         "archtype": TsallisInf,
    # #         "params": {
    # #             "alpha": 0.5,
    # #         }
    # #     },
    # #     # --- RCB algorithm
    # #     {
    # #         "archtype": RCB,
    # #         "params": {
    # #             "alpha": 1.,
    # #             "perturbation": "uniform",  # like UCB?
    # #         }
    # #     },
    # #     # --- PHE algorithm
    # #     {
    # #         "archtype": PHE,
    # #         "params": {
    # #             "perturbation_scale": 1.1,
    # #         }
    # #     },
    # # ] +
    [  # XXX Regular stochastic bandits algorithms!
        # # { "archtype": Uniform, "params": { } },
        # # { "archtype": EmpiricalMeans, "params": { } },
        # # { "archtype": UCBalpha, "params": { "alpha": 1, } },
        # { "archtype": UCB, "params": { } },
        # # { "archtype": SWR_UCBalpha, "params": { "alpha": 1, } },  # WARNING experimental!
        # { "archtype": BESA, "params": { "horizon": HORIZON, "non_binary": True, } },
        # { "archtype": BayesUCB, "params": { "posterior": Beta, } },
        # { "archtype": AdBandits, "params": { "alpha": 1, "horizon": HORIZON, } },
        { "archtype": klUCB, "params": { "klucb": klucb, }, "change_label": "klUCB", },
        # { "archtype": SWR_klUCB, "params": { "klucb": klucb, } },  # WARNING experimental!
        { "archtype": Thompson, "params": { "posterior": Beta, }, "change_label": "Thompson Sampling" },
    ] +
    # DONE the OracleSequentiallyRestartPolicy with klUCB/UCB policy works quite well, but NOT optimally!
    [
        { "archtype": OracleSequentiallyRestartPolicy, "params": {
            "changePoints": CHANGE_POINTS,
            "listOfMeans": LIST_OF_MEANS,
            "policy": policy,
            # "per_arm_restart": per_arm_restart,
            "reset_for_all_change": reset_for_all_change,
            "reset_for_suboptimal_change": reset_for_suboptimal_change,
            # "full_restart_when_refresh": full_restart_when_refresh,
        } }
        for policy in [
            # UCB,  # XXX comment to only test klUCB
            klUCB,
            # Thompson,  # XXX comment to only test klUCB
            # Exp3PlusPlus,  # XXX comment to only test klUCB
        ]
        # for per_arm_restart in [True, False]
        # for full_restart_when_refresh in [True, False]
        for reset_for_all_change, reset_for_suboptimal_change in [
            (True,  False),  # optimal
            # (True,  True),  # sub sub optimal
            # (False, True),  # ? optimal
            # (False, False),  # sub optimal
        ]
    ] +
    # TODO How does the Exp3S algorithm work??
    [
        { "archtype": Exp3S, "params": {
            "horizon": HORIZON,
            "max_nb_random_events": NB_BREAK_POINTS,
            # "alpha0": alpha0,
            # "gamma0": gamma0,
        } }
        # for alpha0 in [100, 10, 1, 0.1, 0.01, 0.001]  # try different values for alpha, but using the formula, just a constant scaling
        # # for alpha0 in [100, 50, 1]  # try different values for alpha, but using the formula, just a constant scaling
        # for gamma0 in [100, 10, 1, 0.1, 0.01, 0.001]  # try different values for alpha, but using the formula, just a constant scaling
        # # for gamma0 in [10, 1, 0.01]  # try different values for alpha, but using the formula, just a constant scaling
    ] +
    # # The Exp3R algorithm works reasonably well
    # [
    #     { "archtype": Exp3R, "params": { "horizon": HORIZON, } }
    # ] +
    # # # XXX The Exp3RPlusPlus variant of Exp3R algorithm works also reasonably well
    # # [
    # #     { "archtype": Exp3RPlusPlus, "params": { "horizon": HORIZON, } }
    # # ] +
    # [
    #     # --- # Different versions of the discounted UCB algorithm
    #     { "archtype": DiscountedklUCB, "params": {
    #         "gamma": gamma,
    #         # "alpha": alpha,
    #         # "useRealDiscount": useRealDiscount,
    #     } }
    #     for gamma in GAMMAS
    #     # for alpha in ALPHAS
    #     # for useRealDiscount in [True, False]
    # ] +
    # # [
    # #     # --- # XXX experimental discounted UCB algorithm, knowing the horizon
    # #     { "archtype": DiscountedklUCBPlus, "params": { "max_nb_random_events": NB_BREAK_POINTS, "horizon": HORIZON, } }
    # # ] +
    # # [
    # #     # --- # XXX experimental sliding window algorithm
    # #     { "archtype": SlidingWindowRestart, "params": {
    # #         "policy": policy,
    # #         "tau": tau,
    # #         "threshold": eps,
    # #         "full_restart_when_refresh": True,
    # #         },
    # #         "change_label": r"SW-klUCB"
    # #     }
    # #     for tau in TAUS
    # #     for eps in EPSS
    # #     for policy in [klUCB]
    # # ] +
    # # [
    # #     # --- # Different versions of the sliding window UCB algorithm
    # #     { "archtype": SWUCB, "params": { "alpha": alpha, "tau": tau, } }
    # #     for alpha in ALPHAS for tau in TAUS
    # # ] +
    # # [
    # #     # --- # XXX experimental other version of the sliding window algorithm, knowing the horizon
    # #     { "archtype": SWUCBPlus, "params": { "horizon": HORIZON, "alpha": alpha, } }
    # #     for alpha in ALPHAS
    # # ] +
    # [
    #     # --- # Different versions of the sliding window klUCB algorithm
    #     { "archtype": SWklUCB, "params": { "tau": tau, }, "change_label": "SW-klUCB" }
    #     for tau in TAUS
    # ] +
    [  # XXX DiscountedThompson works REALLY well!
        {
            "archtype": DiscountedThompson,
            "params": { "posterior": DiscountedBeta, "gamma": gamma, },
            "change_label": "DTS",
        }
        for gamma in GAMMAS
    ] +
    # [  # XXX test the AdSwitch policy and its corrected version
    #     {
    #         "archtype": AdSwitch,
    #         "params": { "horizon": HORIZON, "C1": C1, "C2": C2,},
    #         "change_label": "AdSwitch",
    #     }
    #     for C1 in [1]  #, 10, 0.1]  # WARNING don't test too many parameters!
    #     for C2 in [1]  #, 10, 0.1]  # WARNING don't test too many parameters!
    # ] +
    # # The LM_DSEE algorithm seems to work fine! WARNING it seems TOO efficient!
    # [
    #     # nu = 0.5 means there is of the order Upsilon_T = T^0.5 = sqrt(T) change points
    #     # XXX note that for a fixed T it means nothing…
    #     # XXX But for T=10000 it is at most 100 changes, reasonable!
    #     {
    #         "archtype": LM_DSEE,
    #         "params": { "nu": 0.25, "DeltaMin": 0.1, "a": 1, "b": 0.25, },
    #         "change_label": r"LM-DSEE($\nu=1/4$, $\Delta^{min}=0.1$, $a=1$, $b=1/4$)",
    #     }
    # ] +
    # # DONE The SW_UCB_Hash algorithm works fine!
    # [
    #     { "archtype": SWHash_IndexPolicy, "params": { "alpha": alpha, "lmbda": lmbda, "policy": UCB } }
    #     for alpha in ALPHAS
    #     for lmbda in [1]  # [0.1, 0.5, 1, 5, 10]
    # ] +
    # XXX The Monitored_IndexPolicy with specific tuning of the input parameters
    [
        { "archtype": Monitored_IndexPolicy, "params": {
            "policy": policy,
            # "per_arm_restart": per_arm_restart,
            "horizon": HORIZON,
            "w": w,
            "delta": DELTA_for_MUCB,
        },
            "change_label": "M-klUCB",
        }
        # for per_arm_restart in PER_ARM_RESTART
        for policy in [
            # UCB,
            klUCB,  # XXX comment to only test UCB
        ]
        # for w in [20, 10*NB_ARMS, WINDOW_SIZE, NB_ARMS*WINDOW_SIZE, 2*NB_ARMS*WINDOW_SIZE]
        for w in [WINDOW_SIZE]
    ] +
    # XXX Test a few CD-MAB algorithms that need to know NB_BREAK_POINTS
    [
        { "archtype": archtype, "params": {
            "horizon": HORIZON,
            "policy": policy,
            "max_nb_random_events": NB_BREAK_POINTS,
            # "min_number_of_observation_between_change_point": MIN_NUMBER_OF_OBSERVATION_BETWEEN_CHANGE_POINT,
            "lazy_detect_change_only_x_steps": lazy_detect_change_only_x_steps,
            "epsilon": EPSILON_for_CUSUM,
            "use_localization": use_localization,
        },
            "change_label": "CUSUM-klUCB{}".format("(Localization)" if use_localization else ""),
        }
        for archtype in [
            CUSUM_IndexPolicy,
            # PHT_IndexPolicy,  # OK PHT_IndexPolicy is very much like CUSUM
        ]
        for policy in [
            # UCB,  # XXX comment to only test klUCB
            klUCB,
        ]
        for use_localization in [True, False]
        # for lazy_detect_change_only_x_steps in [1, 2, 5]
        # for lazy_detect_change_only_x_steps in [1]
        for lazy_detect_change_only_x_steps in ([20] if HORIZON <= 20000 else ([35] if HORIZON <= 100000 else [50]))
    ] +
    # # # # XXX Test a UCBLCB_IndexPolicy algorithm
    # # # [
    # # #     { "archtype": UCBLCB_IndexPolicy, "params": {
    # # #         "policy": policy,
    # # #         # "delta0": delta0,
    # # #         # "lazy_detect_change_only_x_steps": lazy_detect_change_only_x_steps,
    # # #         # "lazy_try_value_s_only_x_steps": lazy_try_value_s_only_x_steps,
    # # #     } }
    # # #     for policy in [
    # # #         # UCB,  # XXX comment to only test klUCB
    # # #         klUCB,
    # # #     ]
    # # #     # for delta0 in [10, 1, 0.1, 0.001]  # comment to use default parameter
    # # #     # for lazy_detect_change_only_x_steps in [1, 2, 5]  # XXX uncomment to use default value
    # # #     # for lazy_try_value_s_only_x_steps in [1, 2, 5]  # XXX uncomment to use default value
    # # # ] +
    # # # XXX Test GaussianGLR_IndexPolicy
    # # [
    # #     { "archtype": archtype, "params": {
    # #         "horizon": HORIZON,
    # #         "policy": policy,
    # #         "per_arm_restart": per_arm_restart,
    # #         "max_nb_random_events": NB_BREAK_POINTS,
    # #         "lazy_detect_change_only_x_steps": lazy_detect_change_only_x_steps,
    # #         "lazy_try_value_s_only_x_steps": lazy_try_value_s_only_x_steps,
    # #     } }
    # #     for archtype in [
    # #         GaussianGLR_IndexPolicy,    # OK GaussianGLR_IndexPolicy is very much like Bernoulli GLR
    # #         GaussianGLR_IndexPolicy_WithTracking,    # OK GaussianGLR_IndexPolicy_WithTracking is very much like Gaussian GLR and is more efficient
    # #         GaussianGLR_IndexPolicy_WithDeterministicExploration,    # OK GaussianGLR_IndexPolicy_WithDeterministicExploration is very much like Gaussian GLR and is more efficient
    # #         SubGaussianGLR_IndexPolicy, # OK SubGaussianGLR_IndexPolicy is very much like Gaussian GLR
    # #     ]
    # #     for policy in [
    # #         # UCB,  # XXX comment to only test klUCB
    # #         klUCB,
    # #     ]
    # #     for per_arm_restart in PER_ARM_RESTART
    # #     # for lazy_detect_change_only_x_steps in [50] #+ [2, 10]  # XXX uncomment to use default value
    # #     # for lazy_try_value_s_only_x_steps in [50] #+ [2, 10]  # XXX uncomment to use default value
    # #     for lazy_detect_change_only_x_steps, lazy_try_value_s_only_x_steps in ([(10, 10)] if HORIZON <= 20000 else ([(20, 20)] if HORIZON <= 100000 else [(50, 50)]))
    # # ] +
    # XXX Test BernoulliGLR_IndexPolicy
    [
        { "archtype": archtype, "params": {
            # "horizon": HORIZON,
            "policy": policy,
            "per_arm_restart": per_arm_restart,
            # "max_nb_random_events": NB_BREAK_POINTS,
            # "delta": delta0 * delta,
            "delta": delta,
            "alpha0": mult_alpha0 * alpha0,
            "lazy_detect_change_only_x_steps": lazy_detect_change_only_x_steps,
            "lazy_try_value_s_only_x_steps": lazy_try_value_s_only_x_steps,
            # "variant": variant,
            "use_localization": use_localization,
        },
        "change_label": r"GLR-klUCB({})".format(", ".join(s for s in [
            "Local" if per_arm_restart else "Global",
            "Localization" if use_localization else "",
            archname,
            # "threshold #{}".format(variant),
            # r"\delta 1" if delta == DELTA_1 else r"\delta 2",
            # r"$\delta={:.3g}$".format(delta),
            # r"$\alpha_0={:.3g}$".format(alpha0),
            # r"$\alpha_0={:.3g}$".format(mult_alpha0),
            # # "$\Delta n={}$".format(lazy_detect_change_only_x_steps) if lazy_detect_change_only_x_steps != 10 else "",
            # # "$\Delta s={}$".format(lazy_try_value_s_only_x_steps) if lazy_try_value_s_only_x_steps != 10 else "",
        ] if s))
        }
        for archtype, archname in [
            # (BernoulliGLR_IndexPolicy, "random expl."),   # OK BernoulliGLR_IndexPolicy is very much like CUSUM
            # (BernoulliGLR_IndexPolicy_WithTracking, "tracking"),   # OK GaussianGLR_IndexPolicy_WithTracking is very much like Bernoulli GLR and is more efficient
            # (BernoulliGLR_IndexPolicy_WithDeterministicExploration, "determ.expl."),   # OK GaussianGLR_IndexPolicy_WithDeterministicExploration is very much like Bernoulli GLR and is more efficient  XXX use this if using other exploration mechanisms XXX
            (BernoulliGLR_IndexPolicy_WithDeterministicExploration, ""),   # OK GaussianGLR_IndexPolicy_WithDeterministicExploration is very much like Bernoulli GLR and is more efficient
        ]
        for policy in [
            # UCB,  # XXX comment to only test klUCB
            # klUCB,
            klUCB_forGLR,
        ]
        # for (per_arm_restart, delta, alpha0) in zip(
        #     [True],
        #     [DELTA_LOCAL],
        #     [ALPHA_LOCAL],
        # )
        # for (per_arm_restart, delta, alpha0) in zip(
        #     [False],
        #     [DELTA_GLOBAL],
        #     [ALPHA_GLOBAL],
        # )
        for (per_arm_restart, delta, alpha0) in zip(
            [True, False],
            [DELTA_LOCAL, DELTA_GLOBAL],
            [ALPHA_LOCAL, ALPHA_GLOBAL],
        )
        for use_localization in [True, False]
        # for delta in [DELTA_1] # + [DELTA_2]  # XXX experimental!
        # for alpha0 in [ALPHA_1]  # XXX experimental!
        for mult_alpha0 in [1]  # comment from the + to use default parameter
        # for alpha0 in [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0]  # comment from the + to use default parameter
        # for delta0 in [1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]  # comment from the + to use default parameter
        # for lazy_detect_change_only_x_steps in [1, 5, 10, 20]  # XXX uncomment to use default value
        # for lazy_try_value_s_only_x_steps in [1, 5, 10, 20]  # XXX uncomment to use default value
        # for lazy_detect_change_only_x_steps, lazy_try_value_s_only_x_steps in [(1, 1), (10, 10)]
        # for lazy_detect_change_only_x_steps, lazy_try_value_s_only_x_steps in [(1, 1)]
        for lazy_detect_change_only_x_steps, lazy_try_value_s_only_x_steps in ([(10, 10)] if HORIZON <= 20000 else ([(20, 20)] if HORIZON <= 100000 else [(50, 50)]))
        # for variant in [None, 1, 2, 3]  # XXX variant for the threshold function...
    ] +
    []
})

# XXX Huge hack! Use this if you want to modify the legends
configuration.update({
    "append_labels": {
        policyId: cfg_policy.get("append_label", "")
        for policyId, cfg_policy in enumerate(configuration["policies"])
        if "append_label" in cfg_policy
    },
    "change_labels": {
        policyId: cfg_policy.get("change_label", "")
        for policyId, cfg_policy in enumerate(configuration["policies"])
        if "change_label" in cfg_policy
    }
})

print("Loaded experiments configuration from 'configuration_nonstationnary.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
print("configuration['environment'] =", configuration["environment"])  # DEBUG
