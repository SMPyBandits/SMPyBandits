# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the single-player case, for comparing doubling-trick doubling schemes.
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
    print("Warning: this script 'configuration.py' is NOT executable. Use 'main.py' or 'make single' ...")  # DEBUG
    exit(0)

# Import arms
from Arms import *

# Import algorithms
from Policies import *

#: HORIZON : number of time steps of the experiments.
#: Warning Should be >= 10000 to be interesting "asymptotically".
HORIZON = 45678
HORIZON = int(getenv('T', HORIZON))

#: REPETITIONS : number of repetitions of the experiments.
#: Warning: Should be >= 10 to be statistically trustworthy.
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
REPETITIONS = 1000
REPETITIONS = int(getenv('N', REPETITIONS))

#: To profile the code, turn down parallel computing
DO_PARALLEL = False  # XXX do not let this = False
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1 or REPETITIONS == -1) and DO_PARALLEL

#: Number of jobs to use for the parallel computations. -1 means all the CPU cores, 1 means no parallelization.
N_JOBS = -1 if DO_PARALLEL else 1
if CPU_COUNT > 4:  # We are on a server, let's be nice and not use all cores
    N_JOBS = min(CPU_COUNT, max(int(CPU_COUNT / 3), CPU_COUNT - 8))
N_JOBS = int(getenv('N_JOBS', N_JOBS))
if REPETITIONS == -1:
    REPETITIONS = max(N_JOBS, CPU_COUNT)

# Parameters for the arms
UNBOUNDED_VARIANCE = 1   #: Variance of unbounded Gaussian arms
VARIANCE = 0.05   #: Variance of Gaussian arms

#: Number of arms for non-hard-coded problems (Bayesian problems)
NB_ARMS = 9
NB_ARMS = int(getenv('K', NB_ARMS))
NB_ARMS = int(getenv('NB_ARMS', NB_ARMS))

#: Default value for the lower value of means
lower = 0.
#: Default value for the amplitude value of means
amplitude = 1.

#: Type of arms for non-hard-coded problems (Bayesian problems)
ARM_TYPE = "Bernoulli"
ARM_TYPE = str(getenv('ARM_TYPE', ARM_TYPE))
mapping_ARM_TYPE = {
    "Constant": Constant,
    "Uniform": UniformArm,
    "Bernoulli": Bernoulli, "B": Bernoulli,
    "Gaussian": Gaussian, "Gauss": Gaussian, "G": Gaussian,
    "UnboundedGaussian": UnboundedGaussian,
    "Poisson": Poisson, "P": Poisson,
    "Exponential": ExponentialFromMean, "Exp": ExponentialFromMean, "E": ExponentialFromMean,
    "Gamma": GammaFromMean,
}
if ARM_TYPE == "UnboundedGaussian":
    lower = -5
    amplitude = 10
ARM_TYPE = mapping_ARM_TYPE[ARM_TYPE]


#: This dictionary configures the experiments
configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Arms
    "environment": [  # XXX Bernoulli arms
        {   # XXX A very easy problem, but it is used in a lot of articles
            "arm_type": Bernoulli,
            "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        # XXX Default!
        {   # A very easy problem (X arms), but it is used in a lot of articles
            "arm_type": ARM_TYPE,
            "params": uniformMeans(nbArms=NB_ARMS, delta=1./(1. + NB_ARMS), lower=lower, amplitude=amplitude)
        },
        {   # A Bayesian problem: every repetition use a different mean vectors!
            "arm_type": ARM_TYPE,
            "params": {
                "function": randomMeans,
                "args": {
                    "nbArms": NB_ARMS,
                    "mingap": None,
                    # "mingap": 0.0000001,
                    # "mingap": 0.1,
                    # "mingap": 1. / (3 * NB_ARMS),
                    "lower": 0.,
                    "amplitude": 1.,
                    "isSorted": True,
                }
            }
        },
    ],
}


#: Warning: if using Exponential or Gaussian arms, gives klExp or klGauss to KL-UCB-like policies!
klucb = klucb_mapping.get(str(configuration['environment'][0]['arm_type']), klucbBern)



POLICIES_FOR_DOUBLING_TRICK = [
        # klUCB,  # XXX Don't need the horizon, but suffer from the restart (to compare)
        # UCBH,
        # MOSSH,
        klUCBPlusPlus,
        # ApproximatedFHGittins,
    ]

# Just add the klUCB or UCB baseline
configuration["policies"] = [
    {
        "archtype": klUCB,
        "archtype": UCB,
        "params": {
        }
    }
]
# Smart way of adding list of Doubling Trick versions
for policy in POLICIES_FOR_DOUBLING_TRICK:
    # First add the non-doubling trick version
    accept_horizon = True
    try:
        _ = policy(NB_ARMS, horizon=HORIZON)
    except TypeError:
        accept_horizon = False  # don't use horizon
    configuration["policies"] += [
        {
            "archtype": policy,
            "params": {
                "horizon": HORIZON,
                # "horizon": max(HORIZON + 100, int(1.05 * HORIZON)),
                # "alpha": 0.5,  # only for ApproximatedFHGittins
            } if accept_horizon else {
                # "alpha": 0.5,  # only for ApproximatedFHGittins
            }
        }
    ]
    # Then add the doubling trick version
    configuration["policies"] += [
        {
            "archtype": DoublingTrickWrapper,
            "params": {
                "next_horizon": next_horizon,
                "full_restart": full_restart,
                "policy": policy,
                # "alpha": 0.5,  # only for ApproximatedFHGittins
            }
        }
        for full_restart in [
            True,
            False,
        ]
        for next_horizon in [
            next_horizon__arithmetic,
            next_horizon__geometric,
            # next_horizon__exponential,
            next_horizon__exponential_fast,
            next_horizon__exponential_slow,
            next_horizon__exponential_generic
        ]
    ]


print("Loaded experiments configuration from 'configuration.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
