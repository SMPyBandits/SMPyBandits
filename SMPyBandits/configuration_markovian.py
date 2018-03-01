# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the single-player case.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.5"

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
HORIZON = 500
HORIZON = 2000
HORIZON = 3000
HORIZON = 5000
HORIZON = 10000
# HORIZON = 100000

#: REPETITIONS : number of repetitions of the experiments.
#: Warning: Should be >= 10 to be statistically trustworthy.
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
# REPETITIONS = 1000
# REPETITIONS = 100
REPETITIONS = 50
# REPETITIONS = 20

#: To profile the code, turn down parallel computing
DO_PARALLEL = False  # XXX do not let this = False
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1) and DO_PARALLEL

#: Number of jobs to use for the parallel computations. -1 means all the CPU cores, 1 means no parallelization.
N_JOBS = -1 if DO_PARALLEL else 1
if CPU_COUNT > 4:  # We are on a server, let's be nice and not use all cores
    N_JOBS = min(CPU_COUNT, max(int(CPU_COUNT / 3), CPU_COUNT - 8))
N_JOBS = int(getenv('N_JOBS', N_JOBS))


# Parameters for the arms
VARIANCE = 0.05   #: Variance of Gaussian arms
VARIANCE = 10   #: Variance of Gaussian arms


#: To know if my Aggregator policy is tried.
TEST_Aggregator = False  # XXX do not let this = False if you want to test my Aggregator policy
TEST_Aggregator = True


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
    "environment": [
        # # DONE Bernoulli arms
        # {   # A very very easy problem: 3 arms, one bad, one average, one good
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.5, 0.9]
        # },
        # DONE Markovian arms with {0, 1} rewards
        {
            "arm_type": "Markovian",
            "params": {
                # "rested": True,
                "rested": False,  # FIXME
                # XXX Example from [Kalathil et al., 2012](https://arxiv.org/abs/1206.3582) Table 1
                "transitions": [
                    # 1st arm, either a dictionary, to customize the states
                    {   # Mean = 0.375
                        (0, 0): 0.7, (0, 1): 0.3,
                        (1, 0): 0.5, (1, 1): 0.5,
                    },
                    # 2nd arm, or a right transition matrix, with states [| 0, n-1 |]
                    [[0.2, 0.8], [0.6, 0.4]],  # Mean = 0.571
                ],
                # FIXME make this by default! include it in MAB.py and not in the configuration!
                "steadyArm": Bernoulli
            }
        },
        # # DONE Markovian arms with non-binary rewards
        # {
        #     "arm_type": "Markovian",
        #     "params": {
        #         "rested": True,
        #         # "rested": False,  # FIXME
        #         "transitions": [
        #             # 1st arm, rewars are in {0, 0.5, 1} with 3 states
        #             {   # Mean = 0.5
        #                 (0, 0): 0.75, (0, 0.5): 0.125, (0, 1): 0.125,
        #                 (0.5, 0): 0.125, (0.5, 0.5): 0.75, (0.5, 1): 0.125,
        #                 (1, 0): 0.125, (1, 0.5): 0.125, (1, 1): 0.75,
        #             },
        #             # 2nd arm, rewars are in {0, 1} with 2 states
        #             {   # Mean = 0.357...
        #                 (0, 0): 0.5, (0, 1): 0.5,
        #                 (1, 0): 0.9, (1, 1): 0.1,
        #             },
        #         ],
        #         # FIXME make this by default! include it in MAB.py and not in the configuration!
        #         "steadyArm": Bernoulli
        #     }
        # },
    ],
}

if len(configuration['environment']) > 1:
    raise ValueError("WARNING do not use this hack if you try to use more than one environment.")
    # Note: I dropped the support for more than one environments, for this part of the configuration, but not the simulation code

try:
    #: Number of arms *in the first environment*
    nbArms = int(configuration['environment'][0]['params']['args']['nbArms'])
except (TypeError, KeyError):
    nbArms = len(configuration['environment'][0]['params'])

#: Warning: if using Exponential or Gaussian arms, gives klExp or klGauss to KL-UCB-like policies!
klucb = klucb_mapping.get(str(configuration['environment'][0]['arm_type']), klucbBern)


configuration.update({
    "policies": [
        # --- UCB algorithms
        {
            "archtype": UCBalpha,   # UCB with custom alpha parameter
            "params": {
                "alpha": 1
            }
        },
        {
            "archtype": UCBalpha,   # UCB with custom alpha parameter
            "params": {
                "alpha": 0.5          # XXX Below the theoretically acceptable value!
            }
        },
        # --- DMED algorithm, similar to klUCB
        {
            "archtype": DMED,
            "params": {
                "genuine": True,
            }
        },
        # --- Thompson algorithms
        {
            "archtype": Thompson,
            "params": {}
        },
        # --- KL algorithms
        {
            "archtype": klUCB,
            "params": {
                "klucb": klucb
            }
        },
        {
            "archtype": klUCBPlus,
            "params": {
                "klucb": klucb
            }
        },
        # --- Bayes UCB algorithms
        {
            "archtype": BayesUCB,
            "params": {}
        },
        # --- Finite-Horizon Gittins index
        {
            "archtype": ApproximatedFHGittins,
            "params": {
                "horizon": 1.1 * HORIZON,
                "alpha": 1,
            }
        },
        {
            "archtype": ApproximatedFHGittins,
            "params": {
                "horizon": 1.1 * HORIZON,
                "alpha": 0.5,
            }
        },
    ]
})

# Dynamic hack to force the Aggregator (policies aggregator) to use all the policies previously/already defined
if TEST_Aggregator:
    NON_AGGR_POLICIES = configuration["policies"]
    for UPDATE_LIKE_EXP4 in [False, True]:
        CURRENT_POLICIES = configuration["policies"]
        # Add one Aggregator policy
        configuration["policies"] = [{
            "archtype": Aggregator,
            "params": {
                "unbiased": False,
                "update_all_children": False,
                "decreaseRate": 'auto',
                "learningRate": 1,
                "children": NON_AGGR_POLICIES,
                "update_like_exp4": UPDATE_LIKE_EXP4,
                # "horizon": HORIZON  # XXX uncomment to give the value of horizon to have a better learning rate
            },
        }] + CURRENT_POLICIES

print("Loaded experiments configuration from 'configuration.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
