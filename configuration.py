# -*- coding: utf-8 -*-
"""
Configuration for the simulations.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.2"

# Import arms
from Arms.Bernoulli import Bernoulli
# from Arms.Exponential import Exponential
# from Arms.Gaussian import Gaussian
# from Arms.Poisson import Poisson

# Import algorithms
from Policies import *


# HORIZON : number of time steps of the experiments
# XXX Should be >= 10000 to be interesting "asymptotically"
HORIZON = 500
HORIZON = 3000
HORIZON = 30000
HORIZON = 10000
HORIZON = 1000

# REPETITIONS : number of repetitions of the experiments
# XXX Should be >= 10 to be stastically trustworthy
REPETITIONS = 1
REPETITIONS = 5
REPETITIONS = 20
REPETITIONS = 200
REPETITIONS = 100
REPETITIONS = 50

DO_PARALLEL = False  # XXX do not let this = False
DO_PARALLEL = True
N_JOBS = -1 if DO_PARALLEL else 1

EPSILON = 0.1

# FIXME improve the learning rate for my aggregated bandit
LEARNING_RATE = 0.05
LEARNING_RATE = 0.2
LEARNING_RATE = 0.5
LEARNING_RATE = 0.1

TEST_AGGR = True
UPDATE_ALL_CHILDREN = False  # XXX do not let this = False
UPDATE_ALL_CHILDREN = True
ONE_JOB_BY_CHILDREN = True  # XXX do not let this = True
ONE_JOB_BY_CHILDREN = False


configuration = {
    "horizon": HORIZON,
    "repetitions": REPETITIONS,
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 4,  # Max joblib verbosity
    "environment": [
        # {
        #     "arm_type": Bernoulli,
        #     "probabilities": [0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.78, 0.8, 0.82]
        # },
        # {
        #     "arm_type": Bernoulli,
        #     "probabilities": [0.001, 0.001, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1]
        # },
        {   # One optimal arm, much better than the others, but lots of bad arms
            "arm_type": Bernoulli,
            "probabilities": [0.001, 0.001, 0.001, 0.001, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2, 0.3]
        },
    ],
    "policies": [
        # --- Stupid algorithms
        # {
        #     "archtype": Dummy,   # The stupidest policy
        #     "params": {}
        # },
        # --- Epsilon-... algorithms
        # {
        #     "archtype": EpsilonGreedy,   # This basic EpsilonGreedy is very bad
        #     "params": {
        #         "epsilon": EPSILON
        #     }
        # },
        # {
        #     "archtype": EpsilonDecreasing,   # This basic EpsilonGreedy is also very bad
        #     "params": {
        #         "epsilon": EPSILON,
        #         "decreasingRate": 0.005,
        #     }
        # },
        # {
        #     "archtype": EpsilonFirst,   # This basic EpsilonFirst is also very bad
        #     "params": {
        #         "epsilon": EPSILON,
        #         "horizon": HORIZON
        #     }
        # },
        # --- UCB algorithms
        # {
        #     "archtype": UCB,   # This basic UCB is very worse than the other
        #     "params": {}
        # },
        # {
        #     "archtype": UCBV,   # UCB with variance term
        #     "params": {}
        # },
        {
            "archtype": UCBalpha,   # UCB with custom alpha parameter
            "params": {
                # "alpha": 4          # Like usual UCB
                "alpha": 1          # Limit case
            }
        },
        {
            "archtype": UCBalpha,   # UCB with custom alpha parameter
            "params": {
                "alpha": 1          # Limit case
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
            "params": {}
        },
        # {
        #     "archtype": KLempUCB,   # Empirical KL-UCB algorithm non-parametric policy - XXX does not work well
        #     "params": {}
        # },
        {
            "archtype": BayesUCB,
            "params": {}
        },
        {
            "archtype": AdBandit,
            "params": {
                "alpha": 0.5,
                "horizon": HORIZON
            }
        },
        # {
        #     "archtype": AdBandit,
        #     "params": {
        #         "alpha": 0.25,
        #         "horizon": HORIZON
        #     }
        # },
        # {
        #     "archtype": AdBandit,
        #     "params": {
        #         "alpha": 1,
        #         "horizon": HORIZON
        #     }
        # },
        # {
        #     "archtype": Aggr,
        #     "params": {
        #         "learningRate": LEARNING_RATE,
        #         "children": [
        #             {
        #                 "archtype": Thompson,
        #                 "params": {}
        #             },
        #             {
        #                 "archtype": klUCB,
        #                 "params": {}
        #             },
        #             {
        #                 "archtype": BayesUCB,
        #                 "params": {}
        #             },
        #             {
        #                 "archtype": AdBandit,
        #                 "params": {
        #                     "alpha": 0.5,
        #                     "horizon": HORIZON
        #                 }
        #             },
        #         ]
        #     }
        # },
    ]
}

# Dynamic hack to force the Aggr (policies aggregator) to use all the policies previously/already defined
if TEST_AGGR:
    # N_JOBS = -1  # XXX for experiments
    N_JOBS = 1  # XXX for experiments

    CURRENT_POLICIES = configuration["policies"]
    # print("configuration['policies'] =", CURRENT_POLICIES)  # DEBUG
    configuration["policies"] = CURRENT_POLICIES + [{  # Add one Aggr policy
        "archtype": Aggr,
        "params": {
            "learningRate": LEARNING_RATE,
            "update_all_children": UPDATE_ALL_CHILDREN,
            "children": CURRENT_POLICIES,
            "n_jobs": N_JOBS,
            "verbosity": 0 if N_JOBS == 1 else 1,
            "one_job_by_children": ONE_JOB_BY_CHILDREN
        },
    }]

print("Loaded experiments configuration from 'configuration.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
