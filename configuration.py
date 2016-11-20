# -*- coding: utf-8 -*-
"""
Configuration for the simulations.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.2"


# Import arms
from Arms.Bernoulli import Bernoulli
# Import algorithms
from Policies import *


# HORIZON : number of time steps of the experiments
# XXX Should be >= 10000 to be interesting "asymptotically"
HORIZON = 500
HORIZON = 1000
HORIZON = 3000
HORIZON = 30000
HORIZON = 10000

# REPETITIONS : number of repetitions of the experiments
# XXX Should be >= 10 to be stastically trustworthy
REPETITIONS = 1
REPETITIONS = 200
REPETITIONS = 5
REPETITIONS = 20
REPETITIONS = 100

# DO_PARALLEL = False
DO_PARALLEL = True

EPSILON = 0.05

# FIXME improve the learning rate for my aggregated bandit
LEARNING_RATE = 0.5
LEARNING_RATE = 0.1
LEARNING_RATE = 0.05
LEARNING_RATE = 0.2

TEST_AGGR = True
updateAllChildren = True
updateAllChildren = False


configuration = {
    "horizon": HORIZON,
    "repetitions": REPETITIONS,
    "n_jobs": -1 if DO_PARALLEL else 1,    # = nb of CPU cores
    "verbosity": 5,  # Max joblib verbosity
    "environment": [
        {
            "arm_type": Bernoulli,
            "probabilities": [0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.79, 0.8, 0.81]
        },
        # {
        #     "arm_type": Bernoulli,
        #     "probabilities": [0.001, 0.001, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1]
        # },
        # {   # One optimal arm, much better than the others, but lots of bad arms
        #     "arm_type": Bernoulli,
        #     "probabilities": [0.001, 0.001, 0.001, 0.001, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2, 0.3]
        # },
    ],
    "policies": [
        # {
        #     "archtype": Dummy,   # The stupidest policy
        #     "params": {}
        # },
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
        {
            "archtype": EpsilonFirst,   # This basic EpsilonFirst is also very bad
            "params": {
                "epsilon": EPSILON,
                "horizon": HORIZON
            }
        },
        {
            "archtype": UCB,   # This basic UCB is very worse than the other
            "params": {}
        },
        {
            "archtype": Thompson,
            "params": {}
        },
        # {
        #     "archtype": klUCB,
        #     "params": {}
        # },
        {
            "archtype": BayesUCB,
            "params": {}
        },
        # {
        #     "archtype": AdBandit,
        #     "params": {
        #         "alpha": 0.5,
        #         "horizon": HORIZON
        #     }
        # },
        # {
        #     "archtype": Aggr,
        #     "params": {
        #         "learningRate": LEARNING_RATE,
        #         "children": [
        #             {
        #                 "archtype": UCB,
        #                 "params": {}
        #             },
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
    current_policies = configuration["policies"]
    # print("configuration['policies'] =", current_policies)  # DEBUG
    configuration["policies"] = current_policies + [{  # Add one Aggr policy
        "archtype": Aggr,
        "params": {
            "learningRate": LEARNING_RATE,
            "updateAllChildren": updateAllChildren,
            "children": current_policies,
        },
    }]

# print("Loaded experiments configuration from 'configuration.py' :")
# print("configuration['policies'] =", configuration["policies"])  # DEBUG
