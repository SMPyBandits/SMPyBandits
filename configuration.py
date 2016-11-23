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
HORIZON = 20000
HORIZON = 30000
HORIZON = 10000
HORIZON = 2000

# REPETITIONS : number of repetitions of the experiments
# XXX Should be >= 10 to be stastically trustworthy
REPETITIONS = 1  # To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores
REPETITIONS = 200
REPETITIONS = 500
REPETITIONS = 100
REPETITIONS = 50
REPETITIONS = 20

DO_PARALLEL = False  # XXX do not let this = False
DO_PARALLEL = True
N_JOBS = -1 if DO_PARALLEL else 1

# Parameters for the policies
EPSILON = 0.1

TEMPERATURE = 0.01  # When -> 0, more greedy
TEMPERATURE = 0.1
TEMPERATURE = 0.5
TEMPERATURE = 1
TEMPERATURE = 10
TEMPERATURE = 100   # When -> oo, more uniformly at random
TEMPERATURE = 0.01

# FIXME improve the learning rate for my aggregated bandit
LEARNING_RATE = 0.2
LEARNING_RATE = 0.5
LEARNING_RATE = 0.1
LEARNING_RATE = 0.05
# FIXED I tried to make self.learningRate decrease when self.t increase, it was not better

# To try more learning rates in one run
# LEARNING_RATES = [10, 2, 1, 0.1, 0.01, 0.001, 0.0001, 0.00005]
LEARNING_RATES = [10, 1, 0.1, 0.01, 0.001]
LEARNING_RATES = [LEARNING_RATE]


TEST_AGGR = True
UPDATE_ALL_CHILDREN = False  # XXX do not let this = False
UPDATE_ALL_CHILDREN = True
ONE_JOB_BY_CHILDREN = True  # XXX do not let this = True
ONE_JOB_BY_CHILDREN = False


configuration = {
    "horizon": HORIZON,
    "repetitions": REPETITIONS,
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 5,  # Max joblib verbosity
    "environment": [
        # FIXME try with other arms distribution: Exponential, Gaussian, Poisson, etc!
        # {   # A very easy problem, but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "probabilities": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # },
        {   # An other problem, best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3 - 0.6) and very good arms (0.78, 0.8, 0.82)
            "arm_type": Bernoulli,
            "probabilities": [0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.78, 0.8, 0.82]
        },
        # {   # Lots of bad arms, significative difference between the best and the others
        #     "arm_type": Bernoulli,
        #     "probabilities": [0.001, 0.001, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.3]
        # },
        # {   # One optimal arm, much better than the others, but *lots* of bad arms
        #     "arm_type": Bernoulli,
        #     "probabilities": [0.001, 0.001, 0.001, 0.001, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2, 0.5]
        # },
    ],
    "policies": [
        # # --- Stupid algorithms
        # {
        #     "archtype": Dummy,   # The stupidest policy
        #     "params": {}
        # },
        # # --- Epsilon-... algorithms
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
        # # --- UCB algorithms
        # {
        #     "archtype": UCB,   # This basic UCB is very worse than the other
        #     "params": {}
        # },
        # {
        #     "archtype": UCBV,   # UCB with variance term
        #     "params": {}
        # },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         # "alpha": 4          # Like usual UCB
        #         "alpha": 1          # Limit case
        #     }
        # },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         "alpha": 1.25          # Above the alpha=4 like usual UCB
        #     }
        # },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         "alpha": 0.5          # XXX Below the theoretically acceptable value!
        #     }
        # },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         "alpha": 0.25          # XXX Below the theoretically acceptable value!
        #     }
        # },
        # --- Softmax algorithms
        {
            "archtype": Softmax,   # This basic Softmax is very bad
            "params": {
                "temperature": TEMPERATURE
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
        # # {
        # #     "archtype": KLempUCB,   # Empirical KL-UCB algorithm non-parametric policy - XXX does not work as far as now
        # #     "params": {}
        # # },
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
        {
            "archtype": AdBandit,
            "params": {
                "alpha": 0.125,
                "horizon": HORIZON
            }
        },
        # {
        #     "archtype": AdBandit,
        #     "params": {
        #         "alpha": 0.01,
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
        # # --- Manually, one Aggr policy
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

    # print("configuration['policies'] =", CURRENT_POLICIES)  # DEBUG
    NON_AGGR_POLICIES = configuration["policies"]
    for LEARNING_RATE in LEARNING_RATES:
        CURRENT_POLICIES = configuration["policies"]
        # Add one Aggr policy
        configuration["policies"] = CURRENT_POLICIES + [{
            "archtype": Aggr,
            "params": {
                "learningRate": LEARNING_RATE,
                # "decreaseRate": None,
                "decreaseRate": 1000.,
                "update_all_children": UPDATE_ALL_CHILDREN,
                "children": NON_AGGR_POLICIES,
                "n_jobs": N_JOBS,
                "verbosity": 0 if N_JOBS == 1 else 1,
                "one_job_by_children": ONE_JOB_BY_CHILDREN
            },
        }]

print("Loaded experiments configuration from 'configuration.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
