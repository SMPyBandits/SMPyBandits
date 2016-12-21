# -*- coding: utf-8 -*-
"""
Configuration for the simulations.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.2"

# Import arms
from Arms.Bernoulli import Bernoulli
from Arms.Exponential import Exponential
from Arms.Gaussian import Gaussian
from Arms.Poisson import Poisson
# Import algorithms
from Policies import *

# HORIZON : number of time steps of the experiments
# XXX Should be >= 10000 to be interesting "asymptotically"
HORIZON = 500
HORIZON = 2000
HORIZON = 3000
HORIZON = 10000
# HORIZON = 20000
# HORIZON = 30000
# HORIZON = 100000

# REPETITIONS : number of repetitions of the experiments
# XXX Should be >= 10 to be stastically trustworthy
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
# REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
# REPETITIONS = 500
# REPETITIONS = 200
# REPETITIONS = 100
# REPETITIONS = 50
REPETITIONS = 20
# REPETITIONS = 1  # XXX To profile the code, turn down parallel computing

DO_PARALLEL = False  # XXX do not let this = False  # To profile the code, turn down parallel computing
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1) and DO_PARALLEL
N_JOBS = -1 if DO_PARALLEL else 1

# Parameters for the epsilon-greedy and epsilon-... policies
EPSILON = 0.1

# Temperature for the softmax
TEMPERATURE = 0.01  # When -> 0, more greedy
TEMPERATURE = 0.1
TEMPERATURE = 0.5
TEMPERATURE = 1
TEMPERATURE = 10
TEMPERATURE = 100   # When -> oo, more uniformly at random
# TEMPERATURE = 10.0 / HORIZON  # Not sure ??!
TEMPERATURE = 0.05

# XXX try different values for the learning rate for my aggregated bandit
LEARNING_RATE = 0.05
LEARNING_RATE = 0.1
LEARNING_RATE = 0.2
LEARNING_RATE = 0.5
LEARNING_RATE = 0.01

# To try more learning rates in one run
LEARNING_RATES = [10, 2, 1, 0.1, 0.01, 0.001, 0.0001, 0.00005]
LEARNING_RATES = [10, 1, 0.1, 0.01, 0.001]
LEARNING_RATES = [LEARNING_RATE]

# XXX try different values for time tau for the decreasing rate for my aggregated bandit
# FIXED I tried to make self.learningRate decrease when self.t increase, it was not better
DECREASE_RATE = None
DECREASE_RATE = HORIZON / 2.0


TEST_AGGR = False  # XXX do not let this = False
TEST_AGGR = True
UPDATE_ALL_CHILDREN = True
UPDATE_ALL_CHILDREN = False  # XXX do not let this = False


# Parameters for the arms
VARIANCE = 0.05   # Variance of Gaussian arms


# XXX This dictionary configures the experiments
configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 6,  # Max joblib verbosity
    # # --- Random events - TODO finish the improvement on Evaluator.py to support these parameters
    # "random_shuffle": True,
    # # "random_invert": False,
    # "nb_random_events": 5,
    # --- Arms
    "environment": [  # Bernoulli arms
        # {   # A very very easy problem: 3 arms, one bad, one average, one good
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.5, 0.9]
        # },
        # {   # Another very easy problem: 3 arms, two very bad, one bad
        #     "arm_type": Bernoulli,
        #     "params": [0.04, 0.05, 0.1]
        # },
        # {   # A very easy problem, but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # },
        # {   # An other problem, best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3 - 0.6) and very good arms (0.78, 0.8, 0.82)
        #     "arm_type": Bernoulli,
        #     "params": [0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.78, 0.8, 0.82]
        # },
        # {   # Lots of bad arms, significative difference between the best and the others
        #     "arm_type": Bernoulli,
        #     "params": [0.001, 0.001, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.3]
        # },
        {   # One optimal arm, much better than the others, but *lots* of bad arms
            "arm_type": Bernoulli,
            "params": [0.001, 0.001, 0.001, 0.001, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2, 0.5]
        },
        # {   # An other problem (17 arms), best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3, 0.6) and very good arms (0.78, 0.85)
        #     "arm_type": Bernoulli,
        #     "params": [0.005, 0.01, 0.015, 0.02, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.78, 0.8, 0.82, 0.83, 0.84, 0.85]
        # },
    ],
    # DONE I tried with other arms distribution: Exponential, it works similarly
    # FIXME if using Exponential arms, gives klExp to KL-UCB-like policies!
    # "environment": [  # Exponential arms
    #     {   # An example problem with  arms
    #         "arm_type": Exponential,
    #         "params": [2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     },
    # ],
    # DONE I tried with other arms distribution: Gaussian, it works similarly
    # FIXME if using Gaussian arms, gives klGauss to KL-UCB-like policies!
    # "environment": [  # Gaussian arms
    #     {   # An example problem with  arms
    #         "arm_type": Gaussian,
    #         "params": [(0.1, VARIANCE), (0.2, VARIANCE), (0.3, VARIANCE), (0.4, VARIANCE), (0.5, VARIANCE), (0.6, VARIANCE), (0.7, VARIANCE), (0.8, VARIANCE), (0.9, VARIANCE)]
    #     },
    # ],
}

nbArms = len(configuration['environment'][0]['params'])
if len(configuration['environment']) > 1:
    raise ValueError("WARNING do not use this hack if you try to use more than one environment.")

configuration.update({
    "policies": [
        # # --- Stupid algorithms
        # {
        #     "archtype": Uniform,   # The stupidest policy, fully uniform
        #     "params": {}
        # },
        # {
        #     "archtype": TakeRandomFixedArm,   # The stupidest policy
        #     "params": {}
        # },
        # {
        #     "archtype": TakeRandomFixedArm,   # The stupidest policy
        #     "params": {}
        # },
        # # --- Full or partial knowledge algorithms
        # TakeFixedArm(nbArms, nbArms - 1),  # Take best arm!
        # TakeFixedArm(nbArms, nbArms - 2),  # Take second best arm!
        # TakeFixedArm(nbArms, 0),  # Take worse arm!
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
        # --- UCB algorithms
        # {
        #     "archtype": UCB,   # This basic UCB is very worse than the other
        #     "params": {}
        # },
        # {
        #     "archtype": UCBplus,
        #     "params": {}
        # },
        {
            "archtype": UCBopt,
            "params": {}
        },
        # {
        #     "archtype": UCBrandomInit,
        #     "params": {}
        # },
        # {
        #     "archtype": UCBV,   # UCB with variance term
        #     "params": {}
        # },
        # {
        #     "archtype": UCBtuned,   # UCB with variance term and one trick
        #     "params": {}
        # },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         "alpha": 4          # Below the alpha=4 like old classic UCB
        #     }
        # },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         "alpha": 1
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
        #         "alpha": 0.1          # XXX Below the theoretically acceptable value!
        #     }
        # },
        # --- Softmax algorithms
        # {
        #     "archtype": Softmax,   # This basic Softmax is very bad
        #     "params": {
        #         "temperature": TEMPERATURE
        #     }
        # },
        {
            "archtype": SoftmaxDecreasing,   # Parameter-free Softmax
            "params": {}
        },
        # {
        #     "archtype": SoftmaxWithHorizon,  # Parameter-free Softmax knowing the horizon
        #     "params": {
        #         "horizon": HORIZON
        #     }
        # },
        # --- MOSS algorithm, quite efficient
        {
            "archtype": MOSS,
            "params": {}
        },
        # --- Thompson algorithms
        {
            "archtype": Thompson,
            "params": {}
        },
        # --- KL algorithms
        # {
        #     "archtype": klUCB,
        #     "params": {}
        # },
        {
            "archtype": klUCBPlus,
            "params": {}
        },
        # {
        #     "archtype": klUCBHPlus,
        #     "params": {
        #         "horizon": HORIZON
        #     }
        # },
        # # {
        # #     "archtype": KLempUCB,   # Empirical KL-UCB algorithm non-parametric policy - XXX does not work as far as now
        # #     "params": {}
        # # },
        {
            "archtype": BayesUCB,
            "params": {}
        },
        # # --- AdBandits with different alpha paramters
        # {
        #     "archtype": AdBandits,
        #     "params": {
        #         "alpha": 0.5,
        #         "horizon": HORIZON
        #     }
        # },
        # {
        #     "archtype": AdBandits,
        #     "params": {
        #         "alpha": 0.125,
        #         "horizon": HORIZON
        #     }
        # },
        # {
        #     "archtype": AdBandits,
        #     "params": {
        #         "alpha": 0.01,
        #         "horizon": HORIZON
        #     }
        # },
    ]
})

# Dynamic hack to force the Aggr (policies aggregator) to use all the policies previously/already defined
if TEST_AGGR:
    # print("configuration['policies'] =", CURRENT_POLICIES)  # DEBUG
    NON_AGGR_POLICIES = configuration["policies"]
    for LEARNING_RATE in LEARNING_RATES:
        CURRENT_POLICIES = configuration["policies"]
        # Add one Aggr policy
        configuration["policies"] = CURRENT_POLICIES + [{
            "archtype": Aggr,
            "params": {
                "learningRate": LEARNING_RATE,
                "decreaseRate": DECREASE_RATE,
                "update_all_children": UPDATE_ALL_CHILDREN,
                "children": NON_AGGR_POLICIES
            },
        }]

print("Loaded experiments configuration from 'configuration.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
