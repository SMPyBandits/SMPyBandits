# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the single-player case.
"""
from __future__ import print_function, division

__author__ = "Lilian Besson"
__version__ = "0.5"

# Tries to know number of CPU
try:
    from multiprocessing import cpu_count
    CPU_COUNT = cpu_count()
except ImportError:
    CPU_COUNT = 1

# Import arms
from Arms import makeMeans
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
# HORIZON = 300000

# DELTA_T_SAVE : save only 1 / DELTA_T_SAVE points, to speed up computations, use less RAM, speed up plotting etc.
DELTA_T_SAVE = 1 * (HORIZON < 10000) + 50 * (10000 <= HORIZON < 100000) + 100 * (HORIZON >= 100000)
DELTA_T_SAVE = 1  # XXX to disable this optimization

# REPETITIONS : number of repetitions of the experiments
# XXX Should be >= 10 to be stastically trustworthy
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
REPETITIONS = 1000
REPETITIONS = 200
REPETITIONS = 100
REPETITIONS = 50
REPETITIONS = 20
# REPETITIONS = 1  # XXX To profile the code, turn down parallel computing

DO_PARALLEL = False  # XXX do not let this = False  # To profile the code, turn down parallel computing
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1) and DO_PARALLEL
N_JOBS = -1 if DO_PARALLEL else 1
if CPU_COUNT > 4:  # We are on a server, let's be nice and not use all cores
    N_JOBS = max(int(CPU_COUNT / 2), CPU_COUNT - 4)

# Random events
RANDOM_SHUFFLE = False
RANDOM_INVERT = False
NB_RANDOM_EVENTS = 5

# Cache rewards
CACHE_REWARDS = True

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
DECREASE_RATE = 'auto'  # FIXED using the formula from Theorem 4.2 from [Bubeck & Cesa-Bianchi, 2012]

TEST_AGGR = False  # XXX do not let this = False if you want to test my Aggr policy
TEST_AGGR = True

UPDATE_ALL_CHILDREN = True
UPDATE_ALL_CHILDREN = False  # XXX do not let this = False

# UNBIASED is a flag to know if the rewards are used as biased estimator, ie just r_t, or unbiased estimators, r_t / p_t
UNBIASED = True
UNBIASED = False

# Flag to know if we should update the trusts proba like in Exp4 or like in my initial Aggr proposal
UPDATE_LIKE_EXP4 = True     # trusts^(t+1) = exp(rate_t * estimated rewards upto time t)
UPDATE_LIKE_EXP4 = False    # trusts^(t+1) <-- trusts^t * exp(rate_t * estimate reward at time t)


# Parameters for the arms
VARIANCE = 0.05   # Variance of Gaussian arms


# XXX This dictionary configures the experiments
configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # --- DELTA_T_SAVE
    "delta_t_save": DELTA_T_SAVE,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Random events
    "random_shuffle": RANDOM_SHUFFLE,
    "random_invert": RANDOM_INVERT,
    "nb_random_events": NB_RANDOM_EVENTS,
    # --- Cache rewards
    "cache_rewards": CACHE_REWARDS,
    # --- Arms
    # "environment": [  # Bernoulli arms
    #     # {   # The easier problem: 2 arms, one perfectly bad, one perfectly good
    #     #     "arm_type": Bernoulli,
    #     #     "params": [0, 1]
    #     # },
    #     # {   # A very very easy problem: 3 arms, one bad, one average, one good
    #     #     "arm_type": Bernoulli,
    #     #     "params": [0.1, 0.5, 0.9]
    #     # },
    #     # {   # Another very easy problem: 3 arms, two very bad, one bad
    #     #     "arm_type": Bernoulli,
    #     #     "params": [0.04, 0.05, 0.1]
    #     # },
    #     {   # A very easy problem, but it is used in a lot of articles
    #         "arm_type": Bernoulli,
    #         "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #     },
    #     # {   # An other problem, best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3 - 0.6) and very good arms (0.78, 0.8, 0.82)
    #     #     "arm_type": Bernoulli,
    #     #     "params": [0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.78, 0.8, 0.82]
    #     # },
    #     # {   # Lots of bad arms, significative difference between the best and the others
    #     #     "arm_type": Bernoulli,
    #     #     "params": [0.001, 0.001, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.3]
    #     # },
    #     # {   # One optimal arm, much better than the others, but *lots* of bad arms
    #     #     "arm_type": Bernoulli,
    #     #     "params": [0.001, 0.001, 0.001, 0.001, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2, 0.5]
    #     # },
    #     # {   # An other problem (17 arms), best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3, 0.6) and very good arms (0.78, 0.85)
    #     #     "arm_type": Bernoulli,
    #     #     "params": [0.005, 0.01, 0.015, 0.02, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.78, 0.8, 0.82, 0.83, 0.84, 0.85]
    #     # },
    # ],
    # # DONE I tried with other arms distribution: Exponential, it works similarly
    # # XXX if using Exponential arms, gives klExp to KL-UCB-like policies!
    # "environment": [  # Exponential arms
    #     {   # An example problem with  arms
    #         "arm_type": Exponential,
    #         "params": [2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     },
    # ],
    # DONE I tried with other arms distribution: Gaussian, it works similarly
    # XXX if using Gaussian arms, gives klGauss to KL-UCB-like policies!
    "environment": [  # Gaussian arms
        {   # An example problem with  arms
            "arm_type": Gaussian,
            "params": [(0.1, VARIANCE), (0.2, VARIANCE), (0.3, VARIANCE), (0.4, VARIANCE), (0.5, VARIANCE), (0.6, VARIANCE), (0.7, VARIANCE), (0.8, VARIANCE), (0.9, VARIANCE)]
        },
    ],
}

if len(configuration['environment']) > 1:
    raise ValueError("WARNING do not use this hack if you try to use more than one environment.")
    # Note: I dropped the support for more than one environments, for this part of the configuration, but not the simulation code
nbArms = len(configuration['environment'][0]['params'])
klucb = klucb_mapping.get(str(configuration['environment'][0]['arm_type']), klucbBern)


configuration.update({
    "policies": [
        # --- KL algorithms
        # --- klUCB
        {
            "archtype": klUCB,
            "params": {
                "klucb": klucbBern
            }
        },
        {
            "archtype": klUCB,
            "params": {
                "klucb": klucbExp
            }
        },
        {
            "archtype": klUCB,
            "params": {
                "klucb": klucbGauss
            }
        },
        # --- klUCBlog10
        {
            "archtype": klUCBlog10,
            "params": {
                "klucb": klucbBern
            }
        },
        {
            "archtype": klUCBlog10,
            "params": {
                "klucb": klucbExp
            }
        },
        {
            "archtype": klUCBlog10,
            "params": {
                "klucb": klucbGauss
            }
        },
        # --- klUCBloglog
        {
            "archtype": klUCBloglog,
            "params": {
                "klucb": klucbBern
            }
        },
        {
            "archtype": klUCBloglog,
            "params": {
                "klucb": klucbExp
            }
        },
        {
            "archtype": klUCBloglog,
            "params": {
                "klucb": klucbGauss
            }
        },
        # --- klUCBPlus
        {
            "archtype": klUCBPlus,
            "params": {
                "klucb": klucbBern
            }
        },
        {
            "archtype": klUCBPlus,
            "params": {
                "klucb": klucbExp
            }
        },
        {
            "archtype": klUCBPlus,
            "params": {
                "klucb": klucbGauss
            }
        },
    ]
})

# # XXX Only test with fixed arms
# configuration.update({
#     "policies": [  # --- Full or partial knowledge algorithms
#         TakeFixedArm(nbArms, k) for k in range(nbArms)
#     ]
# })

# # XXX Only test with scenario 1 from [A.Beygelzimer, J.Langfor, L.Li et al, AISTATS 2011]
# from PoliciesMultiPlayers import Scenario1  # XXX remove after testing once
# NB_PLAYERS = 10
# configuration.update({
#     "policies": Scenario1(NB_PLAYERS, nbArms).childs
# })


# from itertools import product  # XXX If needed!

# Dynamic hack to force the Aggr (policies aggregator) to use all the policies previously/already defined
if TEST_AGGR:
    # print("configuration['policies'] =", CURRENT_POLICIES)  # DEBUG
    NON_AGGR_POLICIES = configuration["policies"]
    # for LEARNING_RATE in LEARNING_RATES:  # XXX old code to test different static learning rates, not any more
    # for UNBIASED in [False, True]:  # XXX to test between biased or unabiased estimators
    # for (UNBIASED, UPDATE_LIKE_EXP4) in product([False, True], repeat=2):  # XXX If needed!
    # for (HORIZON, UPDATE_LIKE_EXP4) in product([None, HORIZON], [False, True]):  # XXX If needed!
    for UPDATE_LIKE_EXP4 in [False, True]:
        CURRENT_POLICIES = configuration["policies"]
        # Add one Aggr policy
        configuration["policies"] = [{
            "archtype": Aggr,
            "params": {
                "unbiased": UNBIASED,
                "update_all_children": UPDATE_ALL_CHILDREN,
                "decreaseRate": DECREASE_RATE,
                "learningRate": LEARNING_RATE,
                "children": NON_AGGR_POLICIES,
                "update_like_exp4": UPDATE_LIKE_EXP4,
                # "horizon": HORIZON  # XXX uncomment to give the value of horizon to have a better learning rate
            },
        }] + CURRENT_POLICIES

print("Loaded experiments configuration from 'configuration.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
