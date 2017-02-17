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
HORIZON = 20000
# HORIZON = 30000
# HORIZON = 100000

# DELTA_T_SAVE : save only 1 / DELTA_T_SAVE points, to speed up computations, use less RAM, speed up plotting etc.
DELTA_T_SAVE = 1 * (HORIZON < 10000) + 50 * (10000 <= HORIZON < 100000) + 100 * (HORIZON >= 100000)
DELTA_T_SAVE = 1  # XXX to disable this optimization

# REPETITIONS : number of repetitions of the experiments
# XXX Should be >= 10 to be stastically trustworthy
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
# REPETITIONS = 1000
REPETITIONS = 200
REPETITIONS = 100
REPETITIONS = 50
# REPETITIONS = 20
# REPETITIONS = 1  # XXX To profile the code, turn down parallel computing

DO_PARALLEL = False  # XXX do not let this = False  # To profile the code, turn down parallel computing
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1) and DO_PARALLEL
N_JOBS = -1 if DO_PARALLEL else 1
if CPU_COUNT > 4:  # We are on a server, let's be nice and not use all cores
    N_JOBS = max(int(CPU_COUNT / 2), CPU_COUNT - 4)

# Random events
RANDOM_SHUFFLE = True
RANDOM_INVERT = False
NB_RANDOM_EVENTS = 5

# Cache rewards
CACHE_REWARDS = True

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
TRUNC = 1  # Trunc parameter, ie amplitude, for Exponential arms

VARIANCE = 0.05   # Variance of Gaussian arms
MINI = 0  # lower bound on rewards from Gaussian arms
MAXI = 1  # upper bound on rewards from Gaussian arms, ie amplitude = 20


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
    "environment": [  # 1)  Bernoulli arms
        {   # A very easy problem, but it is used in a lot of articles
            "arm_type": Bernoulli,
            "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        {   # An other problem, best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3 - 0.6) and very good arms (0.78, 0.8, 0.82)
            "arm_type": Bernoulli,
            "params": [0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.795, 0.8, 0.805]
        },
    # ],
    # "environment": [  # 2)  Exponential arms
        {   # An example problem with  arms
            "arm_type": Exponential,
            "params": [(2, TRUNC), (3, TRUNC), (4, TRUNC), (5, TRUNC), (6, TRUNC), (7, TRUNC), (8, TRUNC), (9, TRUNC), (10, TRUNC)]
        },
    # ],
    # "environment": [  # 3)  Gaussian arms
        {   # An example problem with  arms
            "arm_type": Gaussian,
            # "params": [(mean, VARIANCE, MINI, MAXI) for mean in list(range(-8, 10, 2))]
            "params": [(mean, VARIANCE) for mean in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        },
    ],
}

# if len(configuration['environment']) > 1:
#     raise ValueError("WARNING do not use this hack if you try to use more than one environment.")
#     # Note: I dropped the support for more than one environments, for this part of the configuration, but not the simulation code


# And get LOWER, AMPLITUDE values
LOWER, AMPLITUDE = 0, 1
try:
    for env in configuration['environment']:
        nbArms = len(env['params'])
        arm_type = env['arm_type']
        for param in env['params']:
            arm = arm_type(*param) if isinstance(param, (dict, tuple, list)) else arm_type(param)
            l, a = arm.lower_amplitude
            LOWER = min(LOWER, l)
            AMPLITUDE = max(AMPLITUDE, a)
    mini, maxi = LOWER, LOWER + AMPLITUDE
    print("Apparently, the arms have rewards in [{}, {}] (lower = {}, amplitude = {})".format(LOWER, LOWER + AMPLITUDE, LOWER, AMPLITUDE))
except Exception as e:
    print("Warning: Possibly wrong estimate of lower, amplitude ....")


# Custom klucb function
_klucbGauss = klucbGauss
def klucbGauss(x, d, precision=0.):
    """klucbGauss(x, d, sig2) with the good variance (= 0.05)."""
    return _klucbGauss(x, d, VARIANCE)
    # return _klucbGauss(x, d, 1.0)


configuration.update({
    "policies": [
        # --- Thompson algorithm
        {
            "archtype": Thompson,
            "params": {
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # --- BayesUCB algorithm
        {
            "archtype": BayesUCB,
            "params": {
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # --- KL algorithms, here only klUCBPlus with different klucb functions
        {
            "archtype": klUCBPlus,
            "params": {
                "lower": LOWER, "amplitude": AMPLITUDE,
                "klucb": klucbBern
            }
        },
        {
            "archtype": klUCBPlus,
            "params": {
                "lower": LOWER, "amplitude": AMPLITUDE,
                "klucb": klucbExp
            }
        },
        {
            "archtype": klUCBPlus,
            "params": {
                "lower": LOWER, "amplitude": AMPLITUDE,
                "klucb": klucbGauss
            }
        },
    ]
})


# Dynamic hack to force the Aggr (policies aggregator) to use all the policies previously/already defined
if TEST_AGGR:
    NON_AGGR_POLICIES = configuration["policies"]
    for UPDATE_LIKE_EXP4 in [False, True]:
        CURRENT_POLICIES = configuration["policies"]
        # Add one Aggr policy
        configuration["policies"] = [{
            "archtype": Aggr,
            "params": {
                "unbiased": UNBIASED,
                "update_all_children": UPDATE_ALL_CHILDREN,
                "decreaseRate": "auto",
                "children": NON_AGGR_POLICIES,
                "update_like_exp4": UPDATE_LIKE_EXP4
            },
        }] + CURRENT_POLICIES

print("Loaded experiments configuration from 'configuration.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
