# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the single-player case, for comparing Aggregation algorithms.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.7"

# Tries to know number of CPU
try:
    from multiprocessing import cpu_count
    CPU_COUNT = cpu_count()
except ImportError:
    CPU_COUNT = 1

from os import getenv

if __name__ == '__main__':
    print("Warning: this script 'configuration_comparing_KLUCB_aggregation.py' is NOT executable. Use 'main.py configuration_comparing_KLUCB_aggregation' or 'make comparing_KLUCB_aggregation' ...")  # DEBUG
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
# HORIZON = 20000
# HORIZON = 30000
# # # HORIZON = 40000
# HORIZON = 100000
HORIZON = int(getenv('T', HORIZON))

#: REPETITIONS : number of repetitions of the experiments.
#: Warning: Should be >= 10 to be statistically trustworthy.
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
# REPETITIONS = 1000
# REPETITIONS = 200
# REPETITIONS = 100
# REPETITIONS = 50
# REPETITIONS = 20
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

#: Number of arms for non-hard-coded problems (Bayesian problems)
NB_ARMS = 9
NB_ARMS = int(getenv('K', NB_ARMS))
NB_ARMS = int(getenv('NB_ARMS', NB_ARMS))

# Random events
RANDOM_SHUFFLE = False  #: The arms are shuffled (``shuffle(arms)``).
RANDOM_INVERT = False  #: The arms are inverted (``arms = arms[::-1]``).
NB_RANDOM_EVENTS = 5  #: Number of random events. They are uniformly spaced in time steps.

TEST_Aggregator = False  # XXX do not let this = False if you want to test my Aggregator policy
TEST_Aggregator = True

TEST_CORRAL = False  # XXX do not let this = False if you want to test the CORRAL policy
TEST_CORRAL = True

TEST_LEARNEXP = False  # XXX do not let this = False if you want to test the LearnExp policy
TEST_LEARNEXP = True

TEST_HEDGE = False  # XXX do not let this = False if you want to test the Hedge policy
TEST_HEDGE = True

#: Should we cache rewards? The random rewards will be the same for all the REPETITIONS simulations for each algorithms.
CACHE_REWARDS = TEST_Aggregator or TEST_CORRAL or TEST_LEARNEXP or TEST_HEDGE
CACHE_REWARDS = False  # XXX to disable manually this feature

#: Should the Aggregator policy update the trusts in each child or just the one trusted for last decision?
UPDATE_ALL_CHILDREN = True
UPDATE_ALL_CHILDREN = False  # XXX do not let this = False

#: Should the rewards for Aggregator policy use as biased estimator, ie just ``r_t``, or unbiased estimators, ``r_t / p_t``
UNBIASED = True
UNBIASED = False

#: Should we update the trusts proba like in Exp4 or like in my initial Aggregator proposal
UPDATE_LIKE_EXP4 = True     # trusts^(t+1) = exp(rate_t * estimated rewards upto time t)
UPDATE_LIKE_EXP4 = False    # trusts^(t+1) <-- trusts^t * exp(rate_t * estimate reward at time t)


# Parameters for the arms
TRUNC = 1  #: Trunc parameter, ie amplitude, for Exponential arms

VARIANCE = 0.05   #: Variance of Gaussian arms
# VARIANCE = 0.25   #: Variance of Gaussian arms
MINI = 0  #: lower bound on rewards from Gaussian arms
MAXI = 1  #: upper bound on rewards from Gaussian arms, ie amplitude = 1

SCALE = 1   #: Scale of Gamma arms

#: Type of arms for non-hard-coded problems (Bayesian problems)
ARM_TYPE = "Bernoulli"
ARM_TYPE = str(getenv('ARM_TYPE', ARM_TYPE))
mapping_ARM_TYPE = {
    "Constant": Constant,
    "Uniform": UniformArm,
    "Bernoulli": Bernoulli, "B": Bernoulli,
    "Gaussian": Gaussian, "Gauss": Gaussian, "G": Gaussian,
    "Poisson": Poisson, "P": Poisson,
    "Exponential": ExponentialFromMean, "Exp": ExponentialFromMean, "E": ExponentialFromMean,
    "Gamma": GammaFromMean,
}
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
    # --- Random events
    "random_shuffle": RANDOM_SHUFFLE,
    "random_invert": RANDOM_INVERT,
    "nb_random_events": NB_RANDOM_EVENTS,
    # --- Cache rewards: use the same random rewards for the Aggregator[..] and the algorithms
    "cache_rewards": CACHE_REWARDS,
    # --- Arms
    "environment": [  # 1)  Bernoulli arms
        # {   # A very easy problem, but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.5, 0.9]
        # },
        # {   # A easy problem, but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # },
        # {   # An other problem, best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3 - 0.6) and very good arms (0.78, 0.8, 0.82)
        #     "arm_type": Bernoulli,
        #     "params": [0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.795, 0.8, 0.805]
        # },
        # {   # A very hard problem, as used in [Cappé et al, 2012]
        #     "arm_type": Bernoulli,
        #     "params": [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.1]
        # },
        # XXX Default!
        {   # A very easy problem (X arms), but it is used in a lot of articles
            "arm_type": ARM_TYPE,
            "params": uniformMeans(NB_ARMS, 1 / (1. + NB_ARMS))
        }
    # ],
    # # "environment": [  # 2)  Exponential arms
    #     {   # An example problem with 9 arms
    #         "arm_type": Exponential,
    #         "params": [(2, TRUNC), (3, TRUNC), (4, TRUNC), (5, TRUNC), (6, TRUNC), (7, TRUNC), (8, TRUNC), (9, TRUNC), (10, TRUNC)]
    #     },
    # # ],
    # # "environment": [  # 3)  Gaussian arms
    #     {   # An example problem with 3 or 9 arms
    #         "arm_type": Gaussian,
    #         # "params": [(mean, VARIANCE, MINI, MAXI) for mean in list(range(-8, 10, 2))]
    #         "params": [(mean, VARIANCE) for mean in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    #         # "params": [(mean, VARIANCE) for mean in [0.1, 0.5, 0.9]]
    #     },
    # # # "environment": [  # 4)  Mix between Bernoulli and Gaussian and Exponential arms
    #     [
    #         arm_type(mean)
    #         for mean in [0.1, 0.5, 0.9]
    #         for arm_type in [Bernoulli, lambda mean: Gaussian(mean, VARIANCE), ExponentialFromMean]
    #     ],
    # # "environment": [  # 5)  Mix between Bernoulli and Gaussian and Exponential arms
    #     [
    #         arm_type(mean)
    #         for mean in [0.01, 0.02, 0.09]
    #         for arm_type in [Bernoulli, lambda mean: Gaussian(mean, VARIANCE), ExponentialFromMean]
    #     ],
    # # ],
    # # "environment": [  # FIXME Gamma arms
    # #     {   # An example problem with 3 arms
    # #         "arm_type": GammaFromMean,
    # #         "params": [(shape, SCALE, 0, 10) for shape in [1, 2, 3, 4, 5]]
    # #     },
    # # ],
    ],
}

# if len(configuration['environment']) > 1:
#     raise ValueError("WARNING do not use this hack if you try to use more than one environment.")
#     # Note: I dropped the support for more than one environments, for this part of the configuration, but not the simulation code


#: And get LOWER, AMPLITUDE values
LOWER, AMPLITUDE = 0, 1
try:
    for env in configuration['environment']:
        if isinstance(env, dict) and 'params' in env and 'arm_type' in env:
            nbArms = len(env['params'])
            arm_type = env['arm_type']
            for param in env['params']:
                arm = arm_type(*param) if isinstance(param, (dict, tuple, list)) else arm_type(param)
                l, a = arm.lower_amplitude
                LOWER = min(LOWER, l)
                AMPLITUDE = max(AMPLITUDE, a)
        else:  # the env must be a list of arm, already created
            for arm in env:
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
    return _klucbGauss(x, d, 0.25)
    # return _klucbGauss(x, d, VARIANCE)


_klucbGamma = klucbGamma


def klucbGamma(x, d, precision=0.):
    """klucbGamma(x, d, sig2) with the good scale (= 1)."""
    return _klucbGamma(x, d, SCALE)


configuration.update({
    "policies": [
        # {
        #     "archtype": Uniform,   # The stupidest policy, fully uniform
        #     "params": {}
        # },
        # --- UCBalpha algorithm
        # {
        #     "archtype": UCBalpha,
        #     "params": {
        #         "alpha": 4,
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        {
            "archtype": UCBalpha,
            "params": {
                "alpha": 1,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # {
        #     "archtype": UCBalpha,
        #     "params": {
        #         "alpha": 0.5,
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # --- Thompson algorithm
        {
            "archtype": Thompson,
            "params": {
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # --- KL algorithms, here only klUCBPlus with different klucb functions
        {
            "archtype": klUCB,
            "params": {
                "lower": LOWER, "amplitude": AMPLITUDE,
                "klucb": klucbBern,  # "horizon": HORIZON,
            }
        },
        {
            "archtype": klUCB,
            "params": {
                "lower": LOWER, "amplitude": AMPLITUDE,
                "klucb": klucbExp,  # "horizon": HORIZON,
            }
        },
        {
            "archtype": klUCB,
            "params": {
                "lower": LOWER, "amplitude": AMPLITUDE,
                "klucb": klucbGauss,  # "horizon": HORIZON,
            }
        },
        # {
        #     "archtype": klUCB,
        #     "params": {
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #         "klucb": klucbGamma,  # "horizon": HORIZON,
        #     }
        # },
        # --- BayesUCB algorithm
        {
            "archtype": BayesUCB,
            "params": {
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # # --- Finite-Horizon Gittins index
        # {
        #     "archtype": ApproximatedFHGittins,
        #     "params": {
        #         "horizon": 1.1 * HORIZON,
        #         "alpha": 2,
        #     }
        # },
        # {
        #     "archtype": ApproximatedFHGittins,
        #     "params": {
        #         "horizon": 1.1 * HORIZON,
        #         "alpha": 1,
        #     }
        # },
        # {
        #     "archtype": ApproximatedFHGittins,
        #     "params": {
        #         "horizon": 1.1 * HORIZON,
        #         "alpha": 0.5,
        #     }
        # },
    ]
})


from itertools import product  # XXX If needed!
NON_AGGR_POLICIES = configuration["policies"]
# NON_AGGR_POLICIES += [{
#     "archtype": Uniform,   # The stupidest policy, fully uniform
#     "params": {}
# }]


# Dynamic hack to force the LearnExp (policies aggregator) to use all the policies previously/already defined
if TEST_LEARNEXP:
    # ETA_VALUES = [0.2, 0.4, 0.6, 0.8]
    ETA_VALUES = [0.9]
    # UNBIASED_VALUES = [False, True]
    UNBIASED_VALUES = [True]
    for ETA in ETA_VALUES:
        for UNBIASED in UNBIASED_VALUES:
            CURRENT_POLICIES = configuration["policies"]
            # Add one LearnExp policy
            configuration["policies"] = [{
                "archtype": LearnExp,
                "params": {
                    "children": NON_AGGR_POLICIES,
                    "unbiased": UNBIASED,
                    "eta": ETA,
                },
            }] + CURRENT_POLICIES


# Dynamic hack to force the CORRAL (policies aggregator) to use all the policies previously/already defined
if TEST_CORRAL:
    # UNBIASED_VALUES = [False, True]
    UNBIASED_VALUES = [True]
    # BROADCAST_ALL_VALUES = [False, True]
    BROADCAST_ALL_VALUES = [True]
    for UNBIASED in UNBIASED_VALUES:
       for BROADCAST_ALL in BROADCAST_ALL_VALUES:
            CURRENT_POLICIES = configuration["policies"]
            # Add one CORRAL policy
            configuration["policies"] = [{
                "archtype": CORRAL,
                "params": {
                    "children": NON_AGGR_POLICIES,
                    "horizon": HORIZON,
                    "unbiased": UNBIASED,
                    "broadcast_all": BROADCAST_ALL,
                },
            }] + CURRENT_POLICIES


# Dynamic hack to force the Aggregator (policies aggregator) to use all the policies previously/already defined
if TEST_Aggregator:
    UPDATE_LIKE_EXP4_VALUES = [False, True]
    # UPDATE_LIKE_EXP4_VALUES = [True]
    # UPDATE_ALL_CHILDREN_VALUES = [False, True]
    UPDATE_ALL_CHILDREN_VALUES = [True]
    # for UPDATE_LIKE_EXP4 in UPDATE_LIKE_EXP4_VALUES:
    #    for UPDATE_ALL_CHILDREN in UPDATE_ALL_CHILDREN_VALUES:
    # for (UPDATE_LIKE_EXP4, UPDATE_ALL_CHILDREN) in [(True, False), (False, True), (False, False)]:
    for (UPDATE_LIKE_EXP4, UPDATE_ALL_CHILDREN) in [(True, False), (False, False)]:
            CURRENT_POLICIES = configuration["policies"]
            # Add one Aggregator policy
            configuration["policies"] = [{
                "archtype": Aggregator,
                "params": {
                    "children": NON_AGGR_POLICIES,
                    "unbiased": UNBIASED,
                    "update_all_children": UPDATE_ALL_CHILDREN,
                    "decreaseRate": "auto",
                    "update_like_exp4": UPDATE_LIKE_EXP4
                },
            }] + CURRENT_POLICIES


print("Loaded experiments configuration from 'configuration.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
