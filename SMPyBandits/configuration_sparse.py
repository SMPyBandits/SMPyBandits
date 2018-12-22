# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for single-player sparse bandit.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

# Tries to know number of CPU
try:
    from multiprocessing import cpu_count
    CPU_COUNT = cpu_count()
except ImportError:
    CPU_COUNT = 1

from os import getenv

if __name__ == '__main__':
    print("Warning: this script 'configuration_sparse.py' is NOT executable. Use 'main.py configuration_sparse' or 'make sparse' ...")  # DEBUG
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
REPETITIONS = 1000
# REPETITIONS = 200
REPETITIONS = 100
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

# Random events
RANDOM_SHUFFLE = False  #: The arms are shuffled (``shuffle(arms)``).
RANDOM_INVERT = False  #: The arms are inverted (``arms = arms[::-1]``).
NB_RANDOM_EVENTS = 5  #: Number of random events. They are uniformly spaced in time steps.

#: Should the Aggregator policy update the trusts in each child or just the one trusted for last decision?
UPDATE_ALL_CHILDREN = True
UPDATE_ALL_CHILDREN = False  # XXX do not let this = False

#: Learning rate for my aggregated bandit (it can be autotuned)
LEARNING_RATE = 0.01

#: Constant time tau for the decreasing rate for my aggregated bandit.
# FIXED I tried to make self.learningRate decrease when self.t increase, it was not better
DECREASE_RATE = None
DECREASE_RATE = HORIZON / 2.0
DECREASE_RATE = 'auto'  # FIXED using the formula from Theorem 4.2 from [Bubeck & Cesa-Bianchi, 2012](http://sbubeck.com/SurveyBCB12.pdf)

#: Should the rewards for Aggregator policy use as biased estimator, ie just ``r_t``, or unbiased estimators, ``r_t / p_t``
UNBIASED = True
UNBIASED = False

#: Should we update the trusts proba like in Exp4 or like in my initial Aggregator proposal
UPDATE_LIKE_EXP4 = True     # trusts^(t+1) = exp(rate_t * estimated rewards upto time t)
UPDATE_LIKE_EXP4 = False    # trusts^(t+1) <-- trusts^t * exp(rate_t * estimate reward at time t)

#: To know if my Aggregator policy is tried.
TEST_Aggregator = True
TEST_Aggregator = False  # XXX do not let this = False if you want to test my Aggregator policy

#: Should we cache rewards? The random rewards will be the same for all the REPETITIONS simulations for each algorithms.
CACHE_REWARDS = False  # XXX to disable manually this feature
CACHE_REWARDS = TEST_Aggregator


# Parameters for the arms
TRUNC = 1  #: Trunc parameter, ie amplitude, for Exponential arms

VARIANCE = 0.05   #: Variance of Gaussian arms
MINI = 0  #: lower bound on rewards from Gaussian arms
MAXI = 1  #: upper bound on rewards from Gaussian arms, ie amplitude = 1

SCALE = 1   #: Scale of Gamma arms

# --- Parameters for the sparsity

#: Number of arms for non-hard-coded problems (Bayesian problems)
NB_ARMS = 15
NB_ARMS = int(getenv('K', NB_ARMS))
NB_ARMS = int(getenv('NB_ARMS', NB_ARMS))

#: Sparsity for non-hard-coded problems (Bayesian problems)
SPARSITY = 7
SPARSITY = int(getenv('S', SPARSITY))
SPARSITY = int(getenv('SPARSITY', SPARSITY))

#: Default value for the lower value of means
LOWER = 0.
#: Default value for the lower value of non-zero means
LOWERNONZERO = 0.25
#: Default value for the amplitude value of means
AMPLITUDE = 1.

#: Type of arms for non-hard-coded problems (Bayesian problems)
ARM_TYPE = "Gaussian"
ARM_TYPE = str(getenv('ARM_TYPE', ARM_TYPE))

# WARNING That's nonsense, rewards of unbounded distributions just don't have lower, amplitude values...
if ARM_TYPE in [
            "UnboundedGaussian",
            # "Gaussian",
        ]:
    LOWER = -5
    AMPLITUDE = 10

LOWER = float(getenv('LOWER', LOWER))
LOWERNONZERO = float(getenv('LOWERNONZERO', LOWERNONZERO))
AMPLITUDE = float(getenv('AMPLITUDE', AMPLITUDE))
assert AMPLITUDE > 0, "Error: invalid amplitude = {:.3g} but has to be > 0."  # DEBUG
VARIANCE = float(getenv('VARIANCE', VARIANCE))

ARM_TYPE_str = str(ARM_TYPE)
ARM_TYPE = mapping_ARM_TYPE[ARM_TYPE]

#: True to use bayesian problem
ENVIRONMENT_BAYESIAN = False
ENVIRONMENT_BAYESIAN = getenv('BAYES', str(ENVIRONMENT_BAYESIAN)) == 'True'

#: Means of arms for non-hard-coded problems (non Bayesian)
MEANS = uniformMeansWithSparsity(nbArms=NB_ARMS, sparsity=SPARSITY, delta=0.005, lower=LOWER, lowerNonZero=LOWERNONZERO, amplitude=AMPLITUDE, isSorted=True)

import numpy as np
# MEANS = [0.05] * (NB_ARMS - SPARSITY) + list(np.linspace(LOWERNONZERO, LOWER + AMPLITUDE, num=SPARSITY))
# more parametric? Read from cli?
MEANS_STR = getenv('MEANS', '')
if MEANS_STR:
    MEANS_STR = MEANS_STR.replace('[', '').replace(']', '')
    MEANS = np.asarray([ float(m) for m in MEANS_STR.split(',') ], dtype=float)

#: Whether to sort the means of the problems or not.
ISSORTED = False
ISSORTED = True


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
    # --- Arms
    "environment": [  # 1)  Bernoulli arms
        # {   # A easy problem, but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        #     "sparsity": SPARSITY,
        # },
        # {   # A very easy problem, but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "params": MEANS,
        #     "sparsity": SPARSITY,
        # },
        # "environment": [  # 2)  custom arms
        {   # A very easy problem, but it is used in a lot of articles
            "arm_type": ARM_TYPE,
            "params": MEANS,
            "sparsity": SPARSITY,
        },
        # # "environment": [  # 3)  Gaussian arms
        # {   # A very easy problem, but it is used in a lot of articles
        #     "arm_type": Gaussian,
        #     "params": MEANS,
        #     "sparsity": SPARSITY,
        # },
        # {   # An example problem with 3 or 9 arms
        #     "arm_type": Gaussian,
        #     # "params": [(mean, VARIANCE, MINI, MAXI) for mean in list(range(-8, 10, 2))],
        #     # "params": [(mean, VARIANCE) for mean in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
        #     "params": [(mean, VARIANCE) for mean in MEANS],
        #     # "params": [(mean, VARIANCE) for mean in [0.1, 0.5, 0.9]],
        #     "sparsity": SPARSITY,
        # },
        # {   # A non-Bayesian random problem
        #     "arm_type": ARM_TYPE,
        #     "params": randomMeansWithSparsity(NB_ARMS, SPARSITY, mingap=None, lower=0., lowerNonZero=0.2, amplitude=1., isSorted=True),
        #     "sparsity": SPARSITY,
        # },
        # # FIXED I need to do Bayesian problems for Gaussian arms also!
        # {   # A Bayesian problem: every repetition use a different mean vectors!
        #     "arm_type": ARM_TYPE,
        #     "params": {
        #         "function": randomMeansWithSparsity,
        #         "args": {
        #             "nbArms": NB_ARMS,
        #             "mingap": None,
        #             # "mingap": 0.1,
        #             # "mingap": 1. / (3 * NB_ARMS),
        #             "lower": -2.,
        #             "lowerNonZero": 2,
        #             "amplitude": 4.,
        #             "isSorted": ISSORTED,
        #             "sparsity": SPARSITY,
        #         }
        #     },
        #     "sparsity": SPARSITY,
        # },
    ],
}

if ENVIRONMENT_BAYESIAN:
    configuration["environment"] = [  # XXX Bernoulli arms
        {   # A Bayesian problem: every repetition use a different mean vectors!
            "arm_type": ARM_TYPE,
            "params": {
                "function": randomMeansWithSparsity,
                "args": {
                    "nbArms": NB_ARMS,
                    "mingap": None,
                    # "mingap": 0.0000001,
                    # "mingap": 0.1,
                    # "mingap": 1. / (3 * NB_ARMS),
                    "lower": LOWER,
                    "lowerNonZero": LOWERNONZERO,
                    "amplitude": AMPLITUDE,
                    "isSorted": ISSORTED,
                    "sparsity": SPARSITY,
                }
            },
            "sparsity": SPARSITY,
        },
    ]
elif ARM_TYPE_str in ["Gaussian", "UnboundedGaussian"]:
    from Policies.OSSB import solve_optimization_problem__sparse_bandits
    means = uniformMeansWithSparsity(nbArms=NB_ARMS, sparsity=SPARSITY, delta=0.2, lower=LOWER, lowerNonZero=LOWERNONZERO, amplitude=AMPLITUDE, isSorted=ISSORTED)
    for s in [SPARSITY-1, SPARSITY, SPARSITY+1]:
        solve_optimization_problem__sparse_bandits(means, sparsity=s, only_strong_or_weak=True)

    configuration.update({
        "environment": [ {
                "arm_type": ARM_TYPE,
                "params": [
                    (mu, VARIANCE, LOWER, LOWER+AMPLITUDE)
                    for mu in means
                ],
                "sparsity": SPARSITY,
        }, ],
    })
# else:
#     configuration.update({
#         "environment": [ {
#                 "arm_type": ARM_TYPE,
#                 "params": uniformMeans(nbArms=NB_ARMS, delta=1./(1. + NB_ARMS), lower=LOWER, amplitude=AMPLITUDE),
#                 "sparsity": SPARSITY,
#             }, ],
#     })

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
    """klucbGauss(x, d, sig2x) with the good variance (= 0.25)."""
    return _klucbGauss(x, d, 0.25)
    # return _klucbGauss(x, d, VARIANCE)


_klucbGamma = klucbGamma


def klucbGamma(x, d, precision=0.):
    """klucbGamma(x, d, sig2x) with the good scale (= 1)."""
    return _klucbGamma(x, d, SCALE)


configuration.update({
    "policies": [
        # --- Naive algorithms
        {
            "archtype": EmpiricalMeans,
            "params": {
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # {
        #     "archtype": EpsilonDecreasing,
        #     "params": {
        #         "epsilon": 1. / (2 * nbArms),
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # --- UCBalpha algorithm
        {
            "archtype": UCBalpha,
            "params": {
                "alpha": 1,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # # --- SparseUCB algorithm
        {
            "archtype": SparseUCB,
            "params": {
                "alpha": 1,
                "sparsity": SPARSITY,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # # --- DONE SparseUCB algorithm with a too small value for s
        # # XXX It fails completely!
        # {
        #     "archtype": SparseUCB,
        #     "params": {
        #         "alpha": 1,
        #         "sparsity": max(SPARSITY - 1, 1),
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # # --- DONE SparseUCB algorithm with a larger value for s
        # # XXX It fails completely!
        # {
        #     "archtype": SparseUCB,
        #     "params": {
        #         "alpha": 1,
        #         "sparsity": min(SPARSITY + 1, NB_ARMS),
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # --- KL algorithms, here only klUCB with different klucb functions
        {
            "archtype": klUCB,
            "params": {
                "klucb": klucbBern,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # {
        #     "archtype": klUCB,
        #     "params": {
        #         "klucb": klucbGauss,  # XXX exactly like UCB !
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # # --- Finite-Horizon Gittins index
        # {
        #     "archtype": ApproximatedFHGittins,
        #     "params": {
        #         "horizon": 1.05 * HORIZON,
        #         "alpha": 1,
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # # --- SparseUCB algorithm
        {
            "archtype": SparseklUCB,
            "params": {
                "sparsity": SPARSITY,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # {
        #     "archtype": SparseWrapper,
        #     "params": {
        #         "sparsity": SPARSITY,
        #         "policy": klUCB,
        #         "klucb": klucbGauss,
        #         "use_ucb_for_set_J": True,
        #         "use_ucb_for_set_K": False,
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # {
        #     "archtype": SparseWrapper,
        #     "params": {
        #         "sparsity": SPARSITY,
        #         "policy": klUCB,
        #         "klucb": klucbGauss,
        #         "use_ucb_for_set_J": False,
        #         "use_ucb_for_set_K": True,
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # {
        #     "archtype": SparseWrapper,
        #     "params": {
        #         "sparsity": SPARSITY,
        #         "policy": klUCB,
        #         "klucb": klucbGauss,
        #         "use_ucb_for_set_J": False,
        #         "use_ucb_for_set_K": False,
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # --- Thompson algorithm
        {
            "archtype": Thompson,
            "params": {
                "posterior": Beta,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # --- SparseWrapper algorithm, 4 different versions whether using old UCB for sets J(t) and K(t) or not
        {
            "archtype": SparseWrapper,
            "params": {
                "sparsity": SPARSITY,
                "policy": Thompson,
                "posterior": Beta,
                "use_ucb_for_set_J": True,
                "use_ucb_for_set_K": True,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # --- Thompson algorithm, with Gaussian posterior
        {
            "archtype": Thompson,
            "params": {
                "posterior": Gauss,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # --- SparseWrapper algorithm, 4 different versions whether using old UCB for sets J(t) and K(t) or not
        {
            "archtype": SparseWrapper,
            "params": {
                "sparsity": SPARSITY,
                "policy": Thompson,
                "posterior": Gauss,   # WARNING Gaussian posterior is still experimental and VERY slow
                "use_ucb_for_set_J": True,
                "use_ucb_for_set_K": True,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # --- BayesUCB algorithm
        {
            "archtype": BayesUCB,
            "params": {
                "posterior": Beta,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        {
            "archtype": SparseWrapper,
            "params": {
                "sparsity": SPARSITY,
                "policy": BayesUCB,
                "posterior": Beta,
                "use_ucb_for_set_J": True,
                "use_ucb_for_set_K": True,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        {
            "archtype": BayesUCB,
            "params": {
                "posterior": Gauss,  # XXX does not work yet!
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        {
            "archtype": SparseWrapper,
            "params": {
                "sparsity": SPARSITY,
                "posterior": Gauss,  # XXX does not work yet!
                "policy": BayesUCB,
                "use_ucb_for_set_J": True,
                "use_ucb_for_set_K": True,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # --- The new OSSB algorithm
        {
            "archtype": OSSB,
            "params": {
                "epsilon": 0.0,  # XXX test to change these values!
                "gamma": 0.0,  # XXX test to change these values!
            }
        },
        # --- FIXME The new OSSB algorithm, tuned for Gaussian bandits
        {
            "archtype": GaussianOSSB,
            "params": {
                "epsilon": 0.0,
                "gamma": 0.0,
                "variance": VARIANCE,
            }
        },
        # --- FIXME The new OSSB algorithm, tuned for Sparse bandits
        {
            "archtype": SparseOSSB,
            "params": {
                "epsilon": 0.0,
                "gamma": 0.0,
                "sparsity": SPARSITY,
            }
        },
        {
            "archtype": SparseOSSB,
            "params": {
                "epsilon": 0.001,
                "gamma": 0.0,
                "sparsity": SPARSITY,
            }
        },
        {
            "archtype": SparseOSSB,
            "params": {
                "epsilon": 0.0,
                "gamma": 0.01,
                "sparsity": SPARSITY,
            }
        },
        {
            "archtype": SparseOSSB,
            "params": {
                "epsilon": 0.001,
                "gamma": 0.01,
                "sparsity": SPARSITY,
            }
        },
    ]
})


NON_AGGR_POLICIES_1 = [
    {
        "archtype": SparseWrapper,
        "params": {
            "policy": klUCB,
            "sparsity": s,
            "use_ucb_for_set_J": True, "use_ucb_for_set_K": True,
            "lower": LOWER, "amplitude": AMPLITUDE,
        }
    }
    for s in [SPARSITY - 1, SPARSITY, SPARSITY + 1]
    # for s in range(1, 1 + NB_ARMS)
]


# Dynamic hack to force the Aggregator (policies aggregator) to use all the policies previously/already defined
if TEST_Aggregator:
    NON_AGGR_POLICIES_0 = configuration["policies"]
    # XXX Very simulation-specific settings!
    EXTRA_STRS = ["[all non Aggr]", "[Sparse-KLUCB for s={}..{}]".format(1, NB_ARMS)]

    for NON_AGGR_POLICIES, EXTRA_STR in zip(
            [NON_AGGR_POLICIES_0, NON_AGGR_POLICIES_1],
            EXTRA_STRS
        ):
        for UPDATE_LIKE_EXP4 in [False, True]:
            CURRENT_POLICIES = configuration["policies"]
            print("configuration['policies'] =", CURRENT_POLICIES)  # DEBUG
            # Add one Aggregator policy
            configuration["policies"] = CURRENT_POLICIES + [{
                "archtype": Aggregator,
                "params": {
                    "unbiased": UNBIASED,
                    "update_all_children": UPDATE_ALL_CHILDREN,
                    "decreaseRate": DECREASE_RATE,
                    "learningRate": LEARNING_RATE,
                    "children": NON_AGGR_POLICIES,
                    "update_like_exp4": UPDATE_LIKE_EXP4,
                    "extra_str": EXTRA_STR,
                    # "horizon": HORIZON  # XXX uncomment to give the value of horizon to have a better learning rate
                },
            }]


print("Loaded experiments configuration from 'configuration_sparse.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
