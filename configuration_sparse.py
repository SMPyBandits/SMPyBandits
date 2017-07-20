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
HORIZON = 20000
HORIZON = 30000
# # # HORIZON = 40000
# HORIZON = 100000

#: DELTA_T_SAVE : save only 1 / DELTA_T_SAVE points, to speed up computations, use less RAM, speed up plotting etc.
#: Warning: not perfectly finished right now.
DELTA_T_SAVE = 1 * (HORIZON < 10000) + 50 * (10000 <= HORIZON < 100000) + 100 * (HORIZON >= 100000)
DELTA_T_SAVE = 1  # XXX to disable this optimization

#: REPETITIONS : number of repetitions of the experiments.
#: Warning: Should be >= 10 to be stastically trustworthy.
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
# REPETITIONS = 1000
# REPETITIONS = 200
REPETITIONS = 100
# REPETITIONS = 50
# REPETITIONS = 20

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

#: Should the Aggragorn policy update the trusts in each child or just the one trusted for last decision?
UPDATE_ALL_CHILDREN = True
UPDATE_ALL_CHILDREN = False  # XXX do not let this = False

#: Should the rewards for Aggragorn policy use as biased estimator, ie just ``r_t``, or unbiased estimators, ``r_t / p_t``
UNBIASED = True
UNBIASED = False

#: Should we update the trusts proba like in Exp4 or like in my initial Aggragorn proposal
UPDATE_LIKE_EXP4 = True     # trusts^(t+1) = exp(rate_t * estimated rewards upto time t)
UPDATE_LIKE_EXP4 = False    # trusts^(t+1) <-- trusts^t * exp(rate_t * estimate reward at time t)


# Parameters for the arms
TRUNC = 1  #: Trunc parameter, ie amplitude, for Exponential arms

VARIANCE = 0.01   #: Variance of Gaussian arms
# VARIANCE = 0.25   #: Variance of Gaussian arms
MINI = 0  #: lower bound on rewards from Gaussian arms
MAXI = 1  #: upper bound on rewards from Gaussian arms, ie amplitude = 1

SCALE = 1   #: Scale of Gamma arms

# --- Parameters for the sparsity
NB_ARMS = 15
SPARSITY = 4
# MEANS = randomMeansWithSparsity(nbArms=NB_ARMS, sparsity=SPARSITY, mingap=0.05, lower=0., lowerNonZero=0.7, amplitude=1.)
MEANS = randomMeansWithSparsity(nbArms=NB_ARMS, sparsity=SPARSITY, mingap=0.05, lower=0., lowerNonZero=2., amplitude=10.)


#: This dictionary configures the experiments
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
        # {   # A very easy problem, but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "params": MEANS
        # },
        # "environment": [  # 2)  Gaussian arms
        {   # An example problem with 3 or 9 arms
            "arm_type": Gaussian,
            # "params": [(mean, VARIANCE, MINI, MAXI) for mean in list(range(-8, 10, 2))]
            # "params": [(mean, VARIANCE) for mean in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
            "params": [(mean, VARIANCE) for mean in MEANS]
            # "params": [(mean, VARIANCE) for mean in [0.1, 0.5, 0.9]]
        },
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
        # --- FIXME SparseWrapper algorithm, 4 different versions whether using old UCB for sets J(t) and K(t) or not
        {
            "archtype": SparseWrapper,
            "params": {
                "sparsity": SPARSITY,
                "policy": klUCBPlus,
                "use_ucb_for_set_J": True,
                "use_ucb_for_set_K": True,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        {
            "archtype": SparseWrapper,
            "params": {
                "sparsity": SPARSITY,
                "policy": klUCBPlus,
                "use_ucb_for_set_J": True,
                "use_ucb_for_set_K": False,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        {
            "archtype": SparseWrapper,
            "params": {
                "sparsity": SPARSITY,
                "policy": klUCBPlus,
                "use_ucb_for_set_J": False,
                "use_ucb_for_set_K": True,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        {
            "archtype": SparseWrapper,
            "params": {
                "sparsity": SPARSITY,
                "policy": klUCBPlus,
                "use_ucb_for_set_J": False,
                "use_ucb_for_set_K": False,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # --- UCBalpha algorithm
        {
            "archtype": UCBalpha,
            "params": {
                "alpha": 4,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        {
            "archtype": UCBalpha,
            "params": {
                "alpha": 1,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        {
            "archtype": UCBalpha,
            "params": {
                "alpha": 0.5,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # --- SparseUCB algorithm
        {
            "archtype": SparseUCB,
            "params": {
                "alpha": 4,
                "sparsity": SPARSITY,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        {
            "archtype": SparseUCB,
            "params": {
                "alpha": 1,
                "sparsity": SPARSITY,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        {
            "archtype": SparseUCB,
            "params": {
                "alpha": 0.5,
                "sparsity": SPARSITY,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # # --- SparseUCB algorithm with a too small value for s
        # {
        #     "archtype": SparseUCB,
        #     "params": {
        #         "alpha": 4,
        #         "sparsity": max(SPARSITY - 2, 1),
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # {
        #     "archtype": SparseUCB,
        #     "params": {
        #         "alpha": 1,
        #         "sparsity": max(SPARSITY - 2, 1),
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # {
        #     "archtype": SparseUCB,
        #     "params": {
        #         "alpha": 0.5,
        #         "sparsity": max(SPARSITY - 2, 1),
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # # --- DONE SparseUCB algorithm with a larger value for s
        # # XXX It fails completely!
        # {
        #     "archtype": SparseUCB,
        #     "params": {
        #         "alpha": 4,
        #         "sparsity": min(SPARSITY + 1, NB_ARMS),
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # {
        #     "archtype": SparseUCB,
        #     "params": {
        #         "alpha": 1,
        #         "sparsity": min(SPARSITY + 1, NB_ARMS),
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # --- SparseklUCB algorithm, using KL-UCB for sets J(t) and K(t)
        {
            "archtype": SparseklUCB,
            "params": {
                "sparsity": SPARSITY,
                "use_ucb_for_sets": False,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # # --- SparseklUCB algorithm with a too small value for s, using KL-UCB for sets J(t) and K(t)
        # {
        #     "archtype": SparseklUCB,
        #     "params": {
        #         "sparsity": max(SPARSITY - 2, 1),
        #         "use_ucb_for_sets": False,
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # --- SparseklUCB algorithm, using old UCB for sets J(t) and K(t)
        {
            "archtype": SparseklUCB,
            "params": {
                "sparsity": SPARSITY,
                "use_ucb_for_sets": True,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }
        },
        # # --- SparseklUCB algorithm with a too small value for s, using old UCB for sets J(t) and K(t)
        # {
        #     "archtype": SparseklUCB,
        #     "params": {
        #         "sparsity": max(SPARSITY - 2, 1),
        #         "use_ucb_for_sets": True,
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #     }
        # },
        # # --- DONE SparseklUCB algorithm with a larger value for s
        # # XXX It fails completely!
        # {
        #     "archtype": SparseklUCB,
        #     "params": {
        #         "sparsity": min(SPARSITY + 1, NB_ARMS),
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
            "archtype": klUCBPlus,
            "params": {
                "lower": LOWER, "amplitude": AMPLITUDE,
                "klucb": klucbBern,  # "horizon": HORIZON,
            }
        },
        # {
        #     "archtype": klUCBPlus,
        #     "params": {
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #         "klucb": klucbExp,  # "horizon": HORIZON,
        #     }
        # },
        # {
        #     "archtype": klUCBPlus,
        #     "params": {
        #         "lower": LOWER, "amplitude": AMPLITUDE,
        #         "klucb": klucbGauss,  # "horizon": HORIZON,
        #     }
        # },
        # {
        #     "archtype": klUCBPlus,
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
        # --- Finite-Horizon Gittins index
        # {
        #     "archtype": ApproximatedFHGittins,
        #     "params": {
        #         "horizon": 1.1 * HORIZON,
        #         "alpha": 2,
        #     }
        # },
        {
            "archtype": ApproximatedFHGittins,
            "params": {
                "horizon": 1.1 * HORIZON,
                "alpha": 1,
            }
        },
        # {
        #     "archtype": ApproximatedFHGittins,
        #     "params": {
        #         "horizon": 1.1 * HORIZON,
        #         "alpha": 0.5,
        #     }
        # },
    ]
})


print("Loaded experiments configuration from 'configuration.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
