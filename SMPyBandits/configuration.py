# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the single-player case.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

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

# Import arms and algorithms
try:
    from Arms import *
    from Policies import *
except ImportError:
    from SMPyBandits.Arms import *
    from SMPyBandits.Policies import *

#: HORIZON : number of time steps of the experiments.
#: Warning Should be >= 10000 to be interesting "asymptotically".
HORIZON = 10000
HORIZON = int(getenv('T', HORIZON))

#: REPETITIONS : number of repetitions of the experiments.
#: Warning: Should be >= 10 to be statistically trustworthy.
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
# REPETITIONS = 1000
REPETITIONS = int(getenv('N', REPETITIONS))

#: To profile the code, turn down parallel computing
DO_PARALLEL = False  # XXX do not let this = False
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1 or REPETITIONS == -1) and DO_PARALLEL

#: Number of jobs to use for the parallel computations. -1 means all the CPU cores, 1 means no parallelization.
N_JOBS = -1 if DO_PARALLEL else 1
if CPU_COUNT > 4:  # We are on a server, let's be nice and not use all cores
    N_JOBS = min(CPU_COUNT, max(int(CPU_COUNT / 3), CPU_COUNT - 8))
N_JOBS = int(getenv('N_JOBS', N_JOBS))
if REPETITIONS == -1:
    REPETITIONS = max(N_JOBS, CPU_COUNT)

# Random events
RANDOM_SHUFFLE = False  #: The arms won't be shuffled (``shuffle(arms)``).
# RANDOM_SHUFFLE = True  #: The arms will be shuffled (``shuffle(arms)``).
RANDOM_SHUFFLE = getenv('RANDOM_SHUFFLE', str(RANDOM_SHUFFLE)) == 'True'
RANDOM_INVERT = False  #: The arms won't be inverted (``arms = arms[::-1]``).
# RANDOM_INVERT = True  #: The arms will be inverted (``arms = arms[::-1]``).
RANDOM_INVERT = getenv('RANDOM_INVERT', str(RANDOM_INVERT)) == 'True'

NB_BREAK_POINTS = 0  #: Number of true breakpoints. They are uniformly spaced in time steps (and the first one at t=0 does not count).
# NB_BREAK_POINTS = 3  #: Number of true breakpoints. They are uniformly spaced in time steps (and the first one at t=0 does not count).
# NB_BREAK_POINTS = 5  #: Number of true breakpoints. They are uniformly spaced in time steps (and the first one at t=0 does not count).
NB_BREAK_POINTS = int(getenv('NB_BREAK_POINTS', NB_BREAK_POINTS))

#: Parameters for the epsilon-greedy and epsilon-... policies.
EPSILON = 0.1
#: Temperature for the Softmax policies.
TEMPERATURE = 0.05

#: Learning rate for my aggregated bandit (it can be autotuned)
LEARNING_RATE = 0.01

# To try more learning rates in one run
LEARNING_RATES = [LEARNING_RATE]

#: Constant time tau for the decreasing rate for my aggregated bandit.
# FIXED I tried to make self.learningRate decrease when self.t increase, it was not better
DECREASE_RATE = None
DECREASE_RATE = HORIZON / 2.0
DECREASE_RATE = 'auto'  # FIXED using the formula from Theorem 4.2 from [Bubeck & Cesa-Bianchi, 2012](http://sbubeck.com/SurveyBCB12.pdf)

#: To know if my WrapRange policy is tested.
TEST_WrapRange = True
TEST_WrapRange = False  # XXX do not let this = False if you want to test my WrapRange policy

#: Should we cache rewards? The random rewards will be the same for all the REPETITIONS simulations for each algorithms.
CACHE_REWARDS = False  # XXX to manually disable this feature?
CACHE_REWARDS = True  # XXX to manually enable this feature?

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
UNBOUNDED_VARIANCE = 1   #: Variance of unbounded Gaussian arms
VARIANCE = 0.05   #: Variance of Gaussian arms

#: Number of arms for non-hard-coded problems (Bayesian problems)
NB_ARMS = 9
NB_ARMS = int(getenv('K', NB_ARMS))
NB_ARMS = int(getenv('NB_ARMS', NB_ARMS))

#: Default value for the lower value of means
LOWER = 0.
#: Default value for the amplitude value of means
AMPLITUDE = 1.

#: Type of arms for non-hard-coded problems (Bayesian problems)
ARM_TYPE = "Bernoulli"
ARM_TYPE = str(getenv('ARM_TYPE', ARM_TYPE))

# WARNING That's nonsense, rewards of unbounded distributions just don't have lower, amplitude values...
if ARM_TYPE in [
        "UnboundedGaussian",
        # "Gaussian",
    ]:
    LOWER = -5
    AMPLITUDE = 10

LOWER = float(getenv('LOWER', LOWER))
AMPLITUDE = float(getenv('AMPLITUDE', AMPLITUDE))
assert AMPLITUDE > 0, "Error: invalid amplitude = {:.3g} but has to be > 0."  # DEBUG
VARIANCE = float(getenv('VARIANCE', VARIANCE))

ARM_TYPE_str = str(ARM_TYPE)
ARM_TYPE = mapping_ARM_TYPE[ARM_TYPE]

#: True to use bayesian problem
ENVIRONMENT_BAYESIAN = False
ENVIRONMENT_BAYESIAN = getenv('BAYES', str(ENVIRONMENT_BAYESIAN)) == 'True'

#: Means of arms for non-hard-coded problems (non Bayesian)
MEANS = uniformMeans(nbArms=NB_ARMS, delta=0.05, lower=LOWER, amplitude=AMPLITUDE, isSorted=True)

import numpy as np
# more parametric? Read from cli?
MEANS_STR = getenv('MEANS', '')
if MEANS_STR != '':
    MEANS = [ float(m) for m in MEANS_STR.replace('[', '').replace(']', '').split(',') ]
    print("Using cli env variable to use MEANS = {}.".format(MEANS))  # DEBUG

#: True to use full-restart Doubling Trick
USE_FULL_RESTART = True
USE_FULL_RESTART = getenv('FULL_RESTART', str(USE_FULL_RESTART)) == 'True'


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
    "nb_break_points": NB_BREAK_POINTS,
    # --- Should we plot the lower-bounds or not?
    "plot_lowerbound": True,  # XXX Default
    # "plot_lowerbound": False,
    # --- Cache rewards: use the same random rewards for the Aggregator[..] and the algorithms
    "cache_rewards": CACHE_REWARDS,
    "environment_bayesian": ENVIRONMENT_BAYESIAN,
    # --- Arms
    "environment": [  # XXX Bernoulli arms
        # {   # The easier problem: 2 arms, one perfectly bad, one perfectly good
        #     "arm_type": Bernoulli,
        #     "params": [0, 1]
        # },
        # {   # A very very easy problem: 2 arms, one better than the other
        #     "arm_type": Bernoulli,
        #     "params": [0.8, 0.9]
        # },
        # {   # A very very easy problem: 2 arms, one better than the other
        #     "arm_type": Bernoulli,
        #     "params": [0.375, 0.571]
        # },
        # {   # A very very easy problem: 3 arms, one bad, one average, one good
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.5, 0.9]
        # },
        # {   # Another very easy problem: 3 arms, two very bad, one bad
        #     "arm_type": Bernoulli,
        #     "params": [0.04, 0.05, 0.1]
        # },
        {   # Use vector from command line
            "arm_type": ARM_TYPE,
            "params": MEANS
        },
        # {   # XXX A very easy problem, but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # },
        # # XXX Default! FIXME always bring this back as default after experimenting
        # {   # A very easy problem (X arms), but it is used in a lot of articles
        #     "arm_type": ARM_TYPE,
        #     "params": uniformMeans(nbArms=NB_ARMS, delta=1./(1. + NB_ARMS), lower=LOWER, amplitude=AMPLITUDE, isSorted=False)
        # },
        # {   # An other problem, best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3 - 0.6) and very good arms (0.78, 0.8, 0.82)
        #     "arm_type": Bernoulli,
        #     "params": [0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.78, 0.8, 0.82]
        # },
        # {   # Another example problem, from [Fang Liu et al, 2018](https://arxiv.org/abs/1804.05929)
        #     "arm_type": Bernoulli,
        #     "params": [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.1]
        # },
        # {   # Lots of bad arms, significative difference between the best and the others
        #     "arm_type": Bernoulli,
        #     "params": [0.001, 0.001, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.3]
        # },
        # {   # VERY HARD One optimal arm, much better than the others, but *lots* of bad arms (34 arms!)
        #     "arm_type": Bernoulli,
        #     "params": [0.001, 0.001, 0.001, 0.001, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2, 0.5]
        # },
        # {   # HARD An other problem (17 arms), best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3, 0.6) and very good arms (0.78, 0.85)
        #     "arm_type": Bernoulli,
        #     "params": [0.005, 0.01, 0.015, 0.02, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.78, 0.8, 0.82, 0.83, 0.84, 0.85]
        # },
        # {   # A Bayesian problem: every repetition use a different means vector!
        #     "arm_type": ARM_TYPE,
        #     "params": {
        #         "newMeans": randomMeans,
        #         "args": {
        #             "nbArms": NB_ARMS,
        #             "mingap": None,
        #             # "mingap": 0.0000001,
        #             # "mingap": 0.1,
        #             # "mingap": 1. / (3 * NB_ARMS),
        #             "lower": 0.,
        #             "amplitude": 1.,
        #             "isSorted": True,
        #         }
        #     }
        # },
    ],
    # "environment": [  # XXX Exponential arms
    #     {   # An example problem with 9 arms
    #         "arm_type": ExponentialFromMean,
    #         "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #     },
    # ],
    # "environment": [  # XXX Gaussian arms
    #     {   # An example problem with 3 arms
    #         "arm_type": Gaussian,
    #         "params": [(0.2, VARIANCE, LOWER, LOWER+AMPLITUDE), (0.5, VARIANCE, LOWER, LOWER+AMPLITUDE), (0.8, VARIANCE, LOWER, LOWER+AMPLITUDE)]
    #     },
    #     # {   # An example problem with 9 arms
    #     #     "arm_type": Gaussian,
    #     #     "params": [(0.1, VARIANCE, LOWER, LOWER+AMPLITUDE), (0.2, VARIANCE, LOWER, LOWER+AMPLITUDE), (0.3, VARIANCE, LOWER, LOWER+AMPLITUDE), (0.4, VARIANCE, LOWER, LOWER+AMPLITUDE), (0.5, VARIANCE, LOWER, LOWER+AMPLITUDE), (0.6, VARIANCE, LOWER, LOWER+AMPLITUDE), (0.7, VARIANCE, LOWER, LOWER+AMPLITUDE), (0.8, VARIANCE, LOWER, LOWER+AMPLITUDE), (0.9, VARIANCE, LOWER, LOWER+AMPLITUDE)]
    #     # },
    # ],
    # "environment": [  # XXX Unbounded Gaussian arms
    #     {   # An example problem with 9 arms
    #         "arm_type": UnboundedGaussian,
    #         "params": [(-40, VARIANCE), (-30, VARIANCE), (-20, VARIANCE), (-VARIANCE, VARIANCE), (0, VARIANCE), (VARIANCE, VARIANCE), (20, VARIANCE), (30, VARIANCE), (40, VARIANCE)]
    #     },
    # ],
    # "environment": [  # XXX DiscreteArm arms
    #     {   # An example problem with 9 arms
    #         "arm_type": DiscreteArm,
    #         "params": [  # 3-values discrete arms. XXX NOT one-dimensional, NOT parametrized by their means
    #             [{0: 0.45, 0.5: 0.45, 1: 0.1}],
    #             [{0: 0.4,  0.5: 0.4,  1: 0.2}],
    #             [{0: 0.35, 0.5: 0.35, 1: 0.3}],
    #             [{0: 0.3,  0.5: 0.3,  1: 0.4}],
    #             [{0: 0.25, 0.5: 0.25, 1: 0.5}],
    #             [{0: 0.2,  0.5: 0.2,  1: 0.6}],
    #             [{0: 0.15, 0.5: 0.15, 1: 0.7}],
    #             [{0: 0.1,  0.5: 0.1,  1: 0.8}],
    #             [{0: 0.05, 0.5: 0.05, 1: 0.9}],
    #         ]
    #     },
    # ],
}

if ENVIRONMENT_BAYESIAN:
    configuration["environment"] = [
        {   # A Bayesian problem: every repetition use a different means vector!
            "arm_type": ARM_TYPE,
            "params": {
                "newMeans": randomMeans,
                "args": {
                    "nbArms": NB_ARMS,
                    # "mingap": None,
                    "mingap": 0.1 if NB_ARMS <= 3 else 1. / (3 * NB_ARMS),
                    "lower": LOWER,
                    "amplitude": AMPLITUDE,
                    "isSorted": True,
                }
            }
        },
    ]

# if len(configuration['environment']) > 1:
#     raise ValueError("WARNING do not use this hack if you try to use more than one environment.")
#     # Note: I dropped the support for more than one environments, for this part of the configuration, but not the simulation code

try:
    #: Number of arms *in the first environment*
    nbArms = int(configuration['environment'][0]['params']['args']['nbArms'])
except (TypeError, KeyError):
    try:
        nbArms = len(configuration['environment'][0]['params']['listOfMeans'][0])
    except (TypeError, KeyError):
        nbArms = len(configuration['environment'][0]['params'])

#: Warning: if using Exponential or Gaussian arms, gives klExp or klGauss to KL-UCB-like policies!
klucb = klucb_mapping.get(str(configuration['environment'][0]['arm_type']), klucbBern)

#: Compute the gap of the first problem.
#: (for d in MEGA's parameters, and epsilon for MusicalChair's parameters)
try:
    sorted_means = np.sort(configuration['environment'][0]['params'])
    if np.isclose(np.min(sorted_means), np.max(sorted_means)):
        GAP = np.min(np.diff(sorted_means))
    else:
        GAP = np.max(sorted_means) - np.max(sorted_means[sorted_means < np.max(sorted_means)])
except (ValueError, np.AxisError, TypeError):
    print("Warning: using the default value for the GAP (Bayesian environment maybe?)")  # DEBUG
    GAP = 1. / (3 * NB_ARMS)


configuration.update({
    "policies": [
        # # --- Stupid algorithms
        {
            "archtype": Uniform,   # The stupidest policy, fully uniform
            "params": {},
            "change_label": "Pure exploration",
        },
        {
            "archtype": EmpiricalMeans,   # The naive policy, just using empirical means
            "params": {},
            "change_label": "Pure exploitation",
        },
        # # {
        # #     "archtype": TakeRandomFixedArm,   # The stupidest policy
        # #     "params": {}
        # # },
        # # {
        # #     "archtype": TakeRandomFixedArm,   # The stupidest policy
        # #     "params": {}
        # # },
        # # # --- Full or partial knowledge algorithms
        # { "archtype": TakeFixedArm, "params": { "armIndex": 0 }},  # Take worse arm!
        # { "archtype": TakeFixedArm, "params": { "armIndex": 1 }},  # Take second worse arm!
        # { "archtype": TakeFixedArm, "params": { "armIndex": min(2, nbArms - 1) }},  # Take third worse arm!
        # # { "archtype": TakeFixedArm, "params": { "armIndex": nbArms - 2 }},  # Take second best arm!
        # # { "archtype": TakeFixedArm, "params": { "armIndex": nbArms - 1 }},  # Take best arm!
        # # --- Epsilon-... algorithms
        # {
        #     "archtype": EpsilonGreedy,   # This basic EpsilonGreedy is very bad
        #     "params": {
        #         "epsilon": EPSILON,
        #     }
        # },
        {
            "archtype": EpsilonDecreasing,   # This basic EpsilonGreedy is also very bad
            "params": {
                "epsilon": 6 * NB_ARMS / GAP,
                # "decreasingRate": 0.005,
            },
            "change_label": r"$\varepsilon$-greedy",
        },
        # {
        #     "archtype": EpsilonFirst,   # This basic EpsilonFirst is also very bad
        #     "params": {
        #         "epsilon": EPSILON,
        #         "horizon": HORIZON,
        #     }
        # },
        # # --- Explore-Then-Commit policies
        # {
        #     "archtype": ETC_KnownGap,
        #     "params": {
        #         "horizon": HORIZON,
        #         "gap": GAP,
        #     }
        # },
        # # {
        # #     "archtype": ETC_KnownGap,
        # #     "params": {
        # #         "horizon": HORIZON,
        # #         "gap": 0.05,
        # #     }
        # # },
        {
            "archtype": ETC_KnownGap,
            "params": {
                "horizon": HORIZON,
                # "gap": max(0.01, 1.0 / (3 * NB_ARMS)),
                "gap": GAP,
            },
            "change_label": "Explore-then-Exploit",
        },
        # # {
        # #     "archtype": ETC_KnownGap,
        # #     "params": {
        # #         "horizon": HORIZON,
        # #         "gap": 0.5
        # #     }
        # # },
        # {
        #     "archtype": ETC_RandomStop,
        #     "params": {
        #         "horizon": HORIZON,
        #     }
        # },
        # {
        #     "archtype": ETC_FixedBudget,
        #     "params": {
        #         "horizon": HORIZON,
        #         "gap": GAP,
        #     }
        # },
        # {
        #     "archtype": ETC_SPRT,
        #     "params": {
        #         "horizon": HORIZON,
        #         "gap": GAP,
        #     }
        # },
        # {
        #     "archtype": ETC_BAI,
        #     "params": {
        #         "horizon": HORIZON,
        #     }
        # },
        # {
        #     "archtype": DeltaUCB,
        #     "params": {
        #         "horizon": HORIZON,
        #         "gap": GAP,
        #     }
        # },
        # --- Softmax algorithms
        # {
        #     "archtype": Softmax,   # This basic Softmax is very bad
        #     "params": {
        #         "temperature": TEMPERATURE,
        #     }
        # },
        # {
        #     "archtype": SoftmaxDecreasing,   # XXX Efficient parameter-free Softmax
        #     "params": {}
        # },
        # {
        #     "archtype": SoftMix,   # Another parameter-free Softmax
        #     "params": {}
        # },
        # {
        #     "archtype": SoftmaxWithHorizon,  # Other Softmax, knowing the horizon
        #     "params": {
        #         "horizon": HORIZON,
        #     }
        # },
        # # --- Boltzmann-Gumbel algorithms
        # {
        #     "archtype": BoltzmannGumbel,
        #     "params": {
        #         "C": 1.0,
        #     }
        # },
        # {
        #     "archtype": BoltzmannGumbel,
        #     "params": {
        #         "C": 2.0,
        #     }
        # },
        # {
        #     "archtype": BoltzmannGumbel,
        #     "params": {
        #         "C": 0.5,
        #     }
        # },
        # {
        #     "archtype": BoltzmannGumbel,
        #     "params": {
        #         "C": 0.1,
        #     }
        # },
        # {
        #     "archtype": BoltzmannGumbel,
        #     "params": {
        #         "C": 0.01,
        #     }
        # },
        # --- Exp3 algorithms - Very bad !!!!
        # {
        #     "archtype": Exp3,   # This basic Exp3 is not very good
        #     "params": {
        #         "gamma": 0.001,
        #     }
        # },
        # {
        #     "archtype": Exp3Decreasing,
        #     "params": {
        #         "gamma": 0.001,
        #     }
        # },
        # {
        #     "archtype": Exp3SoftMix,   # Another parameter-free Exp3
        #     "params": {}
        # },
        # {
        #     "archtype": Exp3WithHorizon,  # Other Exp3, knowing the horizon
        #     "params": {
        #         "horizon": HORIZON,
        #     }
        # },
        # {
        #     "archtype": Exp3ELM,   # This improved Exp3 is not better, it targets a different problem
        #     "params": {
        #         "delta": 0.1,
        #     }
        # },
        # # --- Exp3PlusPlus algorithm
        # {
        #     "archtype": Exp3PlusPlus,   # Another parameter-free Exp3, better parametrization
        #     "params": {}
        # },
        # # --- Probability pursuit algorithm
        # {
        #     "archtype": ProbabilityPursuit,
        #     "params": {
        #         "beta": 0.5,
        #     }
        # },
        # {
        #     "archtype": ProbabilityPursuit,
        #     "params": {
        #         "beta": 0.1,
        #     }
        # },
        # {
        #     "archtype": ProbabilityPursuit,
        #     "params": {
        #         "beta": 0.05,
        #     }
        # },
        # # --- Hedge algorithm
        # {
        #     "archtype": Hedge,
        #     "params": {
        #         "epsilon": 0.5,
        #     }
        # },
        # {
        #     "archtype": Hedge,
        #     "params": {
        #         "epsilon": 0.1,
        #     }
        # },
        # {
        #     "archtype": Hedge,
        #     "params": {
        #         "epsilon": 0.05,
        #     }
        # },
        # {
        #     "archtype": HedgeDecreasing,
        #     "params": {}
        # },
        # {
        #     "archtype": HedgeWithHorizon,
        #     "params": {
        #         "horizon": HORIZON,
        #     }
        # },
        # # --- UCB algorithms
        # {
        #     "archtype": UCB,   # This basic UCB is very worse than the other
        #     "params": {}
        # },
        # {
        #     "archtype": UCBlog10,   # This basic UCB is very worse than the other
        #     "params": {}
        # },
        # {
        #     "archtype": UCBwrong,  # This wrong UCB is very very worse than the other
        #     "params": {}
        # },
        # {
        #     "archtype": UCBplus,
        #     "params": {}
        # },
        # {
        #     "archtype": UCBmin,
        #     "params": {}
        # },
        # {
        #     "archtype": UCBrandomInit,
        #     "params": {}
        # },
        # {
        #     "archtype": UCBV,   # UCB with variance term
        #     "params": {}
        # },
        # {
        #     "archtype": UCBVtuned,   # UCB with variance term and one trick
        #     "params": {}
        # },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         "alpha": 4,         # Below the alpha=4 like old classic UCB
        #     }
        # },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         "alpha": 3,
        #     }
        # },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         "alpha": 2,
        #     }
        # },
        {
            "archtype": UCBalpha,   # UCB with custom alpha parameter
            "params": {
                "alpha": 1,
            }
        },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         "alpha": 0.5,       # XXX Below the theoretically acceptable value!
        #     }
        # },
        # {
        #     "archtype": SWR_UCBalpha,   # XXX experimental sliding window algorithm
        #     "params": {
        #         "alpha": 0.5,
        #     }
        # },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         "alpha": 0.25,      # XXX Below the theoretically acceptable value!
        #     }
        # },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         "alpha": 0.1,       # XXX Below the theoretically acceptable value!
        #     }
        # },
        # {
        #     "archtype": UCBalpha,   # UCB with custom alpha parameter
        #     "params": {
        #         "alpha": 0.05,      # XXX Below the theoretically acceptable value!
        #     }
        # },
        # # --- new UCBcython algorithm
        # {
        #     "archtype": UCBcython,
        #     "params": {
        #         "alpha": 1.,
        #     }
        # },
        # --- KL algorithms
        {
            "archtype": klUCB,
            "params": {
                "klucb": klucb,
            },
            "change_label": r"Bernoulli kl-UCB",
        },
        # {
        #     "archtype": SWR_klUCB,   # XXX experimental sliding window algorithm
        #     "params": {
        #         "klucb": klucb,
        #     }
        # },
        # {
        #     "archtype": klUCB,
        #     "params": {
        #         "c": 0.434294,  # = 1. / np.log(10) ==> like klUCBlog10
        #         "klucb": klucb,
        #     }
        # },
        # {
        #     "archtype": klUCB,
        #     "params": {
        #         "c": 3.,
        #         "klucb": klucb,
        #     }
        # },
        # {
        #     "archtype": klUCBloglog,
        #     "params": {
        #         "klucb": klucb,
        #     }
        # },
        # {
        #     "archtype": klUCBloglog,
        #     "params": {
        #         "c": 3.,
        #         "klucb": klucb,
        #     }
        # },
        # {
        #     "archtype": klUCBlog10,
        #     "params": {
        #         "klucb": klucb
        #     }
        # },
        # {
        #     "archtype": klUCBloglog10,
        #     "params": {
        #         "klucb": klucb,
        #     }
        # },
        # {
        #     "archtype": klUCBPlus,
        #     "params": {
        #         "klucb": klucb,
        #     }
        # },
        # {
        #     "archtype": klUCBHPlus,
        #     "params": {
        #         "horizon": HORIZON,
        #         "klucb": klucb,
        #     }
        # },
        # {
        #     "archtype": klUCBPlusPlus,
        #     "params": {
        #         "horizon": HORIZON,
        #         "klucb": klucb,
        #     }
        # },
        # --- Thompson algorithms
        {
            "archtype": Thompson,
            "params": {
                "posterior": Beta,
            },
            "change_label": r"Thompson sampling",
        },
        # # {
        # #     "archtype": Thompson,
        # #     "params": {
        # #         "posterior": Gauss,
        # #     }
        # # },
        # # --- Bayes UCB algorithms
        # {
        #     "archtype": BayesUCB,
        #     "params": {
        #         "posterior": Beta,
        #     }
        # },
        # # # {
        # # #     "archtype": BayesUCB,
        # # #     "params": {
        # # #         "posterior": Gauss,
        # # #     }
        # # # },
        # # --- AdBandits with different alpha paramters
        # {
        #     "archtype": AdBandits,
        #     "params": {
        #         "alpha": 0.5,
        #         "horizon": HORIZON,
        #     }
        # },
        # # # DONE Compare AdBandits with Aggregator[BayesUCB, Thompson]
        # # {
        # #     "archtype": Aggregator,
        # #     "params": {
        # #         "unbiased": UNBIASED,
        # #         "update_all_children": UPDATE_ALL_CHILDREN,
        # #         "decreaseRate": DECREASE_RATE,
        # #         "learningRate": LEARNING_RATE,
        # #         "children": [
        # #             { "archtype": BayesUCB, "params": {} },
        # #             { "archtype": Thompson, "params": {} },
        # #         ],
        # #         "update_like_exp4": UPDATE_LIKE_EXP4,
        # #         # "horizon": HORIZON  # XXX uncomment to give the value of horizon to have a better learning rate
        # #     },
        # # },
        # # --- The awesome BESA algorithm
        # # {
        # #     "archtype": BESA,
        # #     "params": {
        # #         "horizon": HORIZON,
        # #         "minPullsOfEachArm": 1,  # Default, don't seem to improve if increasing this one
        # #         "randomized_tournament": True,
        # #         # "randomized_tournament": False,  # XXX Very inefficient!
        # #         "random_subsample": True,
        # #         # "random_subsample": False,  # XXX Very inefficient!
        # #         # "non_binary": False,
        # #         "non_binary": True,
        # #         # "non_recursive": False,
        # #         "non_recursive": True,
        # #     }
        # # },
        # # {
        # #     "archtype": BESA,
        # #     "params": {
        # #         "horizon": HORIZON,
        # #         "non_binary": True,
        # #     }
        # # },
        # # {
        # #     "archtype": BESA,
        # #     "params": {
        # #         "horizon": HORIZON,
        # #         "non_recursive": False,
        # #         "non_binary": False,
        # #     }
        # # },
        # # {
        # #     "archtype": BESA,
        # #     "params": {
        # #         "horizon": HORIZON,
        # #         "non_recursive": False,
        # #         "non_binary": True,
        # #     }
        # # },
        # {
        #     "archtype": BESA,
        #     "params": {
        #         "horizon": HORIZON,
        #         "non_recursive": False,
        #         "non_binary": True,
        #     }
        # },
        # # # --- MOSS algorithm, like UCB
        # # {
        # #     "archtype": MOSS,
        # #     "params": {}
        # # },
        # # # --- MOSS-H algorithm, like UCB-H
        # # {
        # #     "archtype": MOSSH,
        # #     "params": {
        # #         "horizon": HORIZON,
        # #     }
        # # },
        # # --- MOSS-Anytime algorithm, extension of MOSS
        # {
        #     "archtype": MOSSAnytime,
        #     "params": {
        #         "alpha": 1.35,
        #     }
        # },
        # # # --- MOSS-Experimental algorithm, extension of MOSSAnytime
        # # {
        # #     "archtype": MOSSExperimental,
        # #     "params": {}
        # # },
        # # --- Horizon-dependent algorithm ApproximatedFHGittins
        # {
        #     "archtype": ApproximatedFHGittins,
        #     "params": {
        #         "alpha": 0.5,
        #         "horizon": max(HORIZON + 100, int(1.05 * HORIZON)),
        #         # "horizon": HORIZON,
        #         # "horizon": HORIZON + 1,
        #     }
        # },
        # # --- Auto-tuned UCBdagger algorithm
        # {
        #     "archtype": UCBdagger,
        #     "params": {
        #         "horizon": HORIZON,
        #     }
        # },
        # # # --- Optimally-Confident UCB algorithm
        # # {
        # #     "archtype": OCUCB,
        # #     "params": {
        # #         "eta": 1.1,
        # #         "rho": 1,
        # #     }
        # # },
        # # {
        # #     "archtype": OCUCB,
        # #     "params": {
        # #         "eta": 1.1,
        # #         "rho": 0.9,
        # #     }
        # # },
        # # {
        # #     "archtype": OCUCB,
        # #     "params": {
        # #         "eta": 1.1,
        # #         "rho": 0.8,
        # #     }
        # # },
        # # {
        # #     "archtype": OCUCB,
        # #     "params": {
        # #         "eta": 1.1,
        # #         "rho": 0.7,
        # #     }
        # # },
        # # {
        # #     "archtype": OCUCB,
        # #     "params": {
        # #         "eta": 1.1,
        # #         "rho": 0.6,
        # #     }
        # # },
        # # # --- Optimally-Confident UCB algorithm, horizon dependent
        # # {
        # #     "archtype": OCUCBH,
        # #     "params": {
        # #         "psi": 2,
        # #         "alpha": 4,
        # #         "horizon": HORIZON,
        # #     }
        # # },
        # # {
        # #     "archtype": AOCUCBH,
        # #     "params": {
        # #         "horizon": HORIZON,
        # #     }
        # # },
        # # --- CPUCB algorithm, other variant of UCB
        # # {
        # #     "archtype": CPUCB,
        # #     "params": {}
        # # },
        # # # --- DMED algorithm, similar to klUCB
        # # {
        # #     "archtype": DMEDPlus,
        # #     "params": {}
        # # },
        # # # {
        # # #     "archtype": DMED,
        # # #     "params": {}
        # # # },
        # # # --- IMED algorithm, similar to DMED
        # # {
        # #     "archtype": IMED,
        # #     "params": {}
        # # },
        # # # --- new klUCBswitch algorithm!
        # # {
        # #     "archtype": klUCBswitch,
        # #     "params": {
        # #         "horizon": HORIZON,
        # #         "klucb": klucb,
        # #         "threshold": "best"
        # #     }
        # # },
        # # {
        # #     "archtype": klUCBswitch,
        # #     "params": {
        # #         "horizon": HORIZON,
        # #         "klucb": klucb,
        # #         "threshold": "delayed"
        # #     }
        # # },
        # {
        #     "archtype": klUCBswitchAnytime,
        #     "params": {
        #         "klucb": klucb,
        #         "threshold": "best"
        #     }
        # },
        # # {
        # #     "archtype": klUCBswitchAnytime,
        # #     "params": {
        # #         "klucb": klucb,
        # #         "threshold": "delayed"
        # #     }
        # # },
        # # # DONE Compare klUCBSwitch with Aggregator[klUCB, MOSS]
        # # {
        # #     "archtype": Aggregator,
        # #     "params": {
        # #         "unbiased": UNBIASED,
        # #         "update_all_children": UPDATE_ALL_CHILDREN,
        # #         "decreaseRate": DECREASE_RATE,
        # #         "learningRate": LEARNING_RATE,
        # #         "children": [
        # #             { "archtype": klUCB, "params": {} },
        # #             { "archtype": MOSS, "params": {} },
        # #         ],
        # #         "update_like_exp4": UPDATE_LIKE_EXP4,
        # #         # "horizon": HORIZON  # XXX uncomment to give the value of horizon to have a better learning rate
        # #     },
        # # },
        # # --- TsallisInf algorithm
        # {
        #     "archtype": TsallisInf,
        #     "params": {
        #         "alpha": 0.5,
        #     }
        # },
        # # # {
        # # #     "archtype": TsallisInf,
        # # #     "params": {
        # # #         "alpha": 0.25,
        # # #     }
        # # # },
        # # # {
        # # #     "archtype": TsallisInf,
        # # #     "params": {
        # # #         "alpha": 0.75,
        # # #     }
        # # # },
        # # --- RCB algorithm
        # {
        #     "archtype": RCB,
        #     "params": {
        #         "alpha": 1.,
        #         "perturbation": "uniform",  # like UCB?
        #     }
        # },
        # # {
        # #     "archtype": RCB,
        # #     "params": {
        # #         "alpha": 1.,
        # #         "perturbation": "gaussian",  # like Thompson sampling
        # #     }
        # # },
        # # {
        # #     "archtype": RCB,
        # #     "params": {
        # #         "alpha": 1.,
        # #         "perturbation": "exponential",
        # #     }
        # # },
        # # {
        # #     "archtype": RCB,
        # #     "params": {
        # #         "alpha": 1.,
        # #         "perturbation": "gumbel",
        # #     }
        # # },
        # # --- PHE algorithm
        # {
        #     "archtype": PHE,
        #     "params": {
        #         "perturbation_scale": 0.5,
        #     }
        # },
        # # {
        # #     "archtype": PHE,
        # #     "params": {
        # #         "perturbation_scale": 1.1,
        # #     }
        # # },
        # # {
        # #     "archtype": PHE,
        # #     "params": {
        # #         "perturbation_scale": 2.1,
        # #     }
        # # },
        # # --- Black Box optimizer, using Gaussian Processes XXX works well, but VERY SLOW
        # # {
        # #     "archtype": BlackBoxOpt,
        # #     "params": {}
        # # },
        # # # --- The new OSSB algorithm
        # # {
        # #     "archtype": OSSB,
        # #     "params": {
        # #         "epsilon": 0.01,
        # #         "gamma": 0.0,
        # #     }
        # # },
        # # {
        # #     "archtype": OSSB,
        # #     "params": {
        # #         "epsilon": 0.001,
        # #         "gamma": 0.0,
        # #     }
        # # },
        # # {
        # #     "archtype": OSSB,
        # #     "params": {
        # #         "epsilon": 0.0,
        # #         "gamma": 0.0,
        # #     }
        # # },
        # # {
        # #     "archtype": OSSB_DecreasingRate,
        # #     "params": {}
        # # },
        # # {
        # #     "archtype": OSSB_AutoDecreasingRate,
        # #     "params": {}
        # # },
        # # # # --- new UCBoost algorithms
        # # # {
        # # #     "archtype": UCB_bq,
        # # #     "params": {}
        # # # },
        # # # {
        # # #     "archtype": UCB_h,
        # # #     "params": {}
        # # # },
        # # {
        # #     "archtype": UCB_lb,
        # #     "params": {}
        # # },
        # # {
        # #     "archtype": UCBoost_bq_h_lb,
        # #     "params": {}
        # # },
        # # # # --- new UCBoostEpsilon algorithm
        # # # {
        # # #     "archtype": UCBoostEpsilon,
        # # #     "params": {
        # # #         "epsilon": 0.1,
        # # #     }
        # # # },
        # # # {
        # # #     "archtype": UCBoostEpsilon,
        # # #     "params": {
        # # #         "epsilon": 0.05,
        # # #     }
        # # # },
        # # # {
        # # #     "archtype": UCBoostEpsilon,
        # # #     "params": {
        # # #         "epsilon": 0.01,
        # # #     }
        # # # },
        # # # --- new UCBoost_cython algorithms
        # # # {
        # # #     "archtype": UCB_bq_cython,
        # # #     "params": {}
        # # # },
        # # # {
        # # #     "archtype": UCB_h_cython,
        # # #     "params": {}
        # # # },
        # # {
        # #     "archtype": UCB_lb_cython,
        # #     "params": {}
        # # },
        # # # {
        # # #     "archtype": UCBoost_bq_h_lb_cython,
        # # #     "params": {}
        # # # },
        # # # # --- new UCBoostEpsilon_cython algorithm
        # # # {
        # # #     "archtype": UCBoostEpsilon_cython,
        # # #     "params": {
        # # #         "epsilon": 0.05,
        # # #     }
        # # # },
        # # # --- new UCBimproved algorithm
        # # {
        # #     "archtype": UCBimproved,
        # #     "params": {
        # #         "horizon": HORIZON,
        # #     }
        # # }
        # # # DONE Compare Generic Aggregation
        # # {
        # #     "archtype": GenericAggregation,
        # #     "params": {
        # #         "master": { "archtype": BayesUCB, "params": {} },
        # #         "children": [
        # #             # # Aggregating fixed-arm policies == playing the master algorithm (just more inefficient regarding time and storage, but same regret)
        # #             # { "archtype": TakeFixedArm, "params": { "armIndex": armId } }
        # #             # for armId in range(NB_ARMS)
        # #             # Confuse it with stupid algorithms
        # #             { "archtype": Uniform, "params": {} },
        # #             { "archtype": EmpiricalMeans, "params": {} },
        # #             { "archtype": Exp3PlusPlus, "params": {} },
        # #             # And use some smart algorithms
        # #             { "archtype": UCB_lb, "params": {} },
        # #             { "archtype": Thompson, "params": {} },
        # #             { "archtype": klUCB, "params": {} },
        # #             { "archtype": BayesUCB, "params": {} },
        # #             # { "archtype": ApproximatedFHGittins, "params": { "alpha": 1, "horizon": HORIZON } },
        # #         ],
        # #     },
        # # },
    ]
})


# # Tiny configuration, for the paper.pdf illustration.
# configuration.update({
#     # Policies that should be simulated, and their parameters.
#     "policies": [
#         {"archtype": UCBalpha, "params": { "alpha": 1 } },
#         {"archtype": klUCB, "params": {} },
#         {"archtype": klUCBPlusPlus, "params": { "horizon": 10000 } },
#         {"archtype": Thompson, "params": {} },
#     ]
# })


# Tiny configuration, for testing the WrapRange policy.
if ARM_TYPE_str in ["Gaussian", "UnboundedGaussian"]:
    configuration.update({
        "environment": [ {
                "arm_type": ARM_TYPE,
                "params": [
                    (mu, VARIANCE, LOWER, LOWER+AMPLITUDE)
                    for mu in
                    uniformMeans(nbArms=NB_ARMS, delta=1./(1. + NB_ARMS), lower=LOWER, amplitude=AMPLITUDE)
                ],
                # "change_lower_amplitude": True  # XXX an experiment to let Environment.Evaluator load a IncreasingMAB instead of just a MAB
        }, ],
    })
elif MEANS_STR == '' and ARM_TYPE_str != "DiscreteArm" and not ENVIRONMENT_BAYESIAN:
    configuration.update({
        "environment": [ {
            "arm_type": ARM_TYPE,
            "params": uniformMeans(nbArms=NB_ARMS, delta=1./(1. + NB_ARMS), lower=LOWER, amplitude=AMPLITUDE)
        }, ],
    })

# Dynamic hack
if TEST_WrapRange:
    configuration.update({
        # Policies that should be simulated, and their parameters.
        "policies": [
            # --- UCB
            {
                "archtype": UCB, "append_label": " on $[0,1]$",
                "params": {
                    "lower": 0.0,
                    "amplitude": 1.0,
                }
            },
            {
                "archtype": WrapRange,
                "params": {
                    "policy": UCB
                }
            },
            # Reference policy knowing the range
            {
                "archtype": UCB, "append_label": " on $[{:.3g},{:.3g}]$".format(LOWER, LOWER + AMPLITUDE),
                "params": {
                    "lower": LOWER,
                    "amplitude": AMPLITUDE,
                }
            },
            # --- Thompson
            # # Thompson (and any BayesianIndexPolicy) fails when receiving a reward outside its range, so the first Thompson should fail!
            # {
            #     "archtype": Thompson, "append_label": " on $[0,1]$",
            #     "params": {
            #         "lower": 0.0,
            #         "amplitude": 1.0,
            #     }
            # },
            {
                "archtype": WrapRange,
                "params": {
                    "policy": Thompson
                }
            },
            # Reference policy knowing the range
            {
                "archtype": Thompson, "append_label": " on $[{:.3g},{:.3g}]$".format(LOWER, LOWER + AMPLITUDE),
                "params": {
                    "lower": LOWER,
                    "amplitude": AMPLITUDE,
                }
            },
            # --- klUCB
            {
                "archtype": klUCB, "append_label": " on $[0,1]$",
                "params": {
                    "lower": 0.0,
                    "amplitude": 1.0,
                }
            },
            {
                "archtype": WrapRange,
                "params": {
                    "policy": klUCB
                }
            },
            # Reference policy knowing the range
            {
                "archtype": klUCB, "append_label": " on $[{:.3g},{:.3g}]$".format(LOWER, LOWER + AMPLITUDE),
                "params": {
                    "lower": LOWER,
                    "amplitude": AMPLITUDE,
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


# # Custom klucb function
# _klucbGauss = klucbGauss


# def klucbGauss(x, d, precision=0.):
#     """klucbGauss(x, d, sig2x) with the good variance."""
#     # return _klucbGauss(x, d, 0.25)
#     return _klucbGauss(x, d, VARIANCE)


# # XXX to test just a few algorithms
# configuration.update({
#     "policies": [
#         # --- UCBalpha algorithms
#         {
#             "archtype": UCBalpha,
#             "params": {
#                 "alpha": 1
#             }
#         },
#         # --- KL UCB algorithms
#         {
#             "archtype": klUCBPlus,
#             "params": {
#                 "klucb": klucbGauss,
#             }
#         },
#     ]
# })

# # XXX Only test with scenario 1 from [A.Beygelzimer, J.Langfor, L.Li et al, AISTATS 2011]
# from PoliciesMultiPlayers import Scenario1  # XXX remove after testing once
# NB_PLAYERS = 10
# configuration.update({
#     "policies": Scenario1(NB_PLAYERS, nbArms).children
# })


# XXX Huge hack! Use this if you want to modify the legends
configuration.update({
    "append_labels": {
        policyId: cfg_policy.get("append_label", "")
        for policyId, cfg_policy in enumerate(configuration["policies"])
        if "append_label" in cfg_policy
    },
    "change_labels": {
        policyId: cfg_policy.get("change_label", "")
        for policyId, cfg_policy in enumerate(configuration["policies"])
        if "change_label" in cfg_policy
    }
})

print("Loaded experiments configuration from 'configuration.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
print("configuration['environment'] =", configuration["environment"])  # DEBUG
