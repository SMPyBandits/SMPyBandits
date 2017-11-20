# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the single-player case.
"""
from __future__ import print_function, division

__author__ = "Lilian Besson"
__version__ = "0.8"

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

# Import arms
from Arms import *

# Import algorithms
from Policies import *

#: HORIZON : number of time steps of the experiments.
#: Warning Should be >= 10000 to be interesting "asymptotically".
HORIZON = 100
HORIZON = 500
HORIZON = 2000
HORIZON = 3000
HORIZON = 5000
HORIZON = 10000
# HORIZON = 20000
# HORIZON = 30000
# HORIZON = 40000
# HORIZON = 100000
HORIZON = int(getenv('T', HORIZON))

#: REPETITIONS : number of repetitions of the experiments.
#: Warning: Should be >= 10 to be stastically trustworthy.
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
# REPETITIONS = 10000
# REPETITIONS = 1000
# REPETITIONS = 200
# REPETITIONS = 100
# REPETITIONS = 50
# REPETITIONS = 20
REPETITIONS = int(getenv('N', REPETITIONS))

#: To profile the code, turn down parallel computing
DO_PARALLEL = False  # XXX do not let this = False
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1) and DO_PARALLEL

#: Number of jobs to use for the parallel computations. -1 means all the CPU cores, 1 means no parallelization.
N_JOBS = -1 if DO_PARALLEL else 1
if CPU_COUNT > 4:  # We are on a server, let's be nice and not use all cores
    N_JOBS = min(CPU_COUNT, max(int(CPU_COUNT / 3), CPU_COUNT - 8))
N_JOBS = int(getenv('N_JOBS', N_JOBS))

# Random events
RANDOM_SHUFFLE = False  #: The arms won't be shuffled (``shuffle(arms)``).
# RANDOM_SHUFFLE = True  #: The arms will be shuffled (``shuffle(arms)``).
RANDOM_INVERT = False  #: The arms won't be inverted (``arms = arms[::-1]``).
# RANDOM_INVERT = True  #: The arms will be inverted (``arms = arms[::-1]``).
NB_RANDOM_EVENTS = 3  #: Number of true breakpoints. They are uniformly spaced in time steps (and the first one at t=0 does not count).
# NB_RANDOM_EVENTS = 5  #: Number of true breakpoints. They are uniformly spaced in time steps (and the first one at t=0 does not count).
# NB_RANDOM_EVENTS = 10  #: Number of true breakpoints. They are uniformly spaced in time steps (and the first one at t=0 does not count).
NB_RANDOM_EVENTS = 20  #: Number of true breakpoints. They are uniformly spaced in time steps (and the first one at t=0 does not count).

#: Parameters for the epsilon-greedy and epsilon-... policies.
EPSILON = 0.1
#: Temperature for the Softmax policies.
TEMPERATURE = 0.01  # When -> 0, more greedy
TEMPERATURE = 0.1
TEMPERATURE = 0.5
TEMPERATURE = 1
TEMPERATURE = 10
TEMPERATURE = 100   # When -> oo, more uniformly at random
# TEMPERATURE = 10.0 / HORIZON  # Not sure ??!
TEMPERATURE = 0.05

#: Learning rate for my aggregated bandit (it can be autotuned)
LEARNING_RATE = 0.05
LEARNING_RATE = 0.1
LEARNING_RATE = 0.2
LEARNING_RATE = 0.5
LEARNING_RATE = 0.01

# To try more learning rates in one run
LEARNING_RATES = [10, 2, 1, 0.1, 0.01, 0.001, 0.0001, 0.00005]
LEARNING_RATES = [10, 1, 0.1, 0.01, 0.001]
LEARNING_RATES = [LEARNING_RATE]

#: Constant time tau for the decreasing rate for my aggregated bandit.
# FIXED I tried to make self.learningRate decrease when self.t increase, it was not better
DECREASE_RATE = None
DECREASE_RATE = HORIZON / 2.0
DECREASE_RATE = 'auto'  # FIXED using the formula from Theorem 4.2 from [Bubeck & Cesa-Bianchi, 2012](http://sbubeck.com/SurveyBCB12.pdf)

#: To know if my Aggregator policy is tried.
TEST_Aggregator = True
TEST_Aggregator = False  # XXX do not let this = False if you want to test my Aggregator policy

#: Should we cache rewards? The random rewards will be the same for all the REPETITIONS simulations for each algorithms.
CACHE_REWARDS = TEST_Aggregator
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
VARIANCE = 10   #: Variance of Gaussian arms
VARIANCE = 0.05   #: Variance of Gaussian arms

#: Number of arms for non-hard-coded problems (Bayesian problems)
NB_ARMS = 9
NB_ARMS = int(getenv('K', NB_ARMS))
NB_ARMS = int(getenv('NB_ARMS', NB_ARMS))

#: Type of arms for non-hard-coded problems (Bayesian problems)
ARM_TYPE = "Bernoulli"
ARM_TYPE = str(getenv('ARM_TYPE', ARM_TYPE))
mapping_ARM_TYPE = {
    "Constant": Constant,
    "Uniform": Uniform,
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
    "environment": [  # XXX Bernoulli arms
        # {   # The easier problem: 2 arms, one perfectly bad, one perfectly good
        #     "arm_type": Bernoulli,
        #     "params": [0, 1]
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
        # {   # XXX A very easy problem, but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # },
        # # XXX Default!
        # {   # A very easy problem (X arms), but it is used in a lot of articles
        #     "arm_type": ARM_TYPE,
        #     "params": uniformMeans(NB_ARMS, 1 / (1. + NB_ARMS))
        # },
        # {   # An other problem, best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3 - 0.6) and very good arms (0.78, 0.8, 0.82)
        #     "arm_type": Bernoulli,
        #     "params": [0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.78, 0.8, 0.82]
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
        {   # A Bayesian problem: every repetition use a different mean vectors!
            "arm_type": ARM_TYPE,
            "params": {
                "function": randomMeans,
                "args": {
                    "nbArms": NB_ARMS,
                    "mingap": None,
                    # "mingap": 0.01,
                    # "mingap": 0.1,
                    # "mingap": 1. / (3 * NB_ARMS),
                    "lower": 0.,
                    "amplitude": 1.,
                    # "isSorted": False,
                    "isSorted": True,
                }
            }
        },
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
    #         "params": [(0.2, VARIANCE), (0.5, VARIANCE), (0.8, VARIANCE)]
    #     },
    #     # {   # An example problem with 9 arms
    #     #     "arm_type": Gaussian,
    #     #     "params": [(0.1, VARIANCE), (0.2, VARIANCE), (0.3, VARIANCE), (0.4, VARIANCE), (0.5, VARIANCE), (0.6, VARIANCE), (0.7, VARIANCE), (0.8, VARIANCE), (0.9, VARIANCE)]
    #     # },
    # ],
    # "environment": [  # XXX Unbounded Gaussian arms
    #     {   # An example problem with 9 arms
    #         "arm_type": UnboundedGaussian,
    #         "params": [(-40, VARIANCE), (-30, VARIANCE), (-20, VARIANCE), (-VARIANCE, VARIANCE), (0, VARIANCE), (VARIANCE, VARIANCE), (20, VARIANCE), (30, VARIANCE), (40, VARIANCE)]
    #     },
    # ],
}

# if len(configuration['environment']) > 1:
#     raise ValueError("WARNING do not use this hack if you try to use more than one environment.")
#     # Note: I dropped the support for more than one environments, for this part of the configuration, but not the simulation code

try:
    #: Number of arms *in the first environment*
    nbArms = int(configuration['environment'][0]['params']['args']['nbArms'])
except (TypeError, KeyError):
    nbArms = len(configuration['environment'][0]['params'])

#: Warning: if using Exponential or Gaussian arms, gives klExp or klGauss to KL-UCB-like policies!
klucb = klucb_mapping.get(str(configuration['environment'][0]['arm_type']), klucbBern)


configuration.update({
    "policies": [
        # # --- Stupid algorithms
        # {
        #     "archtype": Uniform,   # The stupidest policy, fully uniform
        #     "params": {}
        # },
        # {
        #     "archtype": EmpiricalMeans,   # The naive policy, just using empirical means
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
        # TakeFixedArm(nbArms, 1),  # Take second worse arm!
        # # --- Epsilon-... algorithms
        # {
        #     "archtype": EpsilonGreedy,   # This basic EpsilonGreedy is very bad
        #     "params": {
        #         "epsilon": EPSILON,
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
        #         "horizon": HORIZON,
        #     }
        # },
        # --- Explore-Then-Commit policies
        # {
        #     "archtype": ETC_KnownGap,
        #     "params": {
        #         "horizon": HORIZON,
        #         "gap": 0.1,
        #     }
        # },
        # {
        #     "archtype": ETC_KnownGap,
        #     "params": {
        #         "horizon": HORIZON,
        #         "gap": 0.05,
        #     }
        # },
        # {
        #     "archtype": ETC_KnownGap,
        #     "params": {
        #         "horizon": HORIZON,
        #         "gap": 0.01,
        #     }
        # },
        # {
        #     "archtype": ETC_KnownGap,
        #     "params": {
        #         "horizon": HORIZON,
        #         "gap": 0.5
        #     }
        # },
        # {
        #     "archtype": ETC_RandomStop,
        #     "params": {
        #         "horizon": HORIZON,
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
        # --- UCB algorithms
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
        {
            "archtype": UCBalpha,   # UCB with custom alpha parameter
            "params": {
                "alpha": 0.5,       # XXX Below the theoretically acceptable value!
            }
        },
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
        {
            "archtype": UCBalpha,   # UCB with custom alpha parameter
            "params": {
                "alpha": 0.05,      # XXX Below the theoretically acceptable value!
            }
        },
        # --- MOSS algorithm, like UCB
        {
            "archtype": MOSS,
            "params": {}
        },
        # # --- Optimally-Confident UCB algorithm
        # {
        #     "archtype": OCUCB,
        #     "params": {
        #         "eta": 1.1,
        #         "rho": 1,
        #     }
        # },
        # {
        #     "archtype": OCUCB,
        #     "params": {
        #         "eta": 1.1,
        #         "rho": 0.9,
        #     }
        # },
        # {
        #     "archtype": OCUCB,
        #     "params": {
        #         "eta": 1.1,
        #         "rho": 0.8,
        #     }
        # },
        # {
        #     "archtype": OCUCB,
        #     "params": {
        #         "eta": 1.1,
        #         "rho": 0.7,
        #     }
        # },
        # {
        #     "archtype": OCUCB,
        #     "params": {
        #         "eta": 1.1,
        #         "rho": 0.6,
        #     }
        # },
        # --- CPUCB algorithm, other variant of UCB
        # {
        #     "archtype": CPUCB,
        #     "params": {}
        # },
        # --- DMED algorithm, similar to klUCB
        {
            "archtype": DMED,
            "params": {
                "genuine": True,
            }
        },
        # {
        #     "archtype": DMED,
        #     "params": {
        #         "genuine": False,
        #     }
        # },
        # --- Thompson algorithms
        {
            "archtype": Thompson,
            "params": {
                "posterior": Beta,
            }
        },
        # {
        #     "archtype": Thompson,
        #     "params": {
        #         "posterior": Gauss,
        #     }
        # },
        # --- KL algorithms
        {
            "archtype": klUCB,
            "params": {
                "klucb": klucb,
            }
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
        {
            "archtype": klUCBPlus,
            "params": {
                "klucb": klucb,
            }
        },
        # {
        #     "archtype": klUCBHPlus,
        #     "params": {
        #         "horizon": HORIZON,
        #         "klucb": klucb,
        #     }
        # },
        {
            "archtype": klUCBPlusPlus,
            "params": {
                "horizon": HORIZON,
                "klucb": klucb
            }
        },
        # --- Doubling trick algorithm
        {
            "archtype": DoublingTrickWrapper,
            "params": {
                "next_horizon": next_horizon__arithmetic,
                "policy": klUCBPlusPlus,
                "klucb": klucb,
            }
        },
        {
            "archtype": DoublingTrickWrapper,
            "params": {
                "next_horizon": next_horizon__geometric,
                "policy": klUCBPlusPlus,
                "klucb": klucb,
            }
        },
        {
            "archtype": DoublingTrickWrapper,
            "params": {
                "next_horizon": next_horizon__exponential,
                "policy": klUCBPlusPlus,
                "klucb": klucb,
            }
        },
        {
            "archtype": DoublingTrickWrapper,
            "params": {
                "next_horizon": next_horizon__exponential_fast,
                "policy": klUCBPlusPlus,
                "klucb": klucb,
            }
        },
        {
            "archtype": DoublingTrickWrapper,
            "params": {
                "next_horizon": next_horizon__exponential_slow,
                "policy": klUCBPlusPlus,
                "klucb": klucb,
            }
        },
        # # --- Empirical KL-UCB algorithm
        # {
        #     "archtype": KLempUCB,
        #     "params": {}
        # },
        # --- Bayes UCB algorithms
        {
            "archtype": BayesUCB,
            "params": {
                "posterior": Beta,
            }
        },
        # {
        #     "archtype": BayesUCB,
        #     "params": {
        #         "posterior": Gauss,  # XXX does not work yet!
        #     }
        # },
        # --- AdBandits with different alpha paramters
        # {
        #     "archtype": AdBandits,
        #     "params": {
        #         "alpha": 0.5,
        #         "horizon": HORIZON,
        #     }
        # },
        # {
        #     "archtype": AdBandits,
        #     "params": {
        #         "alpha": 0.125,
        #         "horizon": HORIZON,
        #     }
        # },
        # {
        #     "archtype": AdBandits,
        #     "params": {
        #         "alpha": 0.01,
        #         "horizon": HORIZON,
        #     }
        # },
        # # --- Horizon-dependent algorithm ApproximatedFHGittins
        # {
        #     "archtype": ApproximatedFHGittins,
        #     "params": {
        #         "alpha": 4,
        #         "horizon": 1.1 * HORIZON,
        #     }
        # },
        # {
        #     "archtype": ApproximatedFHGittins,
        #     "params": {
        #         "alpha": 1,
        #         "horizon": 1.1 * HORIZON,
        #     }
        # },
        {
            "archtype": ApproximatedFHGittins,
            "params": {
                "alpha": 0.5,
                "horizon": 1.1 * HORIZON,
            }
        },
        # --- Doubling trick algorithm
        {
            "archtype": DoublingTrickWrapper,
            "params": {
                "next_horizon": next_horizon__arithmetic,
                "policy": ApproximatedFHGittins,
                "alpha": 0.5,
            }
        },
        {
            "archtype": DoublingTrickWrapper,
            "params": {
                "next_horizon": next_horizon__geometric,
                "policy": ApproximatedFHGittins,
                "alpha": 0.5,
            }
        },
        {
            "archtype": DoublingTrickWrapper,
            "params": {
                "next_horizon": next_horizon__exponential,
                "policy": ApproximatedFHGittins,
                "alpha": 0.5,
            }
        },
        {
            "archtype": DoublingTrickWrapper,
            "params": {
                "next_horizon": next_horizon__exponential_fast,
                "policy": ApproximatedFHGittins,
                "alpha": 0.5,
            }
        },
        {
            "archtype": DoublingTrickWrapper,
            "params": {
                "next_horizon": next_horizon__exponential_slow,
                "policy": ApproximatedFHGittins,
                "alpha": 0.5,
            }
        },
        # --- Black Box optimizer, using Gaussian Processes XXX works well, but VERY SLOW
        # {
        #     "archtype": BlackBoxOpt,
        #     "params": {
        #     }
        # },
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
#     """klucbGauss(x, d, sig2) with the good variance."""
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

# configuration.update({
#     "policies": [
#         # --- Empirical KL-UCB algorithm
#         {
#             "archtype": KLempUCB,
#             "params": {}
#         },
#     ]
# })


# # XXX compare different values of the experimental sliding window algorithm
# EPSS   = [0.1, 0.05]
# ALPHAS = [2, 1, 0.5, 0.1]
# ALPHAS = [2, 0.5, 0.1]
# ALPHAS = [0.5]
# ALPHAS = [1]
# TAUS   = [
#         500, 1000, 2000,
#         # 2 * np.sqrt(HORIZON * np.log(HORIZON) / (1 + NB_RANDOM_EVENTS))  # "optimal" value according to [Garivier & Moulines, 2008]
#     ]
# GAMMAS = [
#         # 0.1, 0.2, 0.3, 0.4, 0.5, 0.7,
#         0.8, 0.9, 0.95, 0.99, 0.999999,
#        # (1 - np.sqrt((1 + NB_RANDOM_EVENTS) / HORIZON)) / 4.  # "optimal" value according to [Garivier & Moulines, 2008]
#     ]

# configuration.update({
#     "policies":
#     # [
#     #     # --- # XXX experimental sliding window algorithm
#     #     {
#     #         "archtype": SlidingWindowRestart(Policy=UCBalpha, tau=tau, threshold=eps, full_restart_when_refresh=True),
#     #         "params": {
#     #             "alpha": alpha
#     #         }
#     #     }
#     #     for tau in TAUS
#     #     for eps in EPSS
#     #     for alpha in ALPHAS
#     # # ] +
#     [
#         # --- # XXX experimental other version of the sliding window algorithm
#         {
#             "archtype": SWUCB,
#             "params": {
#                 "alpha": alpha,
#                 "tau": tau
#             }
#         }
#         for alpha in ALPHAS
#         for tau in TAUS
#     ] +
#     [
#         # --- # XXX experimental other version of the sliding window algorithm, knowing the horizon
#         {
#             "archtype": SWUCBPlus,
#             "params": {
#                 "horizon": HORIZON,
#                 "alpha": alpha
#             }
#         }
#         for alpha in ALPHAS
#     ] +
#     [
#         # --- # XXX experimental discounted UCB algorithm
#         {
#             "archtype": DiscountedUCB,
#             "params": {
#                 "alpha": alpha,
#                 "gamma": gamma
#             }
#         }
#         for gamma in GAMMAS
#         for alpha in ALPHAS
#     ] +
#     [
#         # --- # XXX experimental discounted UCB algorithm, knowing the horizon
#         {
#             "archtype": DiscountedUCBPlus,
#             "params": {
#                 "alpha": alpha,
#                 "horizon": HORIZON
#             }
#         }
#         for alpha in ALPHAS
#     ] +
#     [
#         {
#             "archtype": UCBalpha,
#             "params": {
#                 "alpha": alpha
#             }
#         }
#         for alpha in ALPHAS
#     ]
# })

# # XXX Only test with scenario 1 from [A.Beygelzimer, J.Langfor, L.Li et al, AISTATS 2011]
# from PoliciesMultiPlayers import Scenario1  # XXX remove after testing once
# NB_PLAYERS = 10
# configuration.update({
#     "policies": Scenario1(NB_PLAYERS, nbArms).children
# })


NON_AGGR_POLICIES_1 = [
    # --- Doubling trick algorithm
    {
        "archtype": DoublingTrickWrapper,
        "params": {
            "next_horizon": next_horizon__arithmetic,
            "policy": ApproximatedFHGittins,
            "alpha": 0.5,
        }
    },
    {
        "archtype": DoublingTrickWrapper,
        "params": {
            "next_horizon": next_horizon__geometric,
            "policy": ApproximatedFHGittins,
            "alpha": 0.5,
        }
    },
    {
        "archtype": DoublingTrickWrapper,
        "params": {
            "next_horizon": next_horizon__exponential,
            "policy": ApproximatedFHGittins,
            "alpha": 0.5,
        }
    },
    {
        "archtype": DoublingTrickWrapper,
        "params": {
            "next_horizon": next_horizon__exponential_fast,
            "policy": ApproximatedFHGittins,
            "alpha": 0.5,
        }
    },
    {
        "archtype": DoublingTrickWrapper,
        "params": {
            "next_horizon": next_horizon__exponential_slow,
            "policy": ApproximatedFHGittins,
            "alpha": 0.5,
        }
    }
]

# from itertools import product  # XXX If needed!

# Dynamic hack to force the Aggregator (policies aggregator) to use all the policies previously/already defined
if TEST_Aggregator:
    NON_AGGR_POLICIES_0 = configuration["policies"]

    for NON_AGGR_POLICIES in [NON_AGGR_POLICIES_0, NON_AGGR_POLICIES_1]:
        # for LEARNING_RATE in LEARNING_RATES:  # XXX old code to test different static learning rates, not any more
        # for UNBIASED in [False, True]:  # XXX to test between biased or unabiased estimators
        # for (UNBIASED, UPDATE_LIKE_EXP4) in product([False, True], repeat=2):  # XXX If needed!
        # for (HORIZON, UPDATE_LIKE_EXP4) in product([None, HORIZON], [False, True]):  # XXX If needed!
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
                    # "horizon": HORIZON  # XXX uncomment to give the value of horizon to have a better learning rate
                },
            }]

print("Loaded experiments configuration from 'configuration.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
