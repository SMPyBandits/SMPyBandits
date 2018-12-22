# -*- coding: utf-8 -*-
"""
Configuration for the simulations, to test all the single-player policies.
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
    print("Warning: this script 'configuration_all_singleplayer.py' is NOT executable. Use 'main.py configuration_all_singleplayer' or 'make all_singleplayer' ...")  # DEBUG
    exit(0)

# Import arms and algorithms
try:
    from Arms import *
    from Policies import *
    from Policies.Experimentals import *
except ImportError:
    from SMPyBandits.Arms import *
    from SMPyBandits.Policies import *
    from SMPyBandits.Policies.Experimentals import *

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
#: Warning: Should be >= 10 to be statistically trustworthy.
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
DO_PARALLEL = (REPETITIONS > 1 or REPETITIONS == -1) and DO_PARALLEL

#: Number of jobs to use for the parallel computations. -1 means all the CPU cores, 1 means no parallelization.
N_JOBS = -1 if DO_PARALLEL else 1
if CPU_COUNT > 4:  # We are on a server, let's be nice and not use all cores
    N_JOBS = min(CPU_COUNT, max(int(CPU_COUNT / 3), CPU_COUNT - 8))
N_JOBS = int(getenv('N_JOBS', N_JOBS))
if REPETITIONS == -1:
    REPETITIONS = max(N_JOBS, CPU_COUNT)


#: Parameters for the epsilon-greedy and epsilon-... policies.
EPSILON = 0.1
#: Temperature for the Softmax policies.
TEMPERATURE = 0.05


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
    # --- Should we plot the lower-bounds or not?
    "plot_lowerbound": True,  # XXX Default
    # "plot_lowerbound": False,
    # --- Cache rewards: use the same random rewards for the Aggregator[..] and the algorithms
    "cache_rewards": False,
    # --- Arms
    "environment": [  # XXX Bernoulli arms
        {   # The easier problem: 2 arms, one perfectly bad, one perfectly good
            "arm_type": Bernoulli,
            "params": [0, 1]
        },
        {   # Another very easy problem: 3 arms, two very bad, one bad
            "arm_type": Bernoulli,
            "params": [0.04, 0.05, 0.1]
        },
        {   # XXX A very easy problem, but it is used in a lot of articles
            "arm_type": Bernoulli,
            "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        # XXX Default!
        {   # A very easy problem (X arms), but it is used in a lot of articles
            "arm_type": ARM_TYPE,
            "params": uniformMeans(nbArms=NB_ARMS, delta=1./(1. + NB_ARMS), lower=LOWER, amplitude=AMPLITUDE)
        },
        {   # VERY HARD One optimal arm, much better than the others, but *lots* of bad arms (34 arms!)
            "arm_type": Bernoulli,
            "params": [0.001, 0.001, 0.001, 0.001, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2, 0.5]
        },
        # {   # A Bayesian problem: every repetition use a different mean vectors!
        #     "arm_type": ARM_TYPE,
        #     "params": {
        #         "function": randomMeans,
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
}

if ENVIRONMENT_BAYESIAN:
    configuration["environment"] = [  # XXX Bernoulli arms
        {   # A Bayesian problem: every repetition use a different mean vectors!
            "arm_type": ARM_TYPE,
            "params": {
                "function": randomMeans,
                "args": {
                    "nbArms": NB_ARMS,
                    "mingap": None,
                    # "mingap": 0.0000001,
                    # "mingap": 0.1,
                    # "mingap": 1. / (3 * NB_ARMS),
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
    nbArms = len(configuration['environment'][0]['params'])

#: Warning: if using Exponential or Gaussian arms, gives klExp or klGauss to KL-UCB-like policies!
klucb = klucb_mapping.get(str(configuration['environment'][0]['arm_type']), klucbBern)

configuration.update({
    "policies": [
        # --- Stupid algorithms
        {
            "archtype": Uniform,   # The stupidest policy, fully uniform
            "params": {}
        },
        {
            "archtype": UniformOnSome,   # The stupidest policy, fully uniform
            "params": {
                "armIndexes": list(range(0, min(nbArms, max(nbArms-3, 2)))),
            }
        },
        {
            "archtype": EmpiricalMeans,   # The naive policy, just using empirical means
            "params": {}
        },
        {
            "archtype": TakeRandomFixedArm,   # The stupidest policy
            "params": {}
        },
        # --- Full or partial knowledge algorithms
        {
            "archtype": UniformOnSome,   # The stupidest policy, fully uniform
            "params": {
                "armIndexes": list(range(0, min(nbArms, max(nbArms-3, 2)))),
            }
        },
        {
            "archtype": TakeFixedArm,   # The stupidest policy, fully uniform
            "params": {
                "armIndex": nbArms - 1,  # Take best arm!,
            }
        },
        {
            "archtype": TakeFixedArm,   # The stupidest policy, fully uniform
            "params": {
                "armIndex": nbArms - 2,  # Take second best arm!,
            }
        },
        {
            "archtype": TakeFixedArm,   # The stupidest policy, fully uniform
            "params": {
                "armIndex": 0,  # Take worse arm!,
            }
        },
        {
            "archtype": TakeFixedArm,   # The stupidest policy, fully uniform
            "params": {
                "armIndex": 1,  # Take second worse arm!,
            }
        },
        # --- Epsilon-... algorithms
        {
            "archtype": EpsilonGreedy,   # This basic EpsilonGreedy is very bad
            "params": {
                "epsilon": EPSILON,
            }
        },
        {
            "archtype": EpsilonDecreasing,   # This basic EpsilonGreedy is also very bad
            "params": {
                "epsilon": EPSILON,
            }
        },
        {
            "archtype": EpsilonExpDecreasing,   # This basic EpsilonGreedy is also very bad
            "params": {
                "epsilon": EPSILON,
                "decreasingRate": 0.005,
            }
        },
        {
            "archtype": EpsilonFirst,   # This basic EpsilonFirst is also very bad
            "params": {
                "epsilon": EPSILON,
                "horizon": HORIZON,
            }
        },
        # --- Explore-Then-Commit policies
        {
            "archtype": ETC_KnownGap,
            "params": {
                "horizon": HORIZON,
                "gap": 0.05,
            }
        },
        {
            "archtype": ETC_RandomStop,
            "params": {
                "horizon": HORIZON,
            }
        },
        # --- Softmax algorithms
        {
            "archtype": Softmax,   # This basic Softmax is very bad
            "params": {
                "temperature": TEMPERATURE,
            }
        },
        {
            "archtype": SoftmaxDecreasing,   # XXX Efficient parameter-free Softmax
            "params": {}
        },
        {
            "archtype": SoftMix,   # Another parameter-free Softmax
            "params": {}
        },
        {
            "archtype": SoftmaxWithHorizon,  # Other Softmax, knowing the horizon
            "params": {
                "horizon": HORIZON,
            }
        },
        # --- Boltzmann-Gumbel algorithms
        {
            "archtype": BoltzmannGumbel,
            "params": {
                "C": 0.5,
            }
        },
        # --- Exp3 algorithms - Very bad !!!!
        {
            "archtype": Exp3,   # This basic Exp3 is not very good
            "params": {
                "gamma": 0.001,
            }
        },
        {
            "archtype": Exp3Decreasing,
            "params": {
                "gamma": 0.001,
            }
        },
        {
            "archtype": Exp3SoftMix,   # Another parameter-free Exp3
            "params": {}
        },
        {
            "archtype": Exp3WithHorizon,  # Other Exp3, knowing the horizon
            "params": {
                "horizon": HORIZON,
            }
        },
        {
            "archtype": Exp3ELM,   # This improved Exp3 is not better, it targets a different problem
            "params": {
                "delta": 0.1,
            }
        },
        # --- Exp3PlusPlus algorithm
        {
            "archtype": Exp3PlusPlus,   # Another parameter-free Exp3, better parametrization
            "params": {}
        },
        # --- Probability pursuit algorithm
        {
            "archtype": ProbabilityPursuit,
            "params": {
                "beta": 0.5,
            }
        },
        # --- Hedge algorithm
        {
            "archtype": Hedge,
            "params": {
                "epsilon": 0.5,
            }
        },
        {
            "archtype": HedgeDecreasing,
            "params": {}
        },
        {
            "archtype": HedgeWithHorizon,
            "params": {
                "horizon": HORIZON,
            }
        },
        # --- UCB algorithms
        {
            "archtype": UCB,   # This basic UCB is very worse than the other
            "params": {}
        },
        {
            "archtype": UCBlog10,   # This basic UCB is very worse than the other
            "params": {}
        },
        {
            "archtype": UCBwrong,  # This wrong UCB is very very worse than the other
            "params": {}
        },
        {
            "archtype": UCBalpha,   # UCB with custom alpha parameter
            "params": {
                "alpha": 1,
            }
        },
        {
            "archtype": UCBlog10alpha,   # UCB with custom alpha parameter
            "params": {
                "alpha": 1,
            }
        },
        {
            "archtype": UCBmin,
            "params": {}
        },
        {
            "archtype": UCBplus,
            "params": {}
        },
        {
            "archtype": UCBrandomInit,
            "params": {}
        },
        # {
        #     "archtype": UCBjulia,  # WARNING
        #     "params": {}
        # },
        {
            "archtype": UCBcython,  # WARNING
            "params": {}
        },
        {
            "archtype": UCBV,   # UCB with variance term
            "params": {}
        },
        {
            "archtype": UCBVtuned,   # UCB with variance term and one trick
            "params": {}
        },
        {
            "archtype": SWUCB,   # XXX experimental sliding window algorithm
            "params": {}
        },
        {
            "archtype": SWUCBPlus,   # XXX experimental sliding window algorithm
            "params": {}
        },
        {
            "archtype": DiscountedUCB,   # XXX experimental discounted reward algorithm
            "params": {}
        },
        {
            "archtype": DiscountedUCBPlus,   # XXX experimental discounted reward algorithm
            "params": {}
        },
        {
            "archtype": SWR_UCB,   # XXX experimental sliding window algorithm
            "params": {}
        },
        {
            "archtype": SWR_UCBalpha,   # XXX experimental sliding window algorithm
            "params": {
                "alpha": 0.5,
            }
        },
        # --- SparseUCB and variants policies for sparse stochastic bandit
        {
            "archtype": SparseUCB,
            "params": {
                "alpha": 4,
                "sparsity": min(nbArms, 3),
            }
        },
        {
            "archtype": SparseklUCB,
            "params": {
                "sparsity": min(nbArms, 3),
            }
        },
        # --- MOSS algorithm, like UCB
        {
            "archtype": MOSS,
            "params": {}
        },
        # --- MOSS-H algorithm, like UCB-H
        {
            "archtype": MOSSH,
            "params": {
                "horizon": HORIZON,
            }
        },
        # --- MOSS-Anytime algorithm, extension of MOSS
        {
            "archtype": MOSSAnytime,
            "params": {
                "alpha": 1.35,
            }
        },
        # --- MOSS-Experimental algorithm, extension of MOSSAnytime
        {
            "archtype": MOSSExperimental,
            "params": {}
        },
        # --- Optimally-Confident UCB algorithm
        {
            "archtype": OCUCB,
            "params": {
                "eta": 1.1,
                "rho": 1,
            }
        },
        # --- CPUCB algorithm, other variant of UCB
        {
            "archtype": CPUCB,
            "params": {}
        },
        # --- DMED algorithm, similar to klUCB
        {
            "archtype": DMEDPlus,
            "params": {}
        },
        {
            "archtype": DMED,
            "params": {}
        },
        # --- Thompson algorithms
        {
            "archtype": Thompson,
            "params": {
                "posterior": Beta,
            }
        },
        {
            "archtype": Thompson,
            "params": {
                "posterior": Gauss,
            }
        },
        {
            "archtype": ThompsonRobust,
            "params": {
                "posterior": Beta,
            }
        },
        # --- KL algorithms
        {
            "archtype": klUCB,
            "params": {
                "klucb": klucb,
            }
        },
        {
            "archtype": SWR_klUCB,   # XXX experimental sliding window algorithm
            "params": {
                "klucb": klucb,
            }
        },
        {
            "archtype": klUCBloglog,
            "params": {
                "klucb": klucb,
            }
        },
        {
            "archtype": klUCBlog10,
            "params": {
                "klucb": klucb
            }
        },
        {
            "archtype": klUCBloglog10,
            "params": {
                "klucb": klucb,
            }
        },
        {
            "archtype": klUCBPlus,
            "params": {
                "klucb": klucb,
            }
        },
        {
            "archtype": klUCBH,
            "params": {
                "horizon": HORIZON,
                "klucb": klucb,
            }
        },
        {
            "archtype": klUCBHPlus,
            "params": {
                "horizon": HORIZON,
                "klucb": klucb,
            }
        },
        {
            "archtype": klUCBPlusPlus,
            "params": {
                "horizon": HORIZON,
                "klucb": klucb
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
        {
            "archtype": AdBandits,
            "params": {
                "alpha": 0.5,
                "horizon": HORIZON,
            }
        },
        # # --- Horizon-dependent algorithm ApproximatedFHGittins
        {
            "archtype": ApproximatedFHGittins,
            "params": {
                "alpha": 0.5,
                "horizon": max(HORIZON + 100, int(1.05 * HORIZON)),
            }
        },
        # --- Using unsupervised learning, from scikit-learn, XXX works well, but VERY SLOW
        {
            "archtype": UnsupervisedLearning,
            "params": {}
        },
        # # --- Black Box optimizer, using Gaussian Processes XXX works well, but VERY SLOW
        # {
        #     "archtype": BlackBoxOpt,
        #     "params": {}
        # },
        # --- The new OSSB algorithm
        {
            "archtype": OSSB,
            "params": {
                "epsilon": 0.01,
                "gamma": 0.0,
            }
        },
        # --- The awesome BESA algorithm
        {
            "archtype": BESA,
            "params": {
                "horizon": HORIZON,
                "minPullsOfEachArm": 1,  # Default, don't seem to improve if increasing this one
                "randomized_tournament": True,
                # "randomized_tournament": False,  # XXX Very inefficient!
                "random_subsample": True,
                # "random_subsample": False,  # XXX Very inefficient!
                "non_binary": False,
                # "non_binary": True,
                "non_recursive": False,
                # "non_recursive": True,
            }
        },
        # --- Auto-tuned UCBdagger algorithm
        {
            "archtype": UCBdagger,
            "params": {
                "horizon": HORIZON,
            }
        },
        # --- new UCBoost algorithms
        {
            "archtype": UCB_bq,
            "params": {}
        },
        {
            "archtype": UCB_h,
            "params": {}
        },
        {
            "archtype": UCB_lb,
            "params": {}
        },
        {
            "archtype": UCBoost_bq_h_lb,
            "params": {}
        },
        # # --- new UCBoostEpsilon algorithm
        {
            "archtype": UCBoostEpsilon,
            "params": {
                "epsilon": 0.1,
            }
        },
        # --- new UCBoost_cython algorithms
        {
            "archtype": UCB_bq_cython,
            "params": {}
        },
        {
            "archtype": UCB_h_cython,
            "params": {}
        },
        {
            "archtype": UCB_lb_cython,
            "params": {}
        },
        {
            "archtype": UCBoost_bq_h_lb_cython,
            "params": {}
        },
        # --- new UCBoostEpsilon_cython algorithm
        {
            "archtype": UCBoostEpsilon_cython,
            "params": {
                "epsilon": 0.1,
            }
        },
        {
            "archtype": UCBoostEpsilon_cython,
            "params": {
                "epsilon": 0.05,
            }
        },
        {
            "archtype": UCBoostEpsilon_cython,
            "params": {
                "epsilon": 0.01,
            }
        },
        # new UCBcython algorithm
        {
            "archtype": UCBcython,
            "params": {
                "alpha": 4.0,
            }
        },
        {
            "archtype": UCBcython,
            "params": {
                "alpha": 1.0,
            }
        },
        {
            "archtype": UCBcython,
            "params": {
                "alpha": 0.5,
            }
        },
        # XXX Regular adversarial bandits algorithms!
        {
            "archtype": Exp3PlusPlus,
            "params": {}
        },
        {
            "archtype": DiscountedThompson,
            "params": { "posterior": DiscountedBeta, "gamma": 0.99 }
        },
        # The Exp3R algorithm works reasonably well
        {
            "archtype": Exp3R,
            "params": { "horizon": HORIZON, }
        },
        # XXX The Exp3RPlusPlus variant of Exp3R algorithm works also reasonably well
        {
            "archtype": Exp3RPlusPlus,
            "params": { "horizon": HORIZON, }
        },

        # XXX TODO test the AdSwitch policy and its corrected version
        {
            "archtype": AdSwitch,
            "params": { "horizon": HORIZON, }
        },
        {
            "archtype": LM_DSEE,
            "params": { "nu": 0.25, "DeltaMin": 0.1, "a": 1, "b": 0.25, }
        },
        # XXX Test a few CD-MAB algorithms that need to know NB_BREAK_POINTS
        {
            "archtype": CUSUM_IndexPolicy,
            "params": { "horizon": HORIZON, "max_nb_random_events": 1, "policy": UCB, "per_arm_restart": True, }
        },
        {
            "archtype": PHT_IndexPolicy,
            "params": { "horizon": HORIZON, "max_nb_random_events": 1, "policy": UCB, "per_arm_restart": True, }
        },
        # DONE The SW_UCB_Hash algorithm works fine!
        {
            "archtype": SWHash_IndexPolicy,
            "params": { "alpha": 1, "lmbda": 1, "policy": UCB }
        },
        # --- # XXX experimental sliding window algorithm
        {
            "archtype": SlidingWindowRestart,
            "params": { "policy": UCB }
        },
        # --- # Different versions of the sliding window UCB algorithm
        {
            "archtype": SWUCB,
            "params": { "alpha": 1, "tau": 500, }
        },
        # --- # XXX experimental other version of the sliding window algorithm, knowing the horizon
        {
            "archtype": SWUCBPlus,
            "params": { "horizon": HORIZON, "alpha": 1, }
        },
        # --- # Different versions of the discounted UCB algorithm
        {
            "archtype": DiscountedUCB,
            "params": { "alpha": 1, "gamma": 0.9 }
        },
        # --- # XXX experimental discounted UCB algorithm, knowing the horizon
        {
            "archtype": DiscountedUCBPlus,
            "params": { "max_nb_random_events": 1, "alpha": 1, "horizon": HORIZON, }
        },
        # XXX The Monitored_IndexPolicy works but the default choice of parameters seem bad!
        {
            "archtype": Monitored_IndexPolicy,
            "params": { "horizon": HORIZON, "max_nb_random_events": 1, "delta": 0.1, "policy": UCB, }
        },
        # XXX The Monitored_IndexPolicy with specific tuning of the input parameters
        {
            "archtype": Monitored_IndexPolicy,
            "params": { "policy": UCB, "horizon": HORIZON, "w": 80, "b": np.sqrt(80/2 * np.log(2 * NB_ARMS * HORIZON**2)), }
        },
        # DONE the OracleSequentiallyRestartPolicy with klUCB/UCB policy works quite well, but NOT optimally!
        {
            "archtype": OracleSequentiallyRestartPolicy,
            "params": { "changePoints": [], "policy": UCB, "per_arm_restart": True }
        },
        # XXX Test a few CD-MAB algorithms
        {
            "archtype": BernoulliGLR_IndexPolicy,
            "params": { "horizon": HORIZON, "policy": UCB, "per_arm_restart": True, "max_nb_random_events": 1 }
        },
        {
            "archtype": BernoulliGLR_IndexPolicy_WithTracking,
            "params": { "horizon": HORIZON, "policy": UCB, "per_arm_restart": True, "max_nb_random_events": 1 }
        },
        {
            "archtype": GaussianGLR_IndexPolicy,
            "params": { "horizon": HORIZON, "policy": UCB, "per_arm_restart": True, "max_nb_random_events": 1 }
        },
        {
            "archtype": GaussianGLR_IndexPolicy_WithTracking,
            "params": { "horizon": HORIZON, "policy": UCB, "per_arm_restart": True, "max_nb_random_events": 1 }
        },
        {
            "archtype": SubGaussianGLR_IndexPolicy,
            "params": { "horizon": HORIZON, "policy": UCB, "per_arm_restart": True, "max_nb_random_events": 1 }
        },
    ]
})



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
elif not ENVIRONMENT_BAYESIAN:
    configuration.update({
        "environment": [ {
                "arm_type": ARM_TYPE,
                "params": uniformMeans(nbArms=NB_ARMS, delta=1./(1. + NB_ARMS), lower=LOWER, amplitude=AMPLITUDE)
            }, ],
    })


# XXX Huge hack! Use this if you want to modify the legends
configuration.update({
    "append_labels": {
        policyId: cfgpolicy.get("append_label", "")
        for policyId, cfgpolicy in enumerate(configuration["policies"])
        if "append_label" in cfgpolicy
    },
    "change_labels": {
        policyId: cfgpolicy.get("change_label", "")
        for policyId, cfgpolicy in enumerate(configuration["policies"])
        if "change_label" in cfgpolicy
    }
})

print("Loaded experiments configuration from 'configuration_all_singleplayer.py' :")
print("configuration['policies'] =", configuration["policies"])  # DEBUG
