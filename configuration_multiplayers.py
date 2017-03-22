# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the multi-players case.
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
import numpy as np

# Import arms
from Arms import *

# Import contained classes
from Environment import MAB

# Collision Models
from Environment.CollisionModels import *

# Import algorithms, both single-player and multi-player
from Policies import *
from PoliciesMultiPlayers import *
from PoliciesMultiPlayers.ALOHA import tnext_beta, tnext_log  # XXX do better for these imports


# HORIZON : number of time steps of the experiments
# XXX Should be >= 10000 to be interesting "asymptotically"
HORIZON = 500
HORIZON = 2000
HORIZON = 3000
HORIZON = 5000
HORIZON = 10000
# HORIZON = 20000
# HORIZON = 30000
# HORIZON = 40000
# HORIZON = 100000

# DELTA_T_SAVE : save only 1 / DELTA_T_SAVE points, to speed up computations, use less RAM, speed up plotting etc.
DELTA_T_SAVE = 1 * (HORIZON < 10000) + 50 * (10000 <= HORIZON < 100000) + 100 * (HORIZON >= 100000)
DELTA_T_SAVE = 1  # XXX to disable this optimization

# REPETITIONS : number of repetitions of the experiments
# XXX Should be >= 10 to be statistically trustworthy
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
REPETITIONS = 200
REPETITIONS = 100
REPETITIONS = 50
# REPETITIONS = 20

DO_PARALLEL = False  # XXX do not let this = False  # To profile the code, turn down parallel computing
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1) and DO_PARALLEL
N_JOBS = -1 if DO_PARALLEL else 1
if CPU_COUNT > 4:  # We are on a server, let's be nice and not use all cores
    N_JOBS = min(CPU_COUNT, max(int(CPU_COUNT / 3), CPU_COUNT - 8))
N_JOBS = int(getenv('N_JOBS', N_JOBS))

# Parameters for the epsilon-greedy and epsilon-... policies
EPSILON = 0.1
# Temperature for the Softmax
TEMPERATURE = 0.005
# Parameters for the Aggr policy
LEARNING_RATE = 0.01
LEARNING_RATES = [LEARNING_RATE]
DECREASE_RATE = HORIZON / 2.0
DECREASE_RATE = None

# NB_PLAYERS : number of player
NB_PLAYERS = 1    # Less that the number of arms
NB_PLAYERS = 2    # Less that the number of arms
NB_PLAYERS = 3    # Less that the number of arms
# NB_PLAYERS = 6    # Less that the number of arms
# NB_PLAYERS = 9    # Less that the number of arms
# NB_PLAYERS = 12   # Less that the number of arms
# NB_PLAYERS = 17   # Just the number of arms
# NB_PLAYERS = 25   # XXX More than the number of arms !!
# NB_PLAYERS = 30   # XXX More than the number of arms !!

# Collision model
collisionModel = rewardIsSharedUniformly
collisionModel = noCollision
collisionModel = onlyUniqUserGetsReward    # XXX this is the best one

# distances = np.random.random_sample(NB_PLAYERS)
# print("Each player is at the base station with a certain distance (the lower, the more chance it has to be selected)")
# for i in range(NB_PLAYERS):
#     print("  - Player nb {}\tis at distance {} ...".format(i + 1, distances[i]))
# def closerOneGetsReward(*args): return closerUserGetsReward(*args, distances=distances)
# def closerOneGetsReward(*args): return closerUserGetsReward(*args, distances='random')  # Let it compute the random distances, ONCE by thread, and then cache it
# collisionModel = closerOneGetsReward


# Parameters for the arms
VARIANCE = 0.05   # Variance of Gaussian arms


# Should I test the Aggr algorithm here also ?
TEST_AGGR = True
TEST_AGGR = False  # XXX do not let this = False if you want to test my Aggr policy

# Cache rewards
CACHE_REWARDS = False  # XXX to disable manually this feature
CACHE_REWARDS = TEST_AGGR

UPDATE_ALL_CHILDREN = True
UPDATE_ALL_CHILDREN = False  # XXX do not let this = False

# UNBIASED is a flag to know if the rewards are used as biased estimator, ie just r_t, or unbiased estimators, r_t / p_t
UNBIASED = True
UNBIASED = False

# Flag to know if we should update the trusts proba like in Exp4 or like in my initial Aggr proposal
UPDATE_LIKE_EXP4 = True     # trusts^(t+1) = exp(rate_t * estimated rewards upto time t)
UPDATE_LIKE_EXP4 = False    # trusts^(t+1) <-- trusts^t * exp(rate_t * estimate reward at time t)


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
    # --- Collision model
    "collisionModel": collisionModel,
    # --- Other parameters for the Evaluator
    "finalRanksOnAverage": True,  # Use an average instead of the last value for the final ranking of the tested players
    "averageOn": 1e-3,  # Average the final rank on the 1.% last time steps
    # --- Arms
    "environment": [
        # {   # A damn simple problem: 2 arms, one bad, one good
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.9]  # makeMeans(2, 0.1)
        #     # "params": [0.9, 0.9]
        #     # "params": [0.85, 0.9]
        # }
        # {   # A very very easy problem: 3 arms, one bad, one average, one good
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.5, 0.9]  # makeMeans(3, 0.1)
        # }
        {   # A very easy problem (9 arms), but it is used in a lot of articles
            "arm_type": Bernoulli,
            "params": makeMeans(9, 1 / (1. + 9))
        }
        # {   # An easy problem (14 arms)
        #     "arm_type": Bernoulli,
        #     "params": makeMeans(14, 1 / (1. + 14))
        # }
        # {   # An easy problem (19 arms)
        #     "arm_type": Bernoulli,
        #     "params": makeMeans(19, 1 / (1. + 19))
        # }
        # {   # An other problem (17 arms), best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3, 0.6) and very good arms (0.78, 0.85)
        #     "arm_type": Bernoulli,
        #     "params": [0.005, 0.01, 0.015, 0.02, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.78, 0.8, 0.82, 0.83, 0.84, 0.85]
        # }
        # {   # XXX to test with 1 suboptimal arm only
        #     "arm_type": Bernoulli,
        #     "params": makeMeans((NB_PLAYERS + 1), 1 / (1. + (NB_PLAYERS + 1)))
        # }
        # {   # XXX to test with half very bad arms, half perfect arms
        #     "arm_type": Bernoulli,
        #     # "params": shuffled([0, 0, 0, 1, 1, 1, 0, 0, 0])
        #     "params": shuffled([0] * NB_PLAYERS) + ([1] * NB_PLAYERS)
        # }
        # {   # XXX To only test the orthogonalization (collision avoidance) protocol
        #     "arm_type": Bernoulli,
        #     "params": [1] * NB_PLAYERS
        # }
        # {   # An easy problem (50 arms)
        #     "arm_type": Bernoulli,
        #     "params": makeMeans(50, 1 / (1. + 50))
        # }
        # {   # Scenario 1 from [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779]
        #     "arm_type": Bernoulli,
        #     "params": [0.3, 0.4, 0.5, 0.6, 0.7]
        #     # nbPlayers = 2
        # }
        # {   # Variant on scenario 1 from [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779]
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.2, 0.7, 0.8, 0.9]
        #     # nbPlayers = 2
        # }
        # {   # Scenario 2 from [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779]
        #     "arm_type": Bernoulli,
        #     "params": [0.03] * (20 - 13 + 1) + [0.05] * (12 - 4 + 1) + [0.10, 0.12, 0.15]
        #     # nbPlayers = 3
        # }
    ],
    # DONE I tried with other arms distribution: Exponential, it works similarly
    # "environment": [  # Exponential arms
    #     {   # An example problem with  arms
    #         "arm_type": Exponential,
    #         "params": [2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     }
    # ],
    # DONE I tried with other arms distribution: Gaussian, it works similarly
    # "environment": [  # Exponential arms
    #     {   # An example problem with  arms
    #         "arm_type": Gaussian,
    #         "params": [(0.1, VARIANCE), (0.2, VARIANCE), (0.3, VARIANCE), (0.4, VARIANCE), (0.5, VARIANCE), (0.6, VARIANCE), (0.7, VARIANCE), (0.8, VARIANCE), (0.9, VARIANCE)]
    #     }
    # ],
}


nbArms = len(configuration['environment'][0]['params'])
if len(configuration['environment']) > 1:
    print("WARNING do not use this hack if you try to use more than one environment.")
# XXX compute optimal values for d (MEGA's parameter)
D = max(0.01, np.min(np.diff(np.sort(configuration['environment'][0]['params']))) / 2)


configuration.update({
    # --- DONE Defining manually each child
    # "players": [TakeFixedArm(nbArms, nbArms - 1) for _ in range(NB_PLAYERS)]
    # "players": [TakeRandomFixedArm(nbArms) for _ in range(NB_PLAYERS)]

    # --- Defining each player as one child of a multi-player policy

    # --- DONE Using multi-player dummy Centralized policy
    # XXX each player needs to now the number of players
    # "players": CentralizedFixed(NB_PLAYERS, nbArms).children
    # "players": CentralizedCycling(NB_PLAYERS, nbArms).children
    # --- DONE Using a smart Centralized policy, based on choiceMultiple()
    # "players": CentralizedMultiplePlay(NB_PLAYERS, UCB, nbArms, uniformAllocation=False).children
    # "players": CentralizedMultiplePlay(NB_PLAYERS, UCB, nbArms, uniformAllocation=True).children
    # "players": CentralizedMultiplePlay(NB_PLAYERS, Thompson, nbArms, uniformAllocation=False).children
    # "players": CentralizedMultiplePlay(NB_PLAYERS, Thompson, nbArms, uniformAllocation=True).children

    # --- DONE Using a smart Centralized policy, based on choiceIMP() -- It's not better, in fact
    # "players": CentralizedIMP(NB_PLAYERS, UCB, nbArms, uniformAllocation=False).children
    # "players": CentralizedIMP(NB_PLAYERS, UCB, nbArms, uniformAllocation=True).children
    # "players": CentralizedIMP(NB_PLAYERS, Thompson, nbArms, uniformAllocation=False).children
    # "players": CentralizedIMP(NB_PLAYERS, Thompson, nbArms, uniformAllocation=True).children

    # --- DONE Using multi-player Selfish policy
    # "players": Selfish(NB_PLAYERS, Uniform, nbArms).children
    # "players": Selfish(NB_PLAYERS, TakeRandomFixedArm, nbArms).children
    # "players": Selfish(NB_PLAYERS, Exp3Decreasing, nbArms).children
    # "players": Selfish(NB_PLAYERS, Exp3WithHorizon, nbArms, horizon=HORIZON).children
    # "players": Selfish(NB_PLAYERS, UCB, nbArms).children
    # "players": Selfish(NB_PLAYERS, UCBalpha, nbArms, alpha=0.25).children  # This one is efficient!
    # "players": Selfish(NB_PLAYERS, MOSS, nbArms).children
    # "players": Selfish(NB_PLAYERS, klUCB, nbArms).children
    # "players": Selfish(NB_PLAYERS, klUCBPlus, nbArms).children
    # "players": Selfish(NB_PLAYERS, klUCBHPlus, nbArms, horizon=HORIZON).children  # Worse than simple klUCB and klUCBPlus
    # "players": Selfish(NB_PLAYERS, Thompson, nbArms).children
    # "players": Selfish(NB_PLAYERS, SoftmaxDecreasing, nbArms).children
    # XXX this Selfish[AdBandits] and Selfish[BayesUCB] work crazily well... why?
    # "players": Selfish(NB_PLAYERS, BayesUCB, nbArms).children
    # "players": Selfish(NB_PLAYERS, AdBandits, nbArms, alpha=0.5, horizon=HORIZON).children

    # --- DONE Using multi-player Oracle policy
    # XXX they need a perfect knowledge on the arms, OF COURSE this is not physically plausible at all
    # "players": OracleNotFair(NB_PLAYERS, MAB(configuration['environment'][0])).children
    # "players": OracleFair(NB_PLAYERS, MAB(configuration['environment'][0])).children

    # --- DONE Using single-player Musical Chair policy
    # OK Estimate nbPlayers in Time0 initial rounds
    # "players": Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.2, Time1=HORIZON).children
    # "players": Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.1, Time1=HORIZON).children
    # "players": Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.05, Time1=HORIZON).children
    # "players": Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.005, Time1=HORIZON).children

    # --- DONE Using single-player MEGA policy
    # FIXME how to chose the 5 parameters ??
    # "players": Selfish(NB_PLAYERS, MEGA, nbArms, p0=0.6, alpha=0.5, beta=0.8, c=0.1, d=D).children

    # --- DONE Using single-player ALOHA policy
    # FIXME how to chose the 2 parameters p0 and alpha_p0 ?
    # "players": ALOHA(NB_PLAYERS, EpsilonDecreasingMEGA, nbArms, p0=0.6, alpha_p0=0.5, beta=0.8, c=0.1, d=D).children  # Example to prove that Selfish[MEGA] = ALOHA[EpsilonGreedy]
    # "players": ALOHA(NB_PLAYERS, UCB, nbArms, p0=0.6, alpha_p0=0.5, beta=0.8).children  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, MOSS, nbArms, p0=0.6, alpha_p0=0.5, beta=0.8).children  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, klUCBPlus, nbArms, p0=0.6, alpha_p0=0.5, beta=0.8).children  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, Thompson, nbArms, p0=1. / NB_PLAYERS, alpha_p0=0.01, beta=0.2).children  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, Thompson, nbArms, p0=0.6, alpha_p0=0.99, ftnext=tnext_log).children  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, BayesUCB, nbArms, p0=0.6, alpha_p0=0.5, beta=0.8).children  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, SoftmaxDecreasing, nbArms, p0=0.6, alpha_p0=0.5).children  # TODO try this one!

    # --- DONE Using single-player rhoRand policy
    # "players": rhoRand(NB_PLAYERS, UCB, nbArms).children
    # "players": rhoRand(NB_PLAYERS, klUCBPlus, nbArms).children
    # "players": rhoRand(NB_PLAYERS, Thompson, nbArms).children
    "players": rhoRand(NB_PLAYERS, BayesUCB, nbArms).children
    # "players": rhoRand(NB_PLAYERS, AdBandits, nbArms, alpha=0.5, horizon=HORIZON).children

    # --- DONE Using single-player rhoEst policy
    # "players": rhoEst(NB_PLAYERS, UCB, nbArms, HORIZON).children
    # "players": rhoEst(NB_PLAYERS, klUCBPlus, nbArms, HORIZON).children
    # "players": rhoEst(NB_PLAYERS, Thompson, nbArms, HORIZON).children
    # "players": rhoEst(NB_PLAYERS, BayesUCB, nbArms, HORIZON).children

    # --- DONE Using single-player rhoLearn policy, with same MAB learning algorithm for selecting the ranks
    # "players": rhoLearn(NB_PLAYERS, UCB, nbArms, UCB).children
    # "players": rhoLearn(NB_PLAYERS, klUCBPlus, nbArms, klUCBPlus).children
    # "players": rhoLearn(NB_PLAYERS, Thompson, nbArms, Thompson).children
    # "players": rhoLearn(NB_PLAYERS, BayesUCB, nbArms, BayesUCB).children

    # --- DONE Using single-player stupid rhoRandRand policy
    # "players": rhoRandRand(NB_PLAYERS, UCB, nbArms).children
})
# TODO the EvaluatorMultiPlayers should regenerate the list of players in every repetitions, to have at the end results on the average behavior of these randomized multi-players policies


# configuration["successive_players"] = [
#     rhoRand(NB_PLAYERS, UCBalpha, nbArms, alpha=1).children,  # This one is efficient!
#     rhoRand(NB_PLAYERS, UCBalpha, nbArms, alpha=0.25).children,  # This one is efficient!
#     rhoRand(NB_PLAYERS, MOSS, nbArms).children,
#     rhoRand(NB_PLAYERS, klUCB, nbArms).children,
#     rhoRand(NB_PLAYERS, klUCBPlus, nbArms).children,
#     rhoRand(NB_PLAYERS, Thompson, nbArms).children,
#     rhoRand(NB_PLAYERS, SoftmaxDecreasing, nbArms).children,
#     rhoRand(NB_PLAYERS, BayesUCB, nbArms).children,
#     rhoRand(NB_PLAYERS, AdBandits, nbArms, alpha=0.5, horizon=HORIZON).children,
# ]


# configuration["successive_players"] = [
#     rhoEst(NB_PLAYERS, UCBalpha, nbArms, HORIZON, alpha=1).children,
#     rhoRand(NB_PLAYERS, UCBalpha, nbArms, alpha=1).children,
#     rhoEst(NB_PLAYERS, Thompson, nbArms, HORIZON).children,
#     rhoRand(NB_PLAYERS, Thompson, nbArms).children,
#     rhoEst(NB_PLAYERS, klUCB, nbArms, HORIZON).children,
#     rhoRand(NB_PLAYERS, klUCB, nbArms).children,
#     rhoEst(NB_PLAYERS, BayesUCB, nbArms, HORIZON).children,
#     rhoRand(NB_PLAYERS, BayesUCB, nbArms).children,
# ]

# configuration["successive_players"] = [
#     Selfish(NB_PLAYERS, UCBalpha, nbArms, alpha=1).children,  # This one is efficient!
#     Selfish(NB_PLAYERS, UCBalpha, nbArms, alpha=0.25).children,  # This one is efficient!
#     # Selfish(NB_PLAYERS, MOSS, nbArms).children,
#     Selfish(NB_PLAYERS, klUCB, nbArms).children,
#     Selfish(NB_PLAYERS, klUCBPlus, nbArms).children,
#     Selfish(NB_PLAYERS, Thompson, nbArms).children,
#     Selfish(NB_PLAYERS, SoftmaxDecreasing, nbArms).children,
#     Selfish(NB_PLAYERS, BayesUCB, nbArms).children,
#     # Selfish(NB_PLAYERS, AdBandits, nbArms, alpha=0.5, horizon=HORIZON).children,
# ]

# configuration["successive_players"] = [
#     CentralizedMultiplePlay(NB_PLAYERS, UCBalpha, nbArms, alpha=1).children,
#     CentralizedIMP(NB_PLAYERS, UCBalpha, nbArms, alpha=1).children,
#     CentralizedMultiplePlay(NB_PLAYERS, Thompson, nbArms).children,
#     CentralizedIMP(NB_PLAYERS, Thompson, nbArms).children,
#     CentralizedMultiplePlay(NB_PLAYERS, klUCBPlus, nbArms).children,
# ]

configuration["successive_players"] = [
    # --- 1) CentralizedMultiplePlay
    # CentralizedMultiplePlay(NB_PLAYERS, UCBalpha, nbArms, alpha=1).children,
    # CentralizedMultiplePlay(NB_PLAYERS, BayesUCB, nbArms).children,
    # --- 2) Musical Chair
    # Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.1, Time1=HORIZON).children,
    # Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.05, Time1=HORIZON).children,
    # Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.005, Time1=HORIZON).children,
    # Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.001, Time1=HORIZON).children,
    # Selfish(NB_PLAYERS, EmpiricalMeans, nbArms).children,
    # --- 3) EmpiricalMeans
    # # rhoRand(NB_PLAYERS, EmpiricalMeans, nbArms).children,
    # rhoEst(NB_PLAYERS, EmpiricalMeans, nbArms, HORIZON).children,
    # --- 4) UCBalpha
    # # rhoLearn(NB_PLAYERS, UCBalpha, nbArms, Uniform, alpha=1).children,  # OK, == rhoRand
    # rhoLearn(NB_PLAYERS, UCBalpha, nbArms, UCB, alpha=1).children,  # OK, == rhoRand
    # rhoRand(NB_PLAYERS, UCBalpha, nbArms, alpha=1).children,
    # # rhoEst(NB_PLAYERS, UCBalpha, nbArms, HORIZON, alpha=1).children,
    # Selfish(NB_PLAYERS, UCBalpha, nbArms, alpha=1).children,
    # --- 5) klUCBPlus
    # Selfish(NB_PLAYERS, klUCBPlus, nbArms).children,
    # rhoRand(NB_PLAYERS, klUCBPlus, nbArms).children,
    # # rhoEst(NB_PLAYERS, klUCBPlus, nbArms, HORIZON).children,
    # # rhoLearn(NB_PLAYERS, klUCBPlus, nbArms, klUCBPlus).children,
    # rhoLearn(NB_PLAYERS, klUCBPlus, nbArms, UCB).children,
    # # rhoLearn(NB_PLAYERS, klUCBPlus, nbArms, EpsilonDecreasing).children,
    # # rhoLearn(NB_PLAYERS, klUCBPlus, nbArms, SoftmaxDecreasing).children,
    # # rhoEst(NB_PLAYERS, klUCBPlus, nbArms, HORIZON).children,
    # --- 6) Thompson
    # Selfish(NB_PLAYERS, Thompson, nbArms).children,
    # # rhoRand(NB_PLAYERS, Thompson, nbArms).children,
    # rhoEst(NB_PLAYERS, Thompson, nbArms, HORIZON).children,
    # --- 7) BayesUCB
    # Selfish(NB_PLAYERS, BayesUCB, nbArms).children,
    # rhoRand(NB_PLAYERS, BayesUCB, nbArms).children,
    # # rhoEst(NB_PLAYERS, BayesUCB, nbArms, HORIZON).children,
    # # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, SoftmaxDecreasing).children,
    # # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, UCBalpha).children,
    # # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, Thompson).children,
    # # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, klUCBPlus).children,
    # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, BayesUCB).children,
    # --- 8) Aggr
    Selfish(NB_PLAYERS, Aggr, nbArms, unbiased=UNBIASED, update_all_children=UPDATE_ALL_CHILDREN, decreaseRate="auto", update_like_exp4=UPDATE_LIKE_EXP4, children=[UCBalpha, Thompson, klUCBPlus, BayesUCB]).children,
    rhoRand(NB_PLAYERS, Aggr, nbArms, unbiased=UNBIASED, update_all_children=UPDATE_ALL_CHILDREN, decreaseRate="auto", update_like_exp4=UPDATE_LIKE_EXP4, children=[UCBalpha, Thompson, klUCBPlus, BayesUCB]).children,
    # rhoEst(NB_PLAYERS, Aggr, nbArms, HORIZON, unbiased=UNBIASED, update_all_children=UPDATE_ALL_CHILDREN, decreaseRate="auto", update_like_exp4=UPDATE_LIKE_EXP4, children=[Thompson, klUCBPlus, BayesUCB]).children,
]


# from itertools import product  # XXX If needed!

# p0 = 1. / NB_PLAYERS
# p0 = 0.75

# configuration["successive_players"] = [
#     Selfish(NB_PLAYERS, BayesUCB, nbArms).children,  # This one is efficient!
# ] + [
#     ALOHA(NB_PLAYERS, BayesUCB, nbArms, p0=p0, alpha_p0=alpha_p0, beta=beta).children
#     # ALOHA(NB_PLAYERS, BayesUCB, nbArms, p0=p0, alpha_p0=alpha_p0, ftnext=tnext_log).children,
#     for alpha_p0, beta in product([0.05, 0.25, 0.5, 0.75, 0.95], repeat=2)
#     # for alpha_p0, beta in product([0.1, 0.5, 0.9], repeat=2)
# ]


# DONE
print("Loaded experiments configuration from 'configuration.py' :")
print("configuration =", configuration)  # DEBUG
