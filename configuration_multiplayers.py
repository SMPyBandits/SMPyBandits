# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the multi-players case.
"""
from __future__ import print_function, division

__author__ = "Lilian Besson"
__version__ = "0.6"

# Tries to know number of CPU
try:
    from multiprocessing import cpu_count
    CPU_COUNT = cpu_count()
except ImportError:
    CPU_COUNT = 1

from os import getenv
import numpy as np

if __name__ == '__main__':
    print("Warning: this script 'configuration_multiplayers.py' is NOT executable. Use 'main_multiplayers.py' or 'make multiplayers' or 'make moremultiplayers' ...")  # DEBUG
    exit(0)

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


#: HORIZON : number of time steps of the experiments.
#: Warning Should be >= 10000 to be interesting "asymptotically".
HORIZON = 500
HORIZON = 2000
HORIZON = 3000
HORIZON = 5000
HORIZON = 10000
HORIZON = 20000
HORIZON = 30000
# HORIZON = 40000
HORIZON = 100000

#: DELTA_T_SAVE : save only 1 / DELTA_T_SAVE points, to speed up computations, use less RAM, speed up plotting etc.
#: Warning: not perfectly finished right now.
DELTA_T_SAVE = 1 * (HORIZON < 10000) + 50 * (10000 <= HORIZON < 100000) + 100 * (HORIZON >= 100000)
DELTA_T_SAVE = 1  # XXX to disable this optimization

#: REPETITIONS : number of repetitions of the experiments.
#: Warning: Should be >= 10 to be stastically trustworthy.
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
# REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
# REPETITIONS = 1000
# REPETITIONS = 200
# REPETITIONS = 100
# REPETITIONS = 50
# REPETITIONS = 20
# REPETITIONS = 10

#: To profile the code, turn down parallel computing
DO_PARALLEL = False  # XXX do not let this = False  # To profile the code, turn down parallel computing
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1) and DO_PARALLEL

#: Number of jobs to use for the parallel computations. -1 means all the CPU cores, 1 means no parallelization.
N_JOBS = -1 if DO_PARALLEL else 1
if CPU_COUNT > 4:  # We are on a server, let's be nice and not use all cores
    N_JOBS = min(CPU_COUNT, max(int(CPU_COUNT / 3), CPU_COUNT - 8))
N_JOBS = int(getenv('N_JOBS', N_JOBS))

#: Parameters for the epsilon-greedy and epsilon-... policies
EPSILON = 0.1
#: Temperature for the Softmax policies.
TEMPERATURE = 0.005
#: Learning rate for my aggregated bandit (it can be autotuned)
LEARNING_RATE = 0.01
LEARNING_RATES = [LEARNING_RATE]
#: Constant time tau for the decreasing rate for my aggregated bandit.
DECREASE_RATE = HORIZON / 2.0
DECREASE_RATE = None

#: NB_PLAYERS : number of players for the game. Should be >= 2 and <= number of arms.
NB_PLAYERS = 1    # Less that the number of arms
NB_PLAYERS = 2    # Less that the number of arms
NB_PLAYERS = 3    # Less that the number of arms
NB_PLAYERS = 4    # Less that the number of arms
# NB_PLAYERS = 5    # Less that the number of arms
# NB_PLAYERS = 6    # Less that the number of arms
# NB_PLAYERS = 9    # Less that the number of arms
# NB_PLAYERS = 12   # Less that the number of arms
# NB_PLAYERS = 17   # Just the number of arms
# NB_PLAYERS = 25   # XXX More than the number of arms !!
# NB_PLAYERS = 30   # XXX More than the number of arms !!


# #: Different Collision models
# collisionModel = noCollision  #: Like single player.
# collisionModel = rewardIsSharedUniformly  #: Weird collision model.

# # Based on a distance of each user with the base station: the closer one wins if collision
# distances = 'uniform'  # Uniformly spaced objects
# distances = 'random'  # Let it compute the random distances, ONCE by thread, and then cache it? XXX
# distances = np.random.random_sample(NB_PLAYERS)  # Distance between 0 and 1, randomly affected!
# print("Each player is at the base station with a certain distance (the lower, the more chance it has to be selected)")  # DEBUG
# for i in range(NB_PLAYERS):
#     print("  - Player nb #{}\tis at distance {:.3g} to the Base Station ...".format(i + 1, distances[i]))  # DEBUG


# def onlyCloserUserGetsReward(t, arms, players, choices, rewards, pulls, collisions, distances=distances):
#     return closerUserGetsReward(t, arms, players, choices, rewards, pulls, collisions, distances=distances)


# collisionModel = onlyCloserUserGetsReward
# collisionModel.__doc__ = closerUserGetsReward.__doc__

#: The best collision model: none of the colliding users get any reward
collisionModel = onlyUniqUserGetsReward    # XXX this is the best one

# collisionModel = allGetRewardsAndUseCollision  #: DONE this is a bad collision model


# Parameters for the arms
VARIANCE = 0.05   #: Variance of Gaussian arms


#: Should I test the Aggr algorithm here also ?
TEST_AGGR = True
TEST_AGGR = False  # XXX do not let this = False if you want to test my Aggr policy

#: Should we cache rewards? The random rewards will be the same for all the REPETITIONS simulations for each algorithms.
CACHE_REWARDS = False  # XXX to disable manually this feature
CACHE_REWARDS = TEST_AGGR

#: Should the Aggr policy update the trusts in each child or just the one trusted for last decision?
UPDATE_ALL_CHILDREN = True
UPDATE_ALL_CHILDREN = False  # XXX do not let this = False

#: Should the rewards for Aggr policy use as biased estimator, ie just ``r_t``, or unbiased estimators, ``r_t / p_t``
UNBIASED = True
UNBIASED = False

#: Should we update the trusts proba like in Exp4 or like in my initial Aggr proposal
UPDATE_LIKE_EXP4 = True     # trusts^(t+1) = exp(rate_t * estimated rewards upto time t)
UPDATE_LIKE_EXP4 = False    # trusts^(t+1) <-- trusts^t * exp(rate_t * estimate reward at time t)


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
    # --- Collision model
    "collisionModel": collisionModel,
    # --- Other parameters for the Evaluator
    "finalRanksOnAverage": True,  # Use an average instead of the last value for the final ranking of the tested players
    "averageOn": 1e-3,  # Average the final rank on the 1.% last time steps
    # --- Arms
    "environment": [
        # {   # A damn simple problem: 2 arms, one bad, one good
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.9]  # uniformMeans(2, 0.1)
        #     # "params": [0.9, 0.9]
        #     # "params": [0.85, 0.9]
        # }
        # {   # A very very easy problem: 3 arms, one bad, one average, one good
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.5, 0.9]  # uniformMeans(3, 0.1)
        # }
        {   # A very easy problem (9 arms), but it is used in a lot of articles
            "arm_type": Bernoulli,
            "params": uniformMeans(9, 1 / (1. + 9))
        }
        # {   # An easy problem (14 arms)
        #     "arm_type": Bernoulli,
        #     "params": uniformMeans(14, 1 / (1. + 14))
        # }
        # {   # An easy problem (19 arms)
        #     "arm_type": Bernoulli,
        #     "params": uniformMeans(19, 1 / (1. + 19))
        # }
        # {   # An other problem (17 arms), best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3, 0.6) and 6 very good arms (0.78, 0.85)
        #     "arm_type": Bernoulli,
        #     "params": [0.005, 0.01, 0.015, 0.02, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.78, 0.8, 0.82, 0.83, 0.84, 0.85]
        # }
        # {   # XXX to test with 1 suboptimal arm only
        #     "arm_type": Bernoulli,
        #     "params": uniformMeans((NB_PLAYERS + 1), 1 / (1. + (NB_PLAYERS + 1)))
        # }
        # {   # XXX to test with half very bad arms, half perfect arms
        #     "arm_type": Bernoulli,
        #     "params": shuffled([0] * NB_PLAYERS) + ([1] * NB_PLAYERS)
        # }
        # {   # XXX To only test the orthogonalization (collision avoidance) protocol
        #     "arm_type": Bernoulli,
        #     "params": [1] * NB_PLAYERS
        # }
        # {   # An easy problem, but with a LOT of arms! (50 arms)
        #     "arm_type": Bernoulli,
        #     "params": uniformMeans(50, 1 / (1. + 50))
        # }
        # {   # Scenario 1 from [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779]
        #     "arm_type": Bernoulli,
        #     "params": [0.3, 0.4, 0.5, 0.6, 0.7]
        #     # nbPlayers = 2
        # }
        # {   # Variant on scenario 1 from [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779]
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.2, 0.6, 0.7, 0.8, 0.9]
        #     # nbPlayers = 4
        # }
        # {   # Scenario 2 from [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779]
        #     "arm_type": Bernoulli,
        #     "params": [0.03] * (20 - 13 + 1) + [0.05] * (12 - 4 + 1) + [0.10, 0.12, 0.15]
        #     # nbPlayers = 3
        # }
        # {   # A random problem: every repetition use a different mean vectors!
        #     "arm_type": Bernoulli,
        #     "params": {
        #         "function": randomMeans,
        #         "args": {
        #             "nbArms": NB_PLAYERS,
        #             "lower": 0.,
        #             "amplitude": 1.,
        #             "mingap": 1. / (NB_PLAYERS * 2 + 1),
        #         }
        #     }
        # },
    ],
    # DONE I tried with other arms distribution: Exponential, it works similarly
    # "environment": [  # Exponential arms
    #     {   # An example problem with  arms
    #         "arm_type": Exponential,
    #         "params": [2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     }
    # ],
    # # DONE I tried with other arms distribution: Gaussian, it works similarly
    # "environment": [  # Gaussian arms
    #     {   # An example problem with  arms
    #         "arm_type": Gaussian,
    #         "params": [(0.1, VARIANCE), (0.2, VARIANCE), (0.8, VARIANCE), (0.9, VARIANCE)]
    #         # "params": [(0.1, VARIANCE), (0.2, VARIANCE), (0.3, VARIANCE), (0.4, VARIANCE), (0.5, VARIANCE), (0.6, VARIANCE), (0.7, VARIANCE), (0.8, VARIANCE), (0.9, VARIANCE)]
    #     }
    # ],
}


try:
    #: Number of arms *in the first environment*
    nbArms = int(configuration['environment'][0]['params']['args']['nbArms'])
except (TypeError, KeyError):
    nbArms = len(configuration['environment'][0]['params'])

if len(configuration['environment']) > 1:
    print("WARNING do not use this hack if you try to use more than one environment.")
# XXX compute optimal values for d (MEGA's parameter)
# D = max(0.01, np.min(np.diff(np.sort(configuration['environment'][0]['params']))) / 2)


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
    # "players": Selfish(NB_PLAYERS, BayesUCB, nbArms).children  # FIXME I am working on this line right now!
    # "players": Selfish(int(NB_PLAYERS / 3), BayesUCB, nbArms).children \
    #          + Selfish(int(NB_PLAYERS / 3), Thompson, nbArms).children \
    #          + Selfish(int(NB_PLAYERS / 3), klUCBPlus, nbArms).children
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
    # "players": rhoRand(NB_PLAYERS, BayesUCB, nbArms).children
    # "players": rhoRand(int(NB_PLAYERS / 3), BayesUCB, nbArms, maxRank=NB_PLAYERS).children \
    #          + rhoRand(int(NB_PLAYERS / 3), Thompson, nbArms, maxRank=NB_PLAYERS).children \
    #          + rhoRand(int(NB_PLAYERS / 3), klUCBPlus, nbArms, maxRank=NB_PLAYERS).children
    # "players": rhoRand(NB_PLAYERS, AdBandits, nbArms, alpha=0.5, horizon=HORIZON).children

    # --- DONE Using single-player rhoEst policy
    # "players": rhoEst(NB_PLAYERS, UCB, nbArms, HORIZON).children
    # "players": rhoEst(NB_PLAYERS, klUCBPlus, nbArms, HORIZON).children
    # "players": rhoEst(NB_PLAYERS, Thompson, nbArms, HORIZON).children
    # "players": rhoEst(NB_PLAYERS, BayesUCB, nbArms, HORIZON).children

    # --- DONE Using single-player rhoLearn policy, with same MAB learning algorithm for selecting the ranks
    "players": rhoLearn(NB_PLAYERS, UCB, nbArms, UCB).children
    # "players": rhoLearn(NB_PLAYERS, klUCBPlus, nbArms, klUCBPlus).children
    # "players": rhoLearn(NB_PLAYERS, Thompson, nbArms, Thompson).children
    # "players": rhoLearn(NB_PLAYERS, BayesUCB, nbArms, BayesUCB, change_rank_each_step=True).children
    # "players": rhoLearn(NB_PLAYERS, BayesUCB, nbArms, BayesUCB, change_rank_each_step=False).children

    # --- DONE Using single-player stupid rhoRandRand policy
    # "players": rhoRandRand(NB_PLAYERS, UCB, nbArms).children

    # --- DONE Using single-player rhoRandSticky policy
    # "players": rhoRandSticky(NB_PLAYERS, UCB, nbArms, stickyTime=10).children
    # "players": rhoRandSticky(NB_PLAYERS, klUCBPlus, nbArms, stickyTime=10).children
    # "players": rhoRandSticky(NB_PLAYERS, Thompson, nbArms, stickyTime=10).children
    # "players": rhoRandSticky(NB_PLAYERS, BayesUCB, nbArms, stickyTime=10).children
})
# TODO the EvaluatorMultiPlayers should regenerate the list of players in every repetitions, to have at the end results on the average behavior of these randomized multi-players policies


# XXX Comparing different rhoRand approaches
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


# XXX Comparing different ALOHA approaches
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

# # XXX Comparing different centralized approaches
# configuration["successive_players"] = [
#     CentralizedMultiplePlay(NB_PLAYERS, UCBalpha, nbArms).children,
#     CentralizedIMP(NB_PLAYERS, UCBalpha, nbArms).children,
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
    # rhoEst(NB_PLAYERS, klUCBPlus, nbArms, HORIZON).children,
    # # rhoLearn(NB_PLAYERS, klUCBPlus, nbArms, klUCBPlus).children,
    # rhoLearn(NB_PLAYERS, klUCBPlus, nbArms, UCB).children,
    # # rhoLearn(NB_PLAYERS, klUCBPlus, nbArms, EpsilonDecreasing).children,
    # # rhoLearn(NB_PLAYERS, klUCBPlus, nbArms, SoftmaxDecreasing).children,
    # # rhoEst(NB_PLAYERS, klUCBPlus, nbArms, HORIZON).children,

    # --- 6) Thompson
    # Selfish(NB_PLAYERS, Thompson, nbArms).children,
    # rhoRand(NB_PLAYERS, Thompson, nbArms).children,
    # # rhoEst(NB_PLAYERS, Thompson, nbArms, HORIZON).children,

    # # --- 7) rhoLearn with BayesUCB
    # Selfish(NB_PLAYERS, BayesUCB, nbArms).children,
    # rhoRand(NB_PLAYERS, BayesUCB, nbArms).children,
    # # rhoEst(NB_PLAYERS, BayesUCB, nbArms, HORIZON).children,
    # # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, SoftmaxDecreasing).children,
    # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, UCBalpha).children,
    # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, Thompson).children,
    # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, klUCBPlus).children,
    # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, BayesUCB).children,

    # --- 8) Aggr
    # Selfish(NB_PLAYERS, Aggr, nbArms, unbiased=UNBIASED, update_all_children=UPDATE_ALL_CHILDREN, decreaseRate="auto", update_like_exp4=UPDATE_LIKE_EXP4, children=[UCBalpha, Thompson, klUCBPlus, BayesUCB]).children,
    # rhoRand(NB_PLAYERS, Aggr, nbArms, unbiased=UNBIASED, update_all_children=UPDATE_ALL_CHILDREN, decreaseRate="auto", update_like_exp4=UPDATE_LIKE_EXP4, children=[UCBalpha, Thompson, klUCBPlus, BayesUCB]).children,
    # # rhoEst(NB_PLAYERS, Aggr, nbArms, HORIZON, unbiased=UNBIASED, update_all_children=UPDATE_ALL_CHILDREN, decreaseRate="auto", update_like_exp4=UPDATE_LIKE_EXP4, children=[Thompson, klUCBPlus, BayesUCB]).children,

    # # --- 9) Comparing Selfish, rhoRand (and variants) with different learning algorithms
    # Selfish(NB_PLAYERS, BayesUCB, nbArms).children,
    # rhoRand(NB_PLAYERS, BayesUCB, nbArms).children,
    # # rhoRandRotating(NB_PLAYERS, BayesUCB, nbArms).children,
    # # rhoRandALOHA(NB_PLAYERS, BayesUCB, nbArms).children,
    # Selfish(NB_PLAYERS, klUCBPlus, nbArms).children,
    # rhoRand(NB_PLAYERS, klUCBPlus, nbArms).children,
    # # rhoRandRotating(NB_PLAYERS, klUCBPlus, nbArms).children,
    # # rhoRandALOHA(NB_PLAYERS, klUCBPlus, nbArms).children,
    # Selfish(NB_PLAYERS, Thompson, nbArms).children,
    # rhoRand(NB_PLAYERS, Thompson, nbArms).children,
    # # rhoRandRotating(NB_PLAYERS, Thompson, nbArms).children,
    # # rhoRandALOHA(NB_PLAYERS, Thompson, nbArms).children,

    # --- 10) Mixing rhoRand or Selfish with different learning algorithms
    # rhoRand(int(NB_PLAYERS / 3), BayesUCB, nbArms, maxRank=NB_PLAYERS).children \
    # + rhoRand(int(NB_PLAYERS / 3), klUCBPlus, nbArms, maxRank=NB_PLAYERS).children \
    # + rhoRand(int(NB_PLAYERS / 3), Thompson, nbArms, maxRank=NB_PLAYERS).children,
    # Selfish(int(NB_PLAYERS / 3), BayesUCB, nbArms).children \
    # + Selfish(int(NB_PLAYERS / 3), klUCBPlus, nbArms).children \
    # + Selfish(int(NB_PLAYERS / 3), Thompson, nbArms).children,

    # --- 11) Comparing different "robust" ThompsonSampling algorithms
    # Selfish(NB_PLAYERS, ThompsonRobust, nbArms, averageOn=1).children,
    # rhoRand(NB_PLAYERS, ThompsonRobust, nbArms, averageOn=1).children,
    # Selfish(NB_PLAYERS, ThompsonRobust, nbArms, averageOn=2).children,
    # rhoRand(NB_PLAYERS, ThompsonRobust, nbArms, averageOn=2).children,
    # Selfish(NB_PLAYERS, ThompsonRobust, nbArms, averageOn=5).children,
    # rhoRand(NB_PLAYERS, ThompsonRobust, nbArms, averageOn=5).children,
    # Selfish(NB_PLAYERS, ThompsonRobust, nbArms, averageOn=10).children,
    # rhoRand(NB_PLAYERS, ThompsonRobust, nbArms, averageOn=10).children,

    # --- 12) Comparing different rhoRandSticky algorithms
    # rhoRandSticky(NB_PLAYERS, BayesUCB, nbArms, stickyTime=1).children,
    # rhoRandSticky(NB_PLAYERS, BayesUCB, nbArms, stickyTime=2).children,
    # rhoRandSticky(NB_PLAYERS, BayesUCB, nbArms, stickyTime=5).children,
    # rhoRandSticky(NB_PLAYERS, BayesUCB, nbArms, stickyTime=10).children,
    # rhoRandSticky(NB_PLAYERS, BayesUCB, nbArms, stickyTime=50).children,
    # rhoRandSticky(NB_PLAYERS, BayesUCB, nbArms, stickyTime=100).children,
    # rhoRandSticky(NB_PLAYERS, BayesUCB, nbArms, stickyTime=200).children,
    # rhoRandSticky(NB_PLAYERS, BayesUCB, nbArms, stickyTime=np.inf).children,  # should be = classic rhoRand

    # # --- 13) Comparing Selfish, and rhoRand with or without initial orthogonal ranks
    # Selfish(NB_PLAYERS, BayesUCB, nbArms).children,
    # rhoRand(NB_PLAYERS, BayesUCB, nbArms).children,
    # rhoCentralized(NB_PLAYERS, BayesUCB, nbArms).children,
    # Selfish(NB_PLAYERS, klUCBPlus, nbArms).children,
    # rhoRand(NB_PLAYERS, klUCBPlus, nbArms).children,
    # rhoCentralized(NB_PLAYERS, klUCBPlus, nbArms).children,
    # Selfish(NB_PLAYERS, Thompson, nbArms).children,
    # rhoRand(NB_PLAYERS, Thompson, nbArms).children,
    # rhoCentralized(NB_PLAYERS, Thompson, nbArms).children,

    # # --- 14) Comparing rhoRand or Selfish for ApproximatedFHGittins, different alpha. The smaller alpha, the better
    # CentralizedMultiplePlay(NB_PLAYERS, BayesUCB, nbArms).children,
    # CentralizedIMP(NB_PLAYERS, BayesUCB, nbArms).children,
    # Selfish(NB_PLAYERS, BayesUCB, nbArms).children,
    # # Selfish(NB_PLAYERS, ApproximatedFHGittins, nbArms, horizon=1.1 * HORIZON, alpha=2).children,
    # Selfish(NB_PLAYERS, ApproximatedFHGittins, nbArms, horizon=1.1 * HORIZON, alpha=1).children,
    # Selfish(NB_PLAYERS, ApproximatedFHGittins, nbArms, horizon=1.1 * HORIZON, alpha=0.5).children,
    # Selfish(NB_PLAYERS, ApproximatedFHGittins, nbArms, horizon=1.1 * HORIZON, alpha=0.25).children,
    # # Selfish(NB_PLAYERS, ApproximatedFHGittins, nbArms, horizon=1.1 * HORIZON, alpha=0.05).children,
    # rhoRand(NB_PLAYERS, BayesUCB, nbArms).children,
    # # rhoRand(NB_PLAYERS, ApproximatedFHGittins, nbArms, horizon=1.1 * HORIZON, alpha=2).children,
    # rhoRand(NB_PLAYERS, ApproximatedFHGittins, nbArms, horizon=1.1 * HORIZON, alpha=1).children,
    # rhoRand(NB_PLAYERS, ApproximatedFHGittins, nbArms, horizon=1.1 * HORIZON, alpha=0.5).children,
    # rhoRand(NB_PLAYERS, ApproximatedFHGittins, nbArms, horizon=1.1 * HORIZON, alpha=0.25).children,
    # # rhoRand(NB_PLAYERS, ApproximatedFHGittins, nbArms, horizon=1.1 * HORIZON, alpha=0.05).children,

    # # --- 15) Comparing Selfish, rhoRand (and variants) with different learning algorithms
    # Selfish(NB_PLAYERS, SoftMix, nbArms).children,
    # rhoRand(NB_PLAYERS, SoftMix, nbArms).children,
    # # Selfish(NB_PLAYERS, SoftmaxDecreasing, nbArms).children,
    # # rhoRand(NB_PLAYERS, SoftmaxDecreasing, nbArms).children,
    # # Selfish(NB_PLAYERS, Exp3, nbArms).children,
    # # rhoRand(NB_PLAYERS, Exp3, nbArms).children,
    # # Selfish(NB_PLAYERS, Exp3WithHorizon, nbArms, horizon=HORIZON).children,
    # # rhoRand(NB_PLAYERS, Exp3WithHorizon, nbArms, horizon=HORIZON).children,
    # Selfish(NB_PLAYERS, Exp3SoftMix, nbArms).children,
    # rhoRand(NB_PLAYERS, Exp3SoftMix, nbArms).children,
    # # XXX against stochastic algorithms
    # Selfish(NB_PLAYERS, BayesUCB, nbArms).children,
    # rhoRand(NB_PLAYERS, BayesUCB, nbArms).children,
    # Selfish(NB_PLAYERS, klUCBPlus, nbArms).children,
    # rhoRand(NB_PLAYERS, klUCBPlus, nbArms).children,
    # Selfish(NB_PLAYERS, Thompson, nbArms).children,
    # rhoRand(NB_PLAYERS, Thompson, nbArms).children,

    # # --- 16) Comparing rhoLearn and rhoLearnEst (doesn't know M)
    # Selfish(NB_PLAYERS, BayesUCB, nbArms).children,
    # rhoRand(NB_PLAYERS, BayesUCB, nbArms).children,
    # rhoLearn(NB_PLAYERS, BayesUCB, nbArms).children,  # use Uniform, so = rhoRand
    # rhoLearnEst(NB_PLAYERS, BayesUCB, nbArms).children,  # use Uniform, so ~= bad rhoRand
    # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, BayesUCB).children,
    # rhoLearnEst(NB_PLAYERS, BayesUCB, nbArms, BayesUCB).children,  # should be bad!
    # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, klUCBPlus).children,
    # rhoLearnEst(NB_PLAYERS, BayesUCB, nbArms, klUCBPlus).children,  # should be bad!
    # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, Thompson).children,
    # rhoLearnEst(NB_PLAYERS, BayesUCB, nbArms, Thompson).children,  # should be bad!

    # # --- 17) DONE Comparing rhoRand, rhoLearn[BayesUCB], rhoLearn[klUCBPlus] and rhoLearn[Thompson], against rhoLearnExp3, all with BayesUCB for arm selection
    # CentralizedMultiplePlay(NB_PLAYERS, BayesUCB, nbArms).children,
    # Selfish(NB_PLAYERS, BayesUCB, nbArms).children,
    # # Selfish(NB_PLAYERS, Exp3Decreasing, nbArms).children,
    # # Selfish(NB_PLAYERS, Exp3SoftMix, nbArms).children,
    # rhoRand(NB_PLAYERS, BayesUCB, nbArms).children,
    # # rhoLearn(NB_PLAYERS, BayesUCB, nbArms).children,  # use Uniform, so = rhoRand
    # # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, BayesUCB).children,
    # # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, klUCB).children,
    # # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, Thompson).children,
    # # rhoLearnExp3(NB_PLAYERS, BayesUCB, nbArms, feedback_function=binary_feedback, rankSelectionAlgo=Exp3SoftMix).children,
    # # rhoLearnExp3(NB_PLAYERS, BayesUCB, nbArms, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3SoftMix).children,
    # # rhoLearnExp3(NB_PLAYERS, BayesUCB, nbArms, feedback_function=binary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # # rhoLearnExp3(NB_PLAYERS, BayesUCB, nbArms, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # # # rhoLearnExp3(NB_PLAYERS, BayesUCB, nbArms, feedback_function=binary_feedback, rankSelectionAlgo=lambda nbArms: Exp3WithHorizon(nbArms, HORIZON)).children,
    # # # rhoLearnExp3(NB_PLAYERS, BayesUCB, nbArms, feedback_function=ternary_feedback, rankSelectionAlgo=lambda nbArms: Exp3WithHorizon(nbArms, HORIZON)).children,

    # # --- 18) TODO Comparing rhoRand, rhoLearn[BayesUCB], rhoLearn[klUCBPlus] and rhoLearn[Thompson], against rhoLearnExp3, all with klUCB for arm selection
    # CentralizedMultiplePlay(NB_PLAYERS, klUCB, nbArms).children,
    # Selfish(NB_PLAYERS, klUCB, nbArms).children,
    # Selfish(NB_PLAYERS, Exp3Decreasing, nbArms).children,
    # Selfish(NB_PLAYERS, Exp3SoftMix, nbArms).children,
    # # rhoRand(NB_PLAYERS, klUCB, nbArms).children,
    # rhoLearn(NB_PLAYERS, klUCB, nbArms).children,  # use Uniform, so = rhoRand
    # rhoLearn(NB_PLAYERS, klUCB, nbArms, BayesUCB).children,
    # rhoLearn(NB_PLAYERS, klUCB, nbArms, klUCB).children,
    # rhoLearn(NB_PLAYERS, klUCB, nbArms, Thompson).children,
    # rhoLearnExp3(NB_PLAYERS, klUCB, nbArms, feedback_function=binary_feedback, rankSelectionAlgo=Exp3SoftMix).children,
    # rhoLearnExp3(NB_PLAYERS, klUCB, nbArms, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3SoftMix).children,
    # rhoLearnExp3(NB_PLAYERS, klUCB, nbArms, feedback_function=binary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # rhoLearnExp3(NB_PLAYERS, klUCB, nbArms, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3Decreasing).children,

    # --- 19) DONE Comparing Selfish[UCB], rhoRand[UCB], rhoLearn[UCB], rhoLearnExp3[UCB] against SmartMusicalChair[UCB]
    SmartMusicalChair(NB_PLAYERS, UCB, nbArms, withChair=False).children,
    SmartMusicalChair(NB_PLAYERS, UCB, nbArms, withChair=True).children,
    CentralizedMultiplePlay(NB_PLAYERS, UCB, nbArms).children,
    Selfish(NB_PLAYERS, UCB, nbArms).children,
    rhoRand(NB_PLAYERS, UCB, nbArms).children,
    rhoLearn(NB_PLAYERS, UCB, nbArms, UCB).children,
    # rhoLearn(NB_PLAYERS, UCB, nbArms, klUCB).children,
    # rhoLearn(NB_PLAYERS, UCB, nbArms, Thompson).children,
    rhoLearnExp3(NB_PLAYERS, UCB, nbArms, feedback_function=binary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    rhoLearnExp3(NB_PLAYERS, UCB, nbArms, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3Decreasing).children,

    # # --- 20) TODO Comparing Selfish[BayesUCB], rhoRand[BayesUCB], rhoLearn[BayesUCB], rhoLearnExp3[BayesUCB] against SmartMusicalChair[BayesUCB]
    # # FIXME it is *failing* with SmartMusicalChair[BayesUCB]
    # SmartMusicalChair(NB_PLAYERS, BayesUCB, nbArms, withChair=False).children,
    # SmartMusicalChair(NB_PLAYERS, BayesUCB, nbArms, withChair=True).children,
    # CentralizedMultiplePlay(NB_PLAYERS, BayesUCB, nbArms).children,
    # Selfish(NB_PLAYERS, BayesUCB, nbArms).children,
    # rhoRand(NB_PLAYERS, BayesUCB, nbArms).children,
    # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, BayesUCB).children,
    # # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, klUCB).children,
    # # rhoLearn(NB_PLAYERS, BayesUCB, nbArms, Thompson).children,
    # rhoLearnExp3(NB_PLAYERS, BayesUCB, nbArms, feedback_function=binary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # rhoLearnExp3(NB_PLAYERS, BayesUCB, nbArms, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3Decreasing).children,

    # # --- 21) FIXME Comparing Selfish[klUCB], rhoRand[klUCB], rhoLearn[klUCB], rhoLearnExp3[klUCB] against SmartMusicalChair[klUCB]
    # SmartMusicalChair(NB_PLAYERS, klUCB, nbArms, withChair=False).children,
    # SmartMusicalChair(NB_PLAYERS, klUCB, nbArms, withChair=True).children,
    # CentralizedMultiplePlay(NB_PLAYERS, klUCB, nbArms).children,
    # Selfish(NB_PLAYERS, klUCB, nbArms).children,
    # rhoRand(NB_PLAYERS, klUCB, nbArms).children,
    # rhoLearn(NB_PLAYERS, klUCB, nbArms, klUCB).children,
    # # rhoLearn(NB_PLAYERS, klUCB, nbArms, klUCB).children,
    # # rhoLearn(NB_PLAYERS, klUCB, nbArms, Thompson).children,
    # rhoLearnExp3(NB_PLAYERS, klUCB, nbArms, feedback_function=binary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # rhoLearnExp3(NB_PLAYERS, klUCB, nbArms, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
]


# DONE
print("Loaded experiments configuration from 'configuration.py' :")
print("configuration =", configuration)  # DEBUG
