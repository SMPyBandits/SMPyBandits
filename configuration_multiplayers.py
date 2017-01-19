# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the multi-players case.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.3"

from multiprocessing import cpu_count
CPU_COUNT = cpu_count()
# Import arms
from Arms.Bernoulli import Bernoulli
from Arms.Exponential import Exponential
from Arms.Gaussian import Gaussian
from Arms.Poisson import Poisson

# Import contained classes
from Environment.MAB import MAB
# Import algorithms, both single-player and multi-player
from Policies import *
from PoliciesMultiPlayers import *
from PoliciesMultiPlayers.ALOHA import tnext_beta, tnext_log  # XXX do better for these imports
# Collision Models
from Environment.CollisionModels import *


# HORIZON : number of time steps of the experiments
# XXX Should be >= 10000 to be interesting "asymptotically"
HORIZON = 50
HORIZON = 500
HORIZON = 2000
HORIZON = 3000
HORIZON = 5000
HORIZON = 10000
HORIZON = 20000
# HORIZON = 40000
# HORIZON = 100000

# DELTA_T_SAVE : save only 1 / DELTA_T_SAVE points, to speed up computations, use less RAM, speed up plotting etc.
DELTA_T_SAVE = 50 if HORIZON > 10000 else 1
DELTA_T_SAVE = 1  # XXX to disable this optimisation

# REPETITIONS : number of repetitions of the experiments
# XXX Should be >= 10 to be statistically trustworthy
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 20
REPETITIONS = 1000
REPETITIONS = 100
# REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
# REPETITIONS = 1  # XXX To profile the code, turn down parallel computing

DO_PARALLEL = False  # XXX do not let this = False  # To profile the code, turn down parallel computing
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1) and DO_PARALLEL
N_JOBS = -1 if DO_PARALLEL else 1
if CPU_COUNT > 4:  # We are on a server, let's be nice and not use all cores
    N_JOBS = max(int(CPU_COUNT / 2), CPU_COUNT - 4)

# Parameters for the epsilon-greedy and epsilon-... policies
EPSILON = 0.1
# Temperature for the Softmax
TEMPERATURE = 0.005
# Parameters for the Aggr policy
LEARNING_RATE = 0.01
LEARNING_RATES = [LEARNING_RATE]
DECREASE_RATE = HORIZON / 2.0
DECREASE_RATE = None

# NB_PLAYERS : number of player, for policies who need it ?
NB_PLAYERS = 2    # Less that the number of arms
# NB_PLAYERS = 6    # Less that the number of arms
# NB_PLAYERS = 12   # Less that the number of arms
# NB_PLAYERS = 17   # Just the number of arms
# NB_PLAYERS = 25   # XXX More than the number of arms !!

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
    "verbosity": 6,  # Max joblib verbosity
    # --- Collision model
    "collisionModel": collisionModel,
    # --- Other parameters for the Evaluator
    "finalRanksOnAverage": True,  # Use an average instead of the last value for the final ranking of the tested players
    "averageOn": 1e-3,  # Average the final rank on the 1.0% last time steps
    # --- Arms
    "environment": [
        # {   # A damn simple problem: 2 arms, one bad, one good
        #     "arm_type": Bernoulli,
        #     "params": [0.1, 0.9]
        #     # "params": [0.9, 0.9]
        #     # "params": [0.85, 0.9]
        # }
        {   # A very very easy problem: 3 arms, one bad, one average, one good
            "arm_type": Bernoulli,
            "params": [0.1, 0.5, 0.9]
        }
        # {   # A very easy problem (9 arms), but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "params": [t / 10.0 for t in range(1, 10)]
        # }
        # {   # An easy problem (14 arms)
        #     "arm_type": Bernoulli,
        #     "params": [round(t / 15.0, 2) for t in range(1, 15)]
        # }
        # {   # An easy problem (19 arms)
        #     "arm_type": Bernoulli,
        #     "params": [t / 20.0 for t in range(1, 20)]
        # }
        # {   # An other problem (17 arms), best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3, 0.6) and very good arms (0.78, 0.85)
        #     "arm_type": Bernoulli,
        #     "params": [0.005, 0.01, 0.015, 0.02, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.78, 0.8, 0.82, 0.83, 0.84, 0.85]
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
    raise ValueError("WARNING do not use this hack if you try to use more than one environment.")
# XXX compute optimal values for d (MEGA's parameter)
D = max(0.01, np.min(np.diff(np.sort(configuration['environment'][0]['params']))) / 2)

configuration.update({
    # --- DONE Defining manually each child
    # "players": [TakeFixedArm(nbArms, nbArms - 1) for _ in range(NB_PLAYERS)]
    # "players": [TakeRandomFixedArm(nbArms) for _ in range(NB_PLAYERS)]

    # --- Defining each player as one child of a multi-player policy

    # --- DONE Using multi-player Selfish policy
    # "players": Selfish(NB_PLAYERS, Uniform, nbArms).childs
    # "players": Selfish(NB_PLAYERS, TakeRandomFixedArm, nbArms).childs
    # "players": Selfish(NB_PLAYERS, UCB, nbArms).childs
    # "players": Selfish(NB_PLAYERS, UCBalpha, nbArms, alpha=1./4).childs  # This one is efficient!
    # "players": Selfish(NB_PLAYERS, MOSS, nbArms).childs
    # "players": Selfish(NB_PLAYERS, klUCB, nbArms).childs
    # "players": Selfish(NB_PLAYERS, klUCBPlus, nbArms).childs
    # "players": Selfish(NB_PLAYERS, klUCBHPlus, nbArms, horizon=HORIZON).childs  # Worse than simple klUCB and klUCBPlus
    # "players": Selfish(NB_PLAYERS, BayesUCB, nbArms).childs
    # "players": Selfish(NB_PLAYERS, Thompson, nbArms).childs
    # "players": Selfish(NB_PLAYERS, SoftmaxDecreasing, nbArms).childs
    # "players": Selfish(NB_PLAYERS, AdBandits, nbArms, alpha=0.5, horizon=HORIZON).childs

    # --- DONE Using multi-player dummy Centralized policy
    # XXX each player needs to now the number of players
    # "players": CentralizedFixed(NB_PLAYERS, nbArms).childs
    # "players": CentralizedCycling(NB_PLAYERS, nbArms).childs
    # --- DONE Using a smart Centralized policy, based on choiceMultiple()
    # "players": CentralizedMultiplePlay(NB_PLAYERS, UCB, nbArms).childs
    # "players": CentralizedMultiplePlay(NB_PLAYERS, Thompson, nbArms).childs  # FIXME try it !

    # --- DONE Using multi-player Oracle policy
    # XXX they need a perfect knowledge on the arms, OF COURSE this is not physically plausible at all
    # "players": OracleNotFair(NB_PLAYERS, MAB(configuration['environment'][0])).childs
    # "players": OracleFair(NB_PLAYERS, MAB(configuration['environment'][0])).childs

    # --- DONE Using single-player Musical Chair policy
    # OK Estimate nbPlayers in Time0 initial rounds
    # "players": Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.2, Time1=HORIZON).childs
    # "players": Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.1, Time1=HORIZON).childs
    # "players": Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.05, Time1=HORIZON).childs
    "players": Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.005, Time1=HORIZON).childs

    # --- DONE Using single-player MEGA policy
    # FIXME how to chose the 5 parameters ??
    # "players": Selfish(NB_PLAYERS, MEGA, nbArms, p0=0.6, alpha=0.5, beta=0.8, c=0.1, d=D).childs

    # --- FIXME Using single-player ALOHA policy
    # FIXME how to chose the 2 parameters p0 and alpha_p0 ?
    # "players": ALOHA(NB_PLAYERS, EpsilonDecreasingMEGA, nbArms, p0=0.6, alpha_p0=0.5, beta=0.8, c=0.1, d=D).childs  # Example to prove that Selfish[MEGA] = ALOHA[EpsilonGreedy]
    # "players": ALOHA(NB_PLAYERS, UCB, nbArms, p0=0.6, alpha_p0=0.5, beta=0.8).childs  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, MOSS, nbArms, p0=0.6, alpha_p0=0.5, beta=0.8).childs  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, klUCBPlus, nbArms, p0=0.6, alpha_p0=0.5, beta=0.8).childs  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, Thompson, nbArms, p0=1. / NB_PLAYERS, alpha_p0=1, beta=0.5).childs  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, Thompson, nbArms, p0=0.6, alpha_p0=0.99, ftnext=tnext_log).childs  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, BayesUCB, nbArms, p0=0.6, alpha_p0=0.5, beta=0.8).childs  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, SoftmaxDecreasing, nbArms, p0=0.6, alpha_p0=0.5).childs  # TODO try this one!

    # --- DONE Using single-player rhoRand policy
    # "players": rhoRand(NB_PLAYERS, UCB, nbArms).childs
    # "players": rhoRand(NB_PLAYERS, MOSS, nbArms).childs
    # "players": rhoRand(NB_PLAYERS, klUCBPlus, nbArms).childs
    # "players": rhoRand(NB_PLAYERS, Thompson, nbArms).childs
    # "players": rhoRand(NB_PLAYERS, BayesUCB, nbArms).childs
    # "players": rhoRand(NB_PLAYERS, SoftmaxDecreasing, nbArms).childs
})
# TODO the EvaluatorMultiPlayers should regenerate the list of players in every repetitions, to have at the end results on the average behavior of these randomized multi-players policies

print("Loaded experiments configuration from 'configuration.py' :")
print("configuration =", configuration)  # DEBUG
