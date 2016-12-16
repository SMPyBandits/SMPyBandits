# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the multi-players case.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.2"

# Import arms
from Arms.Bernoulli import Bernoulli
# from Arms.Exponential import Exponential
# from Arms.Gaussian import Gaussian
# from Arms.Poisson import Poisson

# Import contained classes
from Environment.MAB import MAB
# Import algorithms, both single-player and multi-player
from Policies import *
from PoliciesMultiPlayers import *
# Collision Models
from Environment.CollisionModels import *


# HORIZON : number of time steps of the experiments
# XXX Should be >= 10000 to be interesting "asymptotically"
HORIZON = 500
HORIZON = 2000
HORIZON = 3000
HORIZON = 5000
HORIZON = 10000
HORIZON = 20000
# HORIZON = 100000

# REPETITIONS : number of repetitions of the experiments
# XXX Should be >= 10 to be statistically trustworthy
REPETITIONS = 1  # XXX To profile the code, turn down parallel computing
REPETITIONS = 20
# REPETITIONS = 100
# REPETITIONS = 2000
# REPETITIONS = 4  # Nb of cores, to have exactly one repetition process by cores
# REPETITIONS = 1  # XXX To profile the code, turn down parallel computing

DO_PARALLEL = False  # XXX do not let this = False  # To profile the code, turn down parallel computing
DO_PARALLEL = True
DO_PARALLEL = (REPETITIONS > 1) and DO_PARALLEL
N_JOBS = -1 if DO_PARALLEL else 1

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
# NB_PLAYERS = 13   # Less that the number of arms
# NB_PLAYERS = 17   # Just the number of arms
# NB_PLAYERS = 25   # More than the number of arms !!

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
configuration.update({
    # --- DONE Defining manually each child
    # "players": [TakeFixedArm(nbArms, nbArms - 1) for _ in range(NB_PLAYERS)]
    # "players": [TakeRandomFixedArm(nbArms) for _ in range(NB_PLAYERS)]

    # --- Defining each player as one child of a multi-player policy

    # --- DONE Using multi-player Selfish policy
    # "players": Selfish(NB_PLAYERS, Uniform, nbArms).childs
    # "players": Selfish(NB_PLAYERS, TakeRandomFixedArm, nbArms).childs
    # "players": Selfish(NB_PLAYERS, UCB, nbArms).childs
    # "players": Selfish(NB_PLAYERS, UCBalpha, nbArms, alpha=1./2).childs
    # "players": Selfish(NB_PLAYERS, UCBalpha, nbArms, alpha=1./4).childs  # This one is efficient!
    # "players": Selfish(NB_PLAYERS, UCBalpha, nbArms, alpha=1./8).childs
    # "players": Selfish(NB_PLAYERS, MOSS, nbArms).childs
    # "players": Selfish(NB_PLAYERS, klUCB, nbArms).childs  # XXX doesnot work fine!
    # "players": Selfish(NB_PLAYERS, klUCBPlus, nbArms).childs  # XXX doesnot work fine!
    # "players": Selfish(NB_PLAYERS, klUCBHPlus, nbArms, horizon=HORIZON).childs  # XXX doesnot work fine!
    # "players": Selfish(NB_PLAYERS, BayesUCB, nbArms).childs  # XXX doesnot work fine!
    # "players": Selfish(NB_PLAYERS, Thompson, nbArms).childs  # XXX works fine!
    # "players": Selfish(NB_PLAYERS, SoftmaxDecreasing, nbArms).childs
    # "players": Selfish(NB_PLAYERS, AdBandits, nbArms, alpha=0.5, horizon=HORIZON).childs

    # --- DONE Using multi-player Centralized policy
    # XXX each player needs to now the number of players, OF COURSE this is not very physically plausible
    # "players": CentralizedNotFair(NB_PLAYERS, nbArms).childs
    # "players": CentralizedFair(NB_PLAYERS, nbArms).childs

    # --- DONE Using multi-player Oracle policy
    # XXX they need a perfect knowledge on the arms, OF COURSE this is not physically plausible at all
    # "players": OracleNotFair(NB_PLAYERS, MAB(configuration['environment'][0])).childs
    # "players": OracleFair(NB_PLAYERS, MAB(configuration['environment'][0])).childs

    # --- DONE Using single-player Musical Chair policy
    # "players": Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.2, Time1=HORIZON).childs  # OK Estimate nbPlayers in Time0 initial rounds
    # "players": Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.1, Time1=HORIZON).childs  # OK Estimate nbPlayers in Time0 initial rounds
    # "players": Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.05, Time1=HORIZON).childs  # OK Estimate nbPlayers in Time0 initial rounds
    # "players": Selfish(NB_PLAYERS, MusicalChair, nbArms, Time0=0.01, Time1=HORIZON).childs  # OK Estimate nbPlayers in Time0 initial rounds

    # --- DONE Using single-player MEGA policy
    # "players": Selfish(NB_PLAYERS, MEGA, nbArms, p0=0.6, alpha=0.5, beta=0.8, c=0.1, d=0.5).childs  # FIXME how to chose the 5 parameters ??

    # --- DONE Using single-player rhoRand policy
    # "players": rhoRand(NB_PLAYERS, UCB, nbArms).childs
    # "players": rhoRand(NB_PLAYERS, Thompson, nbArms).childs
    "players": rhoRand(NB_PLAYERS, SoftmaxDecreasing, nbArms).childs
    # "players": rhoRand(NB_PLAYERS, klUCB, nbArms).childs
    # "players": rhoRand(NB_PLAYERS, klUCBPlus, nbArms).childs
    # "players": rhoRand(NB_PLAYERS, MOSS, nbArms).childs
})
# TODO the EvaluatorMultiPlayers should regenerate the list of players in every repetitions, to have at the end results on the average behavior of these randomized multi-players policies

print("Loaded experiments configuration from 'configuration.py' :")
print("configuration =", configuration)  # DEBUG
