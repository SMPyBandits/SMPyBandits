# -*- coding: utf-8 -*-
"""
Configuration for the simulations, for the multi-players case.
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
REPETITIONS = 4  # Nb of cores, to h    ave exactly one repetition process by cores
# REPETITIONS = 10000
# REPETITIONS = 1000
REPETITIONS = 200
# REPETITIONS = 100
# REPETITIONS = 50
# REPETITIONS = 20
# REPETITIONS = 10
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

#: NB_PLAYERS : number of players for the game. Should be >= 2 and <= number of arms.
NB_PLAYERS = 1    # Less that the number of arms
NB_PLAYERS = 2    # Less that the number of arms
NB_PLAYERS = 3    # Less that the number of arms
# NB_PLAYERS = 4    # Less that the number of arms
# NB_PLAYERS = 5    # Less that the number of arms
# NB_PLAYERS = 6    # Less that the number of arms
# NB_PLAYERS = 7    # Less that the number of arms
# NB_PLAYERS = 8    # Less that the number of arms
# NB_PLAYERS = 9    # Less that the number of arms
# NB_PLAYERS = 12   # Less that the number of arms
# NB_PLAYERS = 17   # Just the number of arms
# NB_PLAYERS = 25   # XXX More than the number of arms !!
# NB_PLAYERS = 30   # XXX More than the number of arms !!
NB_PLAYERS = int(getenv('M', NB_PLAYERS))
NB_PLAYERS = int(getenv('NB_PLAYERS', NB_PLAYERS))

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


#: Should we cache rewards? The random rewards will be the same for all the REPETITIONS simulations for each algorithms.
CACHE_REWARDS = False  # XXX to disable manually this feature


#: Number of arms for non-hard-coded problems (Bayesian problems)
NB_ARMS = 2 * NB_PLAYERS
NB_ARMS = int(getenv('K', NB_ARMS))
NB_ARMS = int(getenv('NB_ARMS', NB_ARMS))

#: Default value for the lower value of means
LOWER = 0.
#: Default value for the amplitude value of means
AMPLITUDE = 1.

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
if ARM_TYPE in ["UnboundedGaussian"]:
    LOWER = -5
    AMPLITUDE = 10

LOWER = float(getenv('LOWER', LOWER))
AMPLITUDE = float(getenv('AMPLITUDE', AMPLITUDE))
assert AMPLITUDE > 0, "Error: invalid amplitude = {:.3g} but has to be > 0."  # DEBUG

ARM_TYPE_str = str(ARM_TYPE)
ARM_TYPE = mapping_ARM_TYPE[ARM_TYPE]

#: True to use bayesian problem
ENVIRONMENT_BAYESIAN = False
ENVIRONMENT_BAYESIAN = getenv('BAYES', str(ENVIRONMENT_BAYESIAN)) == 'True'


#: This dictionary configures the experiments
configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Collision model
    "collisionModel": collisionModel,
    # --- Other parameters for the Evaluator
    "finalRanksOnAverage": True,  # Use an average instead of the last value for the final ranking of the tested players
    "averageOn": 1e-3,  # Average the final rank on the 1.% last time steps
    # --- Should we plot the lower-bounds or not?
    "plot_lowerbounds": True,  # XXX Default
    # "plot_lowerbounds": False,
    # --- Arms
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
        # },
        # {   # A very very easy problem: 3 arms, one bad, one average, one good
        #     "arm_type": Bernoulli,
        #     "params": [0.3, 0.5, 0.7]  # uniformMeans(3, 0.3)
        # },
        # {   # A harder problem: 3 arms, one bad, one average, one good
        #     "arm_type": Bernoulli,
        #     "params": [0.49, 0.5, 0.51]  # uniformMeans(3, 0.49)
        # },
        # {   # A very easy problem (X arms), but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "params": uniformMeans(NB_PLAYERS, 1 / (1. + NB_PLAYERS))
        # }
        # # XXX Default!
        # {   # A very easy problem (X arms), but it is used in a lot of articles
        #     "arm_type": ARM_TYPE,
        #     "params": uniformMeans(NB_ARMS, 1 / (1. + NB_ARMS))
        # }
        # {   # A very easy problem (9 arms), but it is used in a lot of articles
        #     "arm_type": Bernoulli,
        #     "params": uniformMeans(9, 1 / (1. + 9))
        # }
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
        # {   # XXX To only test the orthogonalization (collision avoidance) protocol
        #     "arm_type": Bernoulli,
        #     "params": [1] * NB_ARMS
        # }
        # {   # XXX To only test the orthogonalization (collision avoidance) protocol
        #     "arm_type": Bernoulli,
        #     "params": ([0] * (NB_ARMS - NB_PLAYERS)) + ([1] * NB_PLAYERS)
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
        # {   # A Bayesian problem: every repetition use a different mean vectors!
        #     "arm_type": ARM_TYPE,
        #     "params": {
        #         "function": randomMeans,
        #         "args": {
        #             "nbArms": NB_ARMS,
        #             "mingap": None,
        #             # "mingap": 0.05,
        #             # "mingap": 1. / (3. * NB_ARMS),
        #             "lower": 0.,
        #             "amplitude": 1.,
        #             "isSorted": True,
        #         }
        #     }
        # },
        # XXX Default!
        {   # A very easy problem (X arms), but it is used in a lot of articles
            "arm_type": ARM_TYPE,
            "params": uniformMeans(NB_ARMS, 1 / (1. + NB_ARMS))
        }
        # {   # A Bayesian problem: every repetition use a different mean vectors!
        #     "arm_type": ARM_TYPE,
        #     "params": {
        #         "function": randomMeansWithGapBetweenMbestMworst,
        #         "args": {
        #             "nbArms": NB_ARMS,
        #             "nbPlayers": NB_PLAYERS,
        #             "mingap": 0.1,
        #             "lower": 0.,
        #             "amplitude": 1.,
        #             "isSorted": True,
        #         }
        #     }
        # },
        # {   # XXX What happens if arms in Mbest are non unique?
        #     "arm_type": Bernoulli,
        #     "params": [0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.8, 0.9, 0.9]
        #     # nbPlayers = 5
        # }
    ],
}

if ENVIRONMENT_BAYESIAN:
    configuration["environment"] = [  # XXX Bernoulli arms
        {   # A Bayesian problem: every repetition use a different mean vectors!
            "arm_type": ARM_TYPE,
            "params": {
                "function": randomMeans,
                "args": {
                    "nbArms": NB_ARMS,
                    # "mingap": None,
                    # "mingap": 0.0000001,
                    # "mingap": 0.1,
                    "mingap": 1. / (3 * NB_ARMS),
                    "lower": LOWER,
                    "amplitude": AMPLITUDE,
                    "isSorted": True,
                }
            }
        },
    ]


try:
    #: Number of arms *in the first environment*
    nbArms = int(configuration['environment'][0]['params']['args']['nbArms'])
except (TypeError, KeyError):
    nbArms = len(configuration['environment'][0]['params'])

if len(configuration['environment']) > 1:
    print("WARNING do not use this hack if you try to use more than one environment.")

#: Compute the gap of the first problem.
#: (for d in MEGA's parameters, and epsilon for MusicalChair's parameters)
try:
    GAP = np.min(np.diff(np.sort(configuration['environment'][0]['params'])))
except (ValueError, np.AxisError):
    print("Warning: using the default value for the GAP (Bayesian environment maybe?)")  # DEBUG
    GAP = 1. / (3 * NB_ARMS)


configuration["successive_players"] = [

    # ---- rhoRand etc
    rhoRand(NB_PLAYERS, nbArms, klUCB).children,
    # # rhoRand(NB_PLAYERS, nbArms, BESA).children,
    # # [ Aggregator(nbArms, children=[  # XXX Not efficient!
    # #         lambda: rhoRand(1 + x, nbArms, klUCB).children[0]
    # #         for x in range(NB_ARMS)
    # #         # for x in set.intersection(set(range(NB_ARMS)), [NB_PLAYERS - 1, NB_PLAYERS, NB_PLAYERS + 1])
    # #     ]) for _ in range(NB_PLAYERS)
    # # ],
    # # EstimateM(NB_PLAYERS, nbArms, rhoRand, klUCB).children,
    # rhoEst(NB_PLAYERS, nbArms, klUCB).children,  # = EstimateM(... rhoRand, klUCB)
    # # rhoEst(NB_PLAYERS, nbArms, BESA).children,  # = EstimateM(... rhoRand, klUCB)
    # # rhoEst(NB_PLAYERS, nbArms, klUCB, threshold=threshold_on_t).children,  # = EstimateM(... rhoRand, klUCB)
    # # EstimateM(NB_PLAYERS, nbArms, rhoRand, klUCB, horizon=HORIZON, threshold=threshold_on_t_with_horizon).children,  # = rhoEstPlus(...)
    # # rhoEstPlus(NB_PLAYERS, nbArms, klUCB, HORIZON).children,
    # # rhoLearn(NB_PLAYERS, nbArms, klUCB, klUCB).children,
    # # rhoLearnExp3(NB_PLAYERS, nbArms, klUCB, feedback_function=binary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # # rhoLearnExp3(NB_PLAYERS, nbArms, klUCB, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3Decreasing).children,

    # # ---- RandTopM
    RandTopM(NB_PLAYERS, nbArms, klUCB).children,
    # # RandTopMCautious(NB_PLAYERS, nbArms, klUCB).children,
    # # RandTopMExtraCautious(NB_PLAYERS, nbArms, klUCB).children,
    # # RandTopMOld(NB_PLAYERS, nbArms, klUCB).children,
    # # [ Aggregator(nbArms, children=[  # XXX Not efficient!
    # #         lambda: RandTopM(1 + x, nbArms, klUCB).children[0]
    # #         for x in range(NB_ARMS)
    # #         # for x in set.intersection(set(range(NB_ARMS)), [NB_PLAYERS - 1, NB_PLAYERS, NB_PLAYERS + 1])
    # #     ]) for _ in range(NB_PLAYERS)
    # # ],
    # # EstimateM(NB_PLAYERS, nbArms, RandTopM, klUCB).children,  # FIXME experimental!
    # RandTopMEst(NB_PLAYERS, nbArms, klUCB).children,  # = EstimateM(... RandTopM, klUCB)
    # # RandTopMEstPlus(NB_PLAYERS, nbArms, klUCB, HORIZON).children,  # FIXME experimental!

    # # ---- Selfish
    # # Selfish(NB_PLAYERS, nbArms, Exp3Decreasing).children,
    # # Selfish(NB_PLAYERS, nbArms, Exp3PlusPlus).children,
    Selfish(NB_PLAYERS, nbArms, klUCB).children,
    # # Selfish(NB_PLAYERS, nbArms, BESA).children,
    # # [ Aggregator(nbArms, children=[Exp3Decreasing, Exp3PlusPlus, UCB, MOSS, klUCB, BayesUCB, Thompson, DMEDPlus]) for _ in range(NB_PLAYERS) ],  # exactly like Selfish(NB_PLAYERS, nbArms, Aggregator, children=[...])
    # # [ Aggregator(nbArms, children=[UCB, MOSS, klUCB, BayesUCB, Thompson, DMEDPlus]) for _ in range(NB_PLAYERS) ],  # exactly like Selfish(NB_PLAYERS, nbArms, Aggregator, children=[...])

    # ---- MCTopM
    MCTopM(NB_PLAYERS, nbArms, klUCB).children,
    # MCTopM(NB_PLAYERS, nbArms, BESA).children,
    # MCTopMCautious(NB_PLAYERS, nbArms, klUCB).children,
    # MCTopMExtraCautious(NB_PLAYERS, nbArms, klUCB).children,
    # MCTopMOld(NB_PLAYERS, nbArms, klUCB).children,
    # [ Aggregator(nbArms, children=[  # XXX Not efficient!
    #         lambda: MCTopM(1 + x, nbArms, klUCB).children[0]
    #         for x in range(NB_ARMS)
    #         # for x in set.intersection(set(range(NB_ARMS)), [NB_PLAYERS - 1, NB_PLAYERS, NB_PLAYERS + 1])
    #     ]) for _ in range(NB_PLAYERS)
    # ],
    # # EstimateM(NB_PLAYERS, nbArms, MCTopM, klUCB).children,  # FIXME experimental!
    # MCTopMEst(NB_PLAYERS, nbArms, klUCB).children,  # = EstimateM(... MCTopM, klUCB)
    # # MCTopMEst(NB_PLAYERS, nbArms, BESA).children,  # = EstimateM(... MCTopM, klUCB)
    # # MCTopMEstPlus(NB_PLAYERS, nbArms, klUCB, HORIZON).children,  # FIXME experimental!
    # # MCTopMEstPlus(NB_PLAYERS, nbArms, BESA, HORIZON).children,  # FIXME experimental!

    # --- 22) Comparing Selfish, rhoRand, rhoLearn, RandTopM for klUCB, and estimating M
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, EmpiricalMeans).children,
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, Exp3Decreasing).children,
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, Exp3PlusPlus).children,
    CentralizedMultiplePlay(NB_PLAYERS, nbArms, klUCB).children,
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, BESA).children,
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, Aggregator, children=[UCB, MOSS, klUCB, BayesUCB, Thompson, DMEDPlus]).children,  # XXX don't work so well

    # # FIXME how to chose the 5 parameters for MEGA policy ?
    # # XXX By trial and error??
    # # d should be smaller than the gap Delta = mu_M* - mu_(M-1)* (gap between Mbest and Mworst)
    # [ MEGA(nbArms, p0=0.1, alpha=0.1, beta=0.5, c=0.1, d=0.99*GAP) for _ in range(NB_PLAYERS) ],  # XXX always linear regret!

    # # # # XXX stupid version with fixed T0 : cannot adapt to any problem
    # # # [ MusicalChair(nbArms, Time0=1000) for _ in range(NB_PLAYERS) ],
    # # [ MusicalChair(nbArms, Time0=50*NB_ARMS) for _ in range(NB_PLAYERS) ],
    # # [ MusicalChair(nbArms, Time0=100*NB_ARMS) for _ in range(NB_PLAYERS) ],
    # # [ MusicalChair(nbArms, Time0=150*NB_ARMS) for _ in range(NB_PLAYERS) ],
    # # # # XXX cheated version, with known gap (epsilon < Delta) and proba of success 5% !
    # # [ MusicalChair(nbArms, Time0=optimalT0(nbArms=NB_ARMS, epsilon=0.99*GAP, delta=0.5)) for _ in range(NB_PLAYERS) ],
    # # [ MusicalChair(nbArms, Time0=optimalT0(nbArms=NB_ARMS, epsilon=0.99*GAP, delta=0.1)) for _ in range(NB_PLAYERS) ],
    # # # XXX cheated version, with known gap and known horizon (proba of success delta < 1 / T) !
    # [ MusicalChair(nbArms, Time0=optimalT0(nbArms=NB_ARMS, epsilon=0.99*GAP, delta=1./(1+HORIZON))) for _ in range(NB_PLAYERS) ],

    # --- 1) CentralizedMultiplePlay
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, UCBalpha, alpha=1).children,
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, BayesUCB).children,

    # --- 2) Musical Chair
    # Selfish(NB_PLAYERS, nbArms, MusicalChair, Time0=0.1, Time1=HORIZON).children,
    # Selfish(NB_PLAYERS, nbArms, MusicalChair, Time0=0.05, Time1=HORIZON).children,
    # Selfish(NB_PLAYERS, nbArms, MusicalChair, Time0=0.005, Time1=HORIZON).children,
    # Selfish(NB_PLAYERS, nbArms, MusicalChair, Time0=0.001, Time1=HORIZON).children,
    # Selfish(NB_PLAYERS, nbArms, EmpiricalMeans).children,

    # --- 3) EmpiricalMeans
    # # rhoRand(NB_PLAYERS, nbArms, EmpiricalMeans).children,
    # rhoEst(NB_PLAYERS, nbArms, EmpiricalMeans).children,

    # --- 4) UCBalpha
    # # rhoLearn(NB_PLAYERS, nbArms, UCBalpha, Uniform, alpha=1).children,  # OK, == rhoRand
    # rhoLearn(NB_PLAYERS, nbArms, UCBalpha, UCB, alpha=1).children,  # OK, == rhoRand
    # rhoRand(NB_PLAYERS, nbArms, UCBalpha, alpha=1).children,
    # # rhoEst(NB_PLAYERS, nbArms, UCBalpha, alpha=1).children,
    # Selfish(NB_PLAYERS, nbArms, UCBalpha, alpha=1).children,

    # --- 5) klUCBPlus
    # Selfish(NB_PLAYERS, nbArms, klUCBPlus).children,
    # rhoRand(NB_PLAYERS, nbArms, klUCBPlus).children,
    # rhoEst(NB_PLAYERS, nbArms, klUCBPlus).children,
    # # rhoLearn(NB_PLAYERS, nbArms, klUCBPlus, klUCBPlus).children,
    # rhoLearn(NB_PLAYERS, nbArms, klUCBPlus, UCB).children,
    # # rhoLearn(NB_PLAYERS, nbArms, klUCBPlus, EpsilonDecreasing).children,
    # # rhoLearn(NB_PLAYERS, nbArms, klUCBPlus, SoftmaxDecreasing).children,
    # # rhoEst(NB_PLAYERS, nbArms, klUCBPlus).children,

    # --- 6) Thompson
    # Selfish(NB_PLAYERS, nbArms, Thompson).children,
    # rhoRand(NB_PLAYERS, nbArms, Thompson).children,
    # # rhoEst(NB_PLAYERS, nbArms, Thompson).children,

    # # --- 7) rhoLearn with BayesUCB
    # Selfish(NB_PLAYERS, nbArms, BayesUCB).children,
    # rhoRand(NB_PLAYERS, nbArms, BayesUCB).children,
    # # rhoEst(NB_PLAYERS, nbArms, BayesUCB).children,
    # # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, SoftmaxDecreasing).children,
    # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, UCBalpha).children,
    # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, Thompson).children,
    # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, klUCBPlus).children,
    # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, BayesUCB).children,

    # --- 8) Aggregator
    # Selfish(NB_PLAYERS, nbArms, Aggregator, unbiased=UNBIASED, update_all_children=UPDATE_ALL_CHILDREN, decreaseRate="auto", update_like_exp4=UPDATE_LIKE_EXP4, children=[UCBalpha, Thompson, klUCBPlus, BayesUCB]).children,
    # rhoRand(NB_PLAYERS, nbArms, Aggregator, unbiased=UNBIASED, update_all_children=UPDATE_ALL_CHILDREN, decreaseRate="auto", update_like_exp4=UPDATE_LIKE_EXP4, children=[UCBalpha, Thompson, klUCBPlus, BayesUCB]).children,
    # # rhoEst(NB_PLAYERS, nbArms, Aggregator, unbiased=UNBIASED, update_all_children=UPDATE_ALL_CHILDREN, decreaseRate="auto", update_like_exp4=UPDATE_LIKE_EXP4, children=[Thompson, klUCBPlus, BayesUCB]).children,

    # # --- 9) Comparing Selfish, rhoRand (and variants) with different learning algorithms
    # Selfish(NB_PLAYERS, nbArms, BayesUCB).children,
    # rhoRand(NB_PLAYERS, nbArms, BayesUCB).children,
    # # rhoRandRotating(NB_PLAYERS, nbArms, BayesUCB).children,
    # # rhoRandALOHA(NB_PLAYERS, nbArms, BayesUCB).children,
    # Selfish(NB_PLAYERS, nbArms, klUCBPlus).children,
    # rhoRand(NB_PLAYERS, nbArms, klUCBPlus).children,
    # # rhoRandRotating(NB_PLAYERS, nbArms, klUCBPlus).children,
    # # rhoRandALOHA(NB_PLAYERS, nbArms, klUCBPlus).children,
    # Selfish(NB_PLAYERS, nbArms, Thompson).children,
    # rhoRand(NB_PLAYERS, nbArms, Thompson).children,
    # # rhoRandRotating(NB_PLAYERS, nbArms, Thompson).children,
    # # rhoRandALOHA(NB_PLAYERS, nbArms, Thompson).children,

    # --- 10) Mixing rhoRand or Selfish with different learning algorithms
    # rhoRand(int(NB_PLAYERS / 3), nbArms, BayesUCB, maxRank=NB_PLAYERS).children \
    # + rhoRand(int(NB_PLAYERS / 3), nbArms, klUCBPlus, maxRank=NB_PLAYERS).children \
    # + rhoRand(int(NB_PLAYERS / 3), nbArms, Thompson, maxRank=NB_PLAYERS).children,
    # Selfish(int(NB_PLAYERS / 3), nbArms, BayesUCB).children \
    # + Selfish(int(NB_PLAYERS / 3), nbArms, klUCBPlus).children \
    # + Selfish(int(NB_PLAYERS / 3), nbArms, Thompson).children,

    # --- 11) Comparing different "robust" ThompsonSampling algorithms
    # Selfish(NB_PLAYERS, nbArms, ThompsonRobust, averageOn=1).children,
    # rhoRand(NB_PLAYERS, nbArms, ThompsonRobust, averageOn=1).children,
    # Selfish(NB_PLAYERS, nbArms, ThompsonRobust, averageOn=2).children,
    # rhoRand(NB_PLAYERS, nbArms, ThompsonRobust, averageOn=2).children,
    # Selfish(NB_PLAYERS, nbArms, ThompsonRobust, averageOn=5).children,
    # rhoRand(NB_PLAYERS, nbArms, ThompsonRobust, averageOn=5).children,
    # Selfish(NB_PLAYERS, nbArms, ThompsonRobust, averageOn=10).children,
    # rhoRand(NB_PLAYERS, nbArms, ThompsonRobust, averageOn=10).children,

    # --- 12) Comparing different rhoRandSticky algorithms
    # rhoRandSticky(NB_PLAYERS, nbArms, BayesUCB, stickyTime=1).children,
    # rhoRandSticky(NB_PLAYERS, nbArms, BayesUCB, stickyTime=2).children,
    # rhoRandSticky(NB_PLAYERS, nbArms, BayesUCB, stickyTime=5).children,
    # rhoRandSticky(NB_PLAYERS, nbArms, BayesUCB, stickyTime=10).children,
    # rhoRandSticky(NB_PLAYERS, nbArms, BayesUCB, stickyTime=50).children,
    # rhoRandSticky(NB_PLAYERS, nbArms, BayesUCB, stickyTime=100).children,
    # rhoRandSticky(NB_PLAYERS, nbArms, BayesUCB, stickyTime=200).children,
    # rhoRandSticky(NB_PLAYERS, nbArms, BayesUCB, stickyTime=np.inf).children,  # should be = classic rhoRand

    # # --- 13) Comparing Selfish, and rhoRand with or without initial orthogonal ranks
    # Selfish(NB_PLAYERS, nbArms, BayesUCB).children,
    # rhoRand(NB_PLAYERS, nbArms, BayesUCB).children,
    # rhoCentralized(NB_PLAYERS, nbArms, BayesUCB).children,
    # Selfish(NB_PLAYERS, nbArms, klUCBPlus).children,
    # rhoRand(NB_PLAYERS, nbArms, klUCBPlus).children,
    # rhoCentralized(NB_PLAYERS, nbArms, klUCBPlus).children,
    # Selfish(NB_PLAYERS, nbArms, Thompson).children,
    # rhoRand(NB_PLAYERS, nbArms, Thompson).children,
    # rhoCentralized(NB_PLAYERS, nbArms, Thompson).children,

    # # --- 14) Comparing rhoRand or Selfish for ApproximatedFHGittins, different alpha. The smaller alpha, the better
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, BayesUCB).children,
    # CentralizedIMP(NB_PLAYERS, nbArms, BayesUCB).children,
    # Selfish(NB_PLAYERS, nbArms, BayesUCB).children,
    # # Selfish(NB_PLAYERS, nbArms, ApproximatedFHGittins, horizon=1.1 * HORIZON, alpha=2).children,
    # Selfish(NB_PLAYERS, nbArms, ApproximatedFHGittins, horizon=1.1 * HORIZON, alpha=1).children,
    # Selfish(NB_PLAYERS, nbArms, ApproximatedFHGittins, horizon=1.1 * HORIZON, alpha=0.5).children,
    # Selfish(NB_PLAYERS, nbArms, ApproximatedFHGittins, horizon=1.1 * HORIZON, alpha=0.25).children,
    # # Selfish(NB_PLAYERS, nbArms, ApproximatedFHGittins, horizon=1.1 * HORIZON, alpha=0.05).children,
    # rhoRand(NB_PLAYERS, nbArms, BayesUCB).children,
    # # rhoRand(NB_PLAYERS, nbArms, ApproximatedFHGittins, horizon=1.1 * HORIZON, alpha=2).children,
    # rhoRand(NB_PLAYERS, nbArms, ApproximatedFHGittins, horizon=1.1 * HORIZON, alpha=1).children,
    # rhoRand(NB_PLAYERS, nbArms, ApproximatedFHGittins, horizon=1.1 * HORIZON, alpha=0.5).children,
    # rhoRand(NB_PLAYERS, nbArms, ApproximatedFHGittins, horizon=1.1 * HORIZON, alpha=0.25).children,
    # # rhoRand(NB_PLAYERS, nbArms, ApproximatedFHGittins, horizon=1.1 * HORIZON, alpha=0.05).children,

    # # --- 15) Comparing Selfish, rhoRand (and variants) with different learning algorithms
    # Selfish(NB_PLAYERS, nbArms, SoftMix).children,
    # rhoRand(NB_PLAYERS, nbArms, SoftMix).children,
    # # Selfish(NB_PLAYERS, nbArms, SoftmaxDecreasing).children,
    # # rhoRand(NB_PLAYERS, nbArms, SoftmaxDecreasing).children,
    # # Selfish(NB_PLAYERS, nbArms, Exp3).children,
    # # rhoRand(NB_PLAYERS, nbArms, Exp3).children,
    # # Selfish(NB_PLAYERS, nbArms, Exp3WithHorizon, horizon=HORIZON).children,
    # # rhoRand(NB_PLAYERS, nbArms, Exp3WithHorizon, horizon=HORIZON).children,
    # Selfish(NB_PLAYERS, nbArms, Exp3SoftMix).children,
    # rhoRand(NB_PLAYERS, nbArms, Exp3SoftMix).children,
    # # XXX against stochastic algorithms
    # Selfish(NB_PLAYERS, nbArms, BayesUCB).children,
    # rhoRand(NB_PLAYERS, nbArms, BayesUCB).children,
    # Selfish(NB_PLAYERS, nbArms, klUCBPlus).children,
    # rhoRand(NB_PLAYERS, nbArms, klUCBPlus).children,
    # Selfish(NB_PLAYERS, nbArms, Thompson).children,
    # rhoRand(NB_PLAYERS, nbArms, Thompson).children,

    # # --- 16) Comparing rhoLearn and rhoLearnEst (doesn't know M)
    # Selfish(NB_PLAYERS, nbArms, BayesUCB).children,
    # rhoRand(NB_PLAYERS, nbArms, BayesUCB).children,
    # rhoLearn(NB_PLAYERS, nbArms, BayesUCB).children,  # use Uniform, so = rhoRand
    # rhoLearnEst(NB_PLAYERS, nbArms, BayesUCB).children,  # use Uniform, so ~= bad rhoRand
    # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, BayesUCB).children,
    # rhoLearnEst(NB_PLAYERS, nbArms, BayesUCB, BayesUCB).children,  # should be bad!
    # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, klUCBPlus).children,
    # rhoLearnEst(NB_PLAYERS, nbArms, BayesUCB, klUCBPlus).children,  # should be bad!
    # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, Thompson).children,
    # rhoLearnEst(NB_PLAYERS, nbArms, BayesUCB, Thompson).children,  # should be bad!

    # # --- 17) Comparing rhoRand, rhoLearn[BayesUCB], rhoLearn[klUCBPlus] and rhoLearn[Thompson], against rhoLearnExp3, all with BayesUCB for arm selection
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, BayesUCB).children,
    # Selfish(NB_PLAYERS, nbArms, BayesUCB).children,
    # # Selfish(NB_PLAYERS, nbArms, Exp3Decreasing).children,
    # # Selfish(NB_PLAYERS, nbArms, Exp3SoftMix).children,
    # rhoRand(NB_PLAYERS, nbArms, BayesUCB).children,
    # # rhoLearn(NB_PLAYERS, nbArms, BayesUCB).children,  # use Uniform, so = rhoRand
    # # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, BayesUCB).children,
    # # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, klUCB).children,
    # # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, Thompson).children,
    # # rhoLearnExp3(NB_PLAYERS, nbArms, BayesUCB, feedback_function=binary_feedback, rankSelectionAlgo=Exp3SoftMix).children,
    # # rhoLearnExp3(NB_PLAYERS, nbArms, BayesUCB, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3SoftMix).children,
    # # rhoLearnExp3(NB_PLAYERS, nbArms, BayesUCB, feedback_function=binary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # # rhoLearnExp3(NB_PLAYERS, nbArms, BayesUCB, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # # # rhoLearnExp3(NB_PLAYERS, nbArms, BayesUCB, feedback_function=binary_feedback, rankSelectionAlgo=lambda nbArms: Exp3WithHorizon(nbArms, HORIZON)).children,
    # # # rhoLearnExp3(NB_PLAYERS, nbArms, BayesUCB, feedback_function=ternary_feedback, rankSelectionAlgo=lambda nbArms: Exp3WithHorizon(nbArms, HORIZON)).children,

    # # --- 18) Comparing rhoRand, rhoLearn[BayesUCB], rhoLearn[klUCBPlus] and rhoLearn[Thompson], against rhoLearnExp3, all with klUCB for arm selection
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, klUCB).children,
    # Selfish(NB_PLAYERS, nbArms, klUCB).children,
    # Selfish(NB_PLAYERS, nbArms, Exp3Decreasing).children,
    # Selfish(NB_PLAYERS, nbArms, Exp3SoftMix).children,
    # # rhoRand(NB_PLAYERS, nbArms, klUCB).children,
    # rhoLearn(NB_PLAYERS, nbArms, klUCB).children,  # use Uniform, so = rhoRand
    # rhoLearn(NB_PLAYERS, nbArms, klUCB, BayesUCB).children,
    # rhoLearn(NB_PLAYERS, nbArms, klUCB, klUCB).children,
    # rhoLearn(NB_PLAYERS, nbArms, klUCB, Thompson).children,
    # rhoLearnExp3(NB_PLAYERS, nbArms, klUCB, feedback_function=binary_feedback, rankSelectionAlgo=Exp3SoftMix).children,
    # rhoLearnExp3(NB_PLAYERS, nbArms, klUCB, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3SoftMix).children,
    # rhoLearnExp3(NB_PLAYERS, nbArms, klUCB, feedback_function=binary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # rhoLearnExp3(NB_PLAYERS, nbArms, klUCB, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3Decreasing).children,

    # # --- 19) Comparing Selfish[UCB], rhoRand[UCB], rhoLearn[UCB], rhoLearnExp3[UCB] against RandTopM[UCB]
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, UCB).children,
    # Selfish(NB_PLAYERS, nbArms, UCB).children,
    # rhoRand(NB_PLAYERS, nbArms, UCB).children,
    # rhoLearn(NB_PLAYERS, nbArms, UCB, UCB).children,
    # # rhoLearn(NB_PLAYERS, nbArms, UCB, klUCB).children,
    # # rhoLearn(NB_PLAYERS, nbArms, UCB, Thompson).children,
    # rhoLearnExp3(NB_PLAYERS, nbArms, UCB, feedback_function=binary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # rhoLearnExp3(NB_PLAYERS, nbArms, UCB, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # RandTopM(NB_PLAYERS, nbArms, UCB).children,
    # MCTopM(NB_PLAYERS, nbArms, UCB).children,

    # # --- 20) Comparing Selfish[BayesUCB], rhoRand[BayesUCB], rhoLearn[BayesUCB], rhoLearnExp3[BayesUCB] against RandTopM[BayesUCB]
    # # XXX it is *failing* with RandTopM[BayesUCB]
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, BayesUCB).children,
    # Selfish(NB_PLAYERS, nbArms, BayesUCB).children,
    # rhoRand(NB_PLAYERS, nbArms, BayesUCB).children,
    # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, BayesUCB).children,
    # # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, klUCB).children,
    # # rhoLearn(NB_PLAYERS, nbArms, BayesUCB, Thompson).children,
    # rhoLearnExp3(NB_PLAYERS, nbArms, BayesUCB, feedback_function=binary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # rhoLearnExp3(NB_PLAYERS, nbArms, BayesUCB, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # RandTopM(NB_PLAYERS, nbArms, BayesUCB).children,
    # MCTopM(NB_PLAYERS, nbArms, BayesUCB).children,

    # # --- 21) Comparing Selfish[Thompson], rhoRand[Thompson], rhoLearn[Thompson], rhoLearnExp3[Thompson] against RandTopM[Thompson]
    # CentralizedMultiplePlay(NB_PLAYERS, nbArms, Thompson).children,
    # Selfish(NB_PLAYERS, nbArms, Thompson).children,
    # rhoRand(NB_PLAYERS, nbArms, Thompson).children,
    # # rhoLearn(NB_PLAYERS, nbArms, Thompson, klUCB).children,
    # # rhoLearn(NB_PLAYERS, nbArms, Thompson, BayesUCB).children,
    # rhoLearn(NB_PLAYERS, nbArms, Thompson, Thompson).children,
    # rhoLearnExp3(NB_PLAYERS, nbArms, Thompson, feedback_function=binary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # rhoLearnExp3(NB_PLAYERS, nbArms, Thompson, feedback_function=ternary_feedback, rankSelectionAlgo=Exp3Decreasing).children,
    # RandTopM(NB_PLAYERS, nbArms, Thompson).children,
    # MCTopM(NB_PLAYERS, nbArms, Thompson).children,
]

# XXX Comparing different rhoRand approaches
# configuration["successive_players"] = [
#     rhoRand(NB_PLAYERS, nbArms, UCBalpha, alpha=1).children,  # This one is efficient!
#     rhoRand(NB_PLAYERS, nbArms, UCBalpha, alpha=0.25).children,  # This one is efficient!
#     rhoRand(NB_PLAYERS, nbArms, MOSS).children,
#     rhoRand(NB_PLAYERS, nbArms, klUCB).children,
#     rhoRand(NB_PLAYERS, nbArms, klUCBPlus).children,
#     rhoRand(NB_PLAYERS, nbArms, Thompson).children,
#     rhoRand(NB_PLAYERS, nbArms, SoftmaxDecreasing).children,
#     rhoRand(NB_PLAYERS, nbArms, BayesUCB).children,
#     rhoRand(NB_PLAYERS, nbArms, AdBandits, alpha=0.5, horizon=HORIZON).children,
# ]


# XXX Comparing different ALOHA approaches
# from itertools import product  # XXX If needed!
# p0 = 1. / NB_PLAYERS
# p0 = 0.75
# configuration["successive_players"] = [
#     Selfish(NB_PLAYERS, nbArms, BayesUCB).children,  # This one is efficient!
# ] + [
#     ALOHA(NB_PLAYERS, nbArms, BayesUCB, p0=p0, alpha_p0=alpha_p0, beta=beta).children
#     # ALOHA(NB_PLAYERS, nbArms, BayesUCB, p0=p0, alpha_p0=alpha_p0, ftnext=tnext_log).children,
#     for alpha_p0, beta in product([0.05, 0.25, 0.5, 0.75, 0.95], repeat=2)
#     # for alpha_p0, beta in product([0.1, 0.5, 0.9], repeat=2)
# ]

# # XXX Comparing different centralized approaches
# configuration["successive_players"] = [
#     CentralizedMultiplePlay(NB_PLAYERS, nbArms, UCBalpha).children,
#     CentralizedIMP(NB_PLAYERS, nbArms, UCBalpha).children,
#     CentralizedMultiplePlay(NB_PLAYERS, nbArms, Thompson).children,
#     CentralizedIMP(NB_PLAYERS, nbArms, Thompson).children,
#     CentralizedMultiplePlay(NB_PLAYERS, nbArms, klUCBPlus).children,
# ]


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
    # "players": CentralizedMultiplePlay(NB_PLAYERS, nbArms, UCB, uniformAllocation=False).children
    # "players": CentralizedMultiplePlay(NB_PLAYERS, nbArms, UCB, uniformAllocation=True).children
    # "players": CentralizedMultiplePlay(NB_PLAYERS, nbArms, Thompson, uniformAllocation=False).children
    # "players": CentralizedMultiplePlay(NB_PLAYERS, nbArms, Thompson, uniformAllocation=True).children

    # --- DONE Using a smart Centralized policy, based on choiceIMP() -- It's not better, in fact
    # "players": CentralizedIMP(NB_PLAYERS, nbArms, UCB, uniformAllocation=False).children
    # "players": CentralizedIMP(NB_PLAYERS, nbArms, UCB, uniformAllocation=True).children
    # "players": CentralizedIMP(NB_PLAYERS, nbArms, Thompson, uniformAllocation=False).children
    # "players": CentralizedIMP(NB_PLAYERS, nbArms, Thompson, uniformAllocation=True).children

    # --- DONE Using multi-player Selfish policy
    # "players": Selfish(NB_PLAYERS, nbArms, Uniform).children
    # "players": Selfish(NB_PLAYERS, nbArms, TakeRandomFixedArm).children
    # "players": Selfish(NB_PLAYERS, nbArms, Exp3Decreasing).children
    # "players": Selfish(NB_PLAYERS, nbArms, Exp3WithHorizon, horizon=HORIZON).children
    "players": Selfish(NB_PLAYERS, nbArms, UCB).children
    # "players": Selfish(NB_PLAYERS, nbArms, UCBalpha, alpha=0.25).children  # This one is efficient!
    # "players": Selfish(NB_PLAYERS, nbArms, MOSS).children
    # "players": Selfish(NB_PLAYERS, nbArms, klUCB).children
    # "players": Selfish(NB_PLAYERS, nbArms, klUCBPlus).children
    # "players": Selfish(NB_PLAYERS, nbArms, klUCBHPlus, horizon=HORIZON).children  # Worse than simple klUCB and klUCBPlus
    # "players": Selfish(NB_PLAYERS, nbArms, Thompson).children
    # "players": Selfish(NB_PLAYERS, nbArms, SoftmaxDecreasing).children
    # "players": Selfish(NB_PLAYERS, nbArms, BayesUCB).children
    # "players": Selfish(int(NB_PLAYERS / 3), nbArms, BayesUCB).children \
    #          + Selfish(int(NB_PLAYERS / 3), nbArms, Thompson).children \
    #          + Selfish(int(NB_PLAYERS / 3), nbArms, klUCBPlus).children
    # "players": Selfish(NB_PLAYERS, nbArms, AdBandits, alpha=0.5, horizon=HORIZON).children

    # --- DONE Using multi-player Oracle policy
    # XXX they need a perfect knowledge on the arms, OF COURSE this is not physically plausible at all
    # "players": OracleNotFair(NB_PLAYERS, MAB(configuration['environment'][0])).children
    # "players": OracleFair(NB_PLAYERS, MAB(configuration['environment'][0])).children

    # --- DONE Using single-player Musical Chair policy
    # OK Estimate nbPlayers in Time0 initial rounds
    # "players": Selfish(NB_PLAYERS, nbArms, MusicalChair, Time0=0.2, Time1=HORIZON).children
    # "players": Selfish(NB_PLAYERS, nbArms, MusicalChair, Time0=0.1, Time1=HORIZON).children
    # "players": Selfish(NB_PLAYERS, nbArms, MusicalChair, Time0=0.05, Time1=HORIZON).children
    # "players": Selfish(NB_PLAYERS, nbArms, MusicalChair, Time0=0.005, Time1=HORIZON).children

    # --- DONE Using single-player MEGA policy
    # FIXME how to chose the 5 parameters ??
    # "players": Selfish(NB_PLAYERS, nbArms, MEGA, p0=0.6, alpha=0.5, beta=0.8, c=0.1, d=D).children

    # --- DONE Using single-player ALOHA policy
    # FIXME how to chose the 2 parameters p0 and alpha_p0 ?
    # "players": ALOHA(NB_PLAYERS, nbArms, EpsilonDecreasingMEGA, p0=0.6, alpha_p0=0.5, beta=0.8, c=0.1, d=D).children  # Example to prove that Selfish[MEGA] = ALOHA[EpsilonGreedy]
    # "players": ALOHA(NB_PLAYERS, nbArms, UCB, p0=0.6, alpha_p0=0.5, beta=0.8).children  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, nbArms, MOSS, p0=0.6, alpha_p0=0.5, beta=0.8).children  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, nbArms, klUCBPlus, p0=0.6, alpha_p0=0.5, beta=0.8).children  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, nbArms, Thompson, p0=1. / NB_PLAYERS, alpha_p0=0.01, beta=0.2).children  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, nbArms, Thompson, p0=0.6, alpha_p0=0.99, ftnext=tnext_log).children  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, nbArms, BayesUCB, p0=0.6, alpha_p0=0.5, beta=0.8).children  # TODO try this one!
    # "players": ALOHA(NB_PLAYERS, nbArms, SoftmaxDecreasing, p0=0.6, alpha_p0=0.5).children  # TODO try this one!

    # --- DONE Using single-player rhoRand policy
    # "players": rhoRand(NB_PLAYERS, nbArms, UCB).children
    # "players": rhoRand(NB_PLAYERS, nbArms, klUCBPlus).children
    # "players": rhoRand(NB_PLAYERS, nbArms, Thompson).children
    # "players": rhoRand(NB_PLAYERS, nbArms, BayesUCB).children
    # "players": rhoRand(int(NB_PLAYERS / 3), nbArms, BayesUCB, maxRank=NB_PLAYERS).children \
    #          + rhoRand(int(NB_PLAYERS / 3), nbArms, Thompson, maxRank=NB_PLAYERS).children \
    #          + rhoRand(int(NB_PLAYERS / 3), nbArms, klUCBPlus, maxRank=NB_PLAYERS).children
    # "players": rhoRand(NB_PLAYERS, nbArms, AdBandits, alpha=0.5, horizon=HORIZON).children

    # --- DONE Using single-player rhoEst policy
    # "players": rhoEst(NB_PLAYERS, nbArms, UCB).children
    # "players": rhoEst(NB_PLAYERS, nbArms, klUCBPlus).children
    # "players": rhoEst(NB_PLAYERS, nbArms, Thompson).children
    # "players": rhoEst(NB_PLAYERS, nbArms, BayesUCB).children

    # --- DONE Using single-player rhoLearn policy, with same MAB learning algorithm for selecting the ranks
    # "players": rhoLearn(NB_PLAYERS, nbArms, UCB, UCB).children
    # "players": rhoLearn(NB_PLAYERS, nbArms, klUCBPlus, klUCBPlus).children
    # "players": rhoLearn(NB_PLAYERS, nbArms, Thompson, Thompson).children
    # "players": rhoLearn(NB_PLAYERS, nbArms, BayesUCB, BayesUCB, change_rank_each_step=True).children
    # "players": rhoLearn(NB_PLAYERS, nbArms, BayesUCB, BayesUCB, change_rank_each_step=False).children

    # --- DONE Using single-player stupid rhoRandRand policy
    # "players": rhoRandRand(NB_PLAYERS, nbArms, UCB).children

    # --- DONE Using single-player rhoRandSticky policy
    # "players": rhoRandSticky(NB_PLAYERS, nbArms, UCB, stickyTime=10).children
    # "players": rhoRandSticky(NB_PLAYERS, nbArms, klUCBPlus, stickyTime=10).children
    # "players": rhoRandSticky(NB_PLAYERS, nbArms, Thompson, stickyTime=10).children
    # "players": rhoRandSticky(NB_PLAYERS, nbArms, BayesUCB, stickyTime=10).children
})
# TODO the EvaluatorMultiPlayers should regenerate the list of players in every repetitions, to have at the end results on the average behavior of these randomized multi-players policies


# XXX Huge hack! Use this if you want to modify the legends
configuration.update({
    "append_labels": {
        playerId: cfgplayer.get("append_label", "")
        for playerId, cfgplayer in enumerate(configuration["successive_players"])
        if "append_label" in cfgplayer
    },
    "change_labels": {
        playerId: cfgplayer.get("change_label", "")
        for playerId, cfgplayer in enumerate(configuration["successive_players"])
        if "change_label" in cfgplayer
    }
})


# DONE
print("Loaded experiments configuration from 'configuration.py' :")
print("configuration =", configuration)  # DEBUG
