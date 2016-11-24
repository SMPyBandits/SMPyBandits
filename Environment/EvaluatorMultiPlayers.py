# -*- coding: utf-8 -*-
""" Evaluator class to wrap and run the simulations, for the multi-players case.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.1"

# Generic imports
from copy import deepcopy
from random import shuffle
# Scientific imports
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    print("joblib not found. Install it from pypi ('pip install joblib') or conda.")
    USE_JOBLIB = False
# Local imports
from .ResultMultiPlayers import ResultMultiPlayers
from .MAB import MAB
from .CollisionModels import defaultCollisionModel

# Fix the issue with colors, cf. my question here https://github.com/matplotlib/matplotlib/issues/7505
# cf. http://matplotlib.org/cycler/ and http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
USE_4_COLORS = True
USE_4_COLORS = False
colors = ['r', 'g', 'b', 'k']
linestyles = ['-', '--', ':', '-.']
linewidths = [2, 3, 3, 3]
if not USE_4_COLORS:
    colors = ['blue', 'green', 'red', 'black', 'purple', 'orange', 'teal', 'pink', 'brown', 'magenta', 'lime', 'coral', 'lightblue', 'plum', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']
    linestyles = linestyles * (1 + int(len(colors) / float(len(linestyles))))
    linestyles = linestyles[:len(colors)]
    linewidths = linewidths * (1 + int(len(colors) / float(len(linewidths))))
    linewidths = linewidths[:len(colors)]
# Default configuration for the plots: cycle through these colors, linestyles and linewidths
# plt.rc('axes', prop_cycle=(cycler('color', colors) + cycler('linestyle', linestyles) + cycler('linewidth', linewidths)))
plt.rc('axes', prop_cycle=(cycler('color', colors)))

# Customize here if you want a signature on the titles of each plot
signature = "\n(By Lilian Besson, Nov.2016 - Code on https://github.com/Naereen/AlgoBandits)"


# --- Class EvaluatorMultiPlayers

class EvaluatorMultiPlayers:
    """ Evaluator class to run the simulations, for the multi-players case.
    """

    def __init__(self, configuration):
        # Configuration
        self.cfg = configuration
        # Attributes
        self.nbPlayers = len(self.cfg['players'])
        print("Number of players in the multi-players game:", self.nbPlayers)
        self.horizon = self.cfg['horizon']
        print("Time horizon:", self.horizon)
        self.repetitions = self.cfg['repetitions']
        print("Number of repetitions:", self.repetitions)
        self.collisionModel = self.cfg.get('collisionModel', defaultCollisionModel)
        print("Using collision model:", self.collisionModel.__name__)
        print("  Detail:", self.collisionModel.__doc__)
        # Flags
        self.finalRanksOnAverage = self.cfg.get('finalRanksOnAverage', True)
        self.averageOn = self.cfg.get('averageOn', 5e-3)
        self.useJoblib = USE_JOBLIB and self.cfg['n_jobs'] != 1
        # Internal object memory
        self.envs = []
        self.players = []
        self.__initEnvironments__()
        # Internal vectorial memory
        self.rewards = dict()
        self.bestArmPulls = dict()
        self.pulls = dict()
        print("Number of environments to try:", len(self.envs))
        for env in range(len(self.envs)):
            self.rewards[env] = np.zeros((self.nbPlayers, self.horizon))
            self.bestArmPulls[env] = np.zeros((self.nbPlayers, self.horizon))
            self.pulls[env] = np.zeros((self.nbPlayers, self.envs[env].nbArms))

    def __initEnvironments__(self):
        for armType in self.cfg['environment']:
            self.envs.append(MAB(armType))

    def __initPlayers__(self, env):
        for playerId, player in enumerate(self.cfg['players']):
            print("- Adding player #{} = {} ...".format(playerId + 1, player))  # DEBUG
            self.players.append(player['archtype'](env.nbArms,
                                                   **player['params']))

    def start_all_env(self):
        for envId, env in enumerate(self.envs):
            self.start_one_env(envId, env)

    # @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof
    def start_one_env(self, envId, env):
        print("\nEvaluating environment:", repr(env))
        self.players = []
        self.__initPlayers__(env)
        if self.useJoblib:
            results = joblib.Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
                joblib.delayed(delayed_play)(env, self.players, self.horizon, self.collisionModel)
                for _ in range(self.repetitions)
            )
        else:
            results = []
            for _ in range(self.repetitions):
                r = delayed_play(env, self.players, self.horizon, self.collisionModel)
                results.append(r)
        # Get the position of the best arms
        env = self.envs[envId]
        means = np.array([arm.mean() for arm in env.arms])
        bestarm = np.max(means)
        index_bestarm = np.nonzero(np.isclose(means, bestarm))[0]
        # Store the results
        for r in results:
            self.rewards[envId] += np.cumsum(r.rewards, axis=1)
            self.pulls[envId] += r.pulls
            # FIXME use r.pulls instead of r.choices to count the bestArmPulls
            for playerId in range(self.nbPlayers):
                self.bestArmPulls[envId][playerId, :] += np.cumsum(np.in1d(r.choices[playerId], index_bestarm))

    def getPulls(self, playerId, environmentId):
        return self.pulls[environmentId][playerId, :] / float(self.repetitions)

    def getbestArmPulls(self, playerId, environmentId):
        # We have to divide by a arange() = cumsum(ones) to get a frequency
        return self.bestArmPulls[environmentId][playerId, :] / (float(self.repetitions) * np.arange(start=1, stop=1 + self.horizon))

    def getReward(self, playerId, environmentId):
        return self.rewards[environmentId][playerId, :] / float(self.repetitions)

    def getRegret(self, playerId, environmentId):
        times = np.arange(self.horizon)
        return times * self.envs[environmentId].maxArm - self.getReward(playerId, environmentId)

    # Plotting
    def plotRewards(self, environmentId, savefig=None, semilogx=False):
        plt.figure()
        ymin = 0
        for i, player in enumerate(self.players):
            label = 'Player #{}: {}'.format(i + 1, str(player))
            Y = self.getRegret(i, environmentId)
            ymin = min(ymin, np.min(Y))  # XXX Should be smarter
            if semilogx:
                plt.semilogx(Y, label=label)
            else:
                plt.plot(Y, label=label)
        plt.legend(loc='upper left')
        plt.grid()
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}$".format(self.horizon))
        ymax = plt.ylim()[1]
        plt.ylim(ymin, ymax)
        plt.ylabel(r"Cumulative Regret $R_t$")
        plt.title("Multi-players: personal regret for each player, averaged ${}$ times\nArms: ${}${}".format(self.repetitions, repr(self.envs[environmentId].arms), signature))
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig)
        plt.show()

    def plotBestArmPulls(self, environmentId, savefig=None):
        plt.figure()
        for i, player in enumerate(self.players):
            Y = self.getbestArmPulls(i, environmentId)
            plt.plot(Y, label=str(player))
        plt.legend(loc='lower right')
        plt.grid()
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}$".format(self.horizon))
        plt.ylim(-0.03, 1.03)
        plt.ylabel(r"Frequency of pulls of the optimal arm")
        plt.title("Best arm pulls frequency for different bandit algoritms, averaged ${}$ times\nArms: ${}${}".format(self.cfg['repetitions'], repr(self.envs[environmentId].arms), signature))
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig)
        plt.show()

    # TODO plot of number of collisions

    def giveFinalRanking(self, environmentId):
        print("\nFinal ranking for this environment #{} :".format(environmentId))
        lastY = np.zeros(self.nbPlayers)
        for i, player in enumerate(self.players):
            Y = self.getRegret(i, environmentId)
            if self.finalRanksOnAverage:
                lastY[i] = np.mean(Y[-int(self.averageOn * self.horizon)])   # get average value during the last 0.5% of the iterations
            else:
                lastY[i] = Y[-1]  # get the last value
        # print("lastY =", lastY)  # DEBUG
        # Sort lastY and give ranking
        index_of_sorting = np.argsort(lastY)
        # print("index_of_sorting =", index_of_sorting)  # DEBUG
        for i, k in enumerate(index_of_sorting):
            player = self.players[k]
            print("- Player '{}'\twas ranked\t{} / {} for this simulation (last regret = {:.3f}).".format(str(player), i + 1, self.nbPlayers, lastY[k]))
        return lastY, index_of_sorting


# Helper function for the parallelization
# @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof
def delayed_play(env, players, horizon, collisionModel):
    # We have to deepcopy because this function is Parallel-ized
    env = deepcopy(env)
    nbArms = env.nbArms
    players = deepcopy(players)
    horizon = deepcopy(horizon)
    nbPlayers = len(players)

    for player in players:
        player.startGame()
    result = ResultMultiPlayers(env.nbArms, horizon, nbPlayers)

    choices = np.zeros(nbPlayers, dtype=int)
    rewards = np.zeros(nbPlayers)
    pulls = np.zeros((nbPlayers, nbArms), dtype=int)
    for t in range(horizon):
        # choices *= 0
        # XXX Is this step needed?
        rewards *= 0
        pulls *= 0
        # Every player decides which arm to pull
        for i, player in enumerate(players):
            choice = player.choice()
            choices[i] = choice
            # print(" Round t = \t{}, player \t#{}/{} ({}) \tchose : {} ...".format(t, i + 1, len(players), player, choice))  # DEBUG
        # Then we decide if there is collisions and what to do why them
        collisionModel(t, env.arms, players, choices, rewards, pulls)
        # Finally we store the results
        result.store(t, choices, rewards, pulls)
    return result
