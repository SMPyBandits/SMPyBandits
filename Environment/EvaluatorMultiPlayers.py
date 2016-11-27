# -*- coding: utf-8 -*-
""" Evaluator class to wrap and run the simulations, for the multi-players case.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.1"

# Generic imports
from copy import deepcopy
# Scientific imports
import numpy as np
import matplotlib.pyplot as plt
try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    print("joblib not found. Install it from pypi ('pip install joblib') or conda.")
    USE_JOBLIB = False
# Local imports
from ._maximizeWindow import maximizeWindow
from .ResultMultiPlayers import ResultMultiPlayers
from .MAB import MAB
from .CollisionModels import defaultCollisionModel

DPI = 140

# Fix the issue with colors, cf. my question here https://github.com/matplotlib/matplotlib/issues/7505
# cf. http://matplotlib.org/cycler/ and http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
try:
    from cycler import cycler
    USE_4_COLORS = True
    USE_4_COLORS = False
    colors = ['r', 'g', 'b', 'k']
    linestyles = ['-', '--', ':', '-.']
    linewidths = [2, 3, 3, 3]
    if not USE_4_COLORS:
        colors = ['blue', 'green', 'red', 'black', 'purple', 'orange', 'teal', 'brown', 'magenta', 'lime', 'coral', 'pink', 'lightblue', 'plum', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']
        linestyles = linestyles * (1 + int(len(colors) / float(len(linestyles))))
        linestyles = linestyles[:len(colors)]
        linewidths = linewidths * (1 + int(len(colors) / float(len(linewidths))))
        linewidths = linewidths[:len(colors)]
    # Default configuration for the plots: cycle through these colors, linestyles and linewidths
    # plt.rc('axes', prop_cycle=(cycler('color', colors) + cycler('linestyle', linestyles) + cycler('linewidth', linewidths)))
    plt.rc('axes', prop_cycle=(cycler('color', colors)))
except:
    print("Using default colors and plot styles (cycler was not available or something went wrong with 'plt.rc' calls?)")

# Customize here if you want a signature on the titles of each plot
signature = "\n(By Lilian Besson, Nov.2016 - Code on https://github.com/Naereen/AlgoBandits)"


# --- Class EvaluatorMultiPlayers

class EvaluatorMultiPlayers(object):
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
        self.pulls = dict()
        self.collisions = dict()
        self.BestArmPulls = dict()
        self.FreeTransmissions = dict()
        print("Number of environments to try:", len(self.envs))
        for env in range(len(self.envs)):
            self.rewards[env] = np.zeros((self.nbPlayers, self.horizon))
            self.pulls[env] = np.zeros((self.nbPlayers, self.envs[env].nbArms))
            self.collisions[env] = np.zeros((self.envs[env].nbArms, self.horizon))
            self.BestArmPulls[env] = np.zeros((self.nbPlayers, self.horizon))
            self.FreeTransmissions[env] = np.zeros((self.nbPlayers, self.horizon))

    def __initEnvironments__(self):
        for armType in self.cfg['environment']:
            self.envs.append(MAB(armType))

    def __initPlayers__(self, env):
        for playerId, player in enumerate(self.cfg['players']):
            print("- Adding player #{} = {} ...".format(playerId + 1, player))  # DEBUG
            if isinstance(player, dict):
                print("  Creating this player from a dictionnary 'player' = {} ...".format(player))  # DEBUG
                self.players.append(player['archtype'](env.nbArms, **player['params']))
            else:
                print("  Using this already created player 'player' = {} ...".format(player))  # DEBUG
                self.players.append(player)

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
        # Get and merge the results from all the 'repetitions'
        for r in results:
            self.rewards[envId] += np.cumsum(r.rewards, axis=1)
            self.pulls[envId] += r.pulls
            self.collisions[envId] += r.collisions
            for playerId in range(self.nbPlayers):
                self.BestArmPulls[envId][playerId, :] += np.cumsum(np.in1d(r.choices[playerId, :], index_bestarm))
                # FIXME there is probably a bug in this computation
                self.FreeTransmissions[envId][playerId, :] += np.array([r.choices[playerId, t] not in r.collisions[:, t] for t in range(self.horizon)])

    def getPulls(self, playerId, environmentId):
        return self.pulls[environmentId][playerId, :] / float(self.repetitions)

    def getBestArmPulls(self, playerId, environmentId):
        # We have to divide by a arange() = cumsum(ones) to get a frequency
        return self.BestArmPulls[environmentId][playerId, :] / (float(self.repetitions) * np.arange(1, 1 + self.horizon))

    def getFreeTransmissions(self, playerId, environmentId):
        return self.FreeTransmissions[environmentId][playerId, :] / float(self.repetitions)

    def getFrequencyCollisions(self, armId, environmentId):
        return self.collisions[environmentId][armId, :] / float(self.repetitions)

    def getReward(self, playerId, environmentId):
        return self.rewards[environmentId][playerId, :] / float(self.repetitions)

    def getRegret(self, playerId, environmentId):
        return np.arange(1, 1 + self.horizon) * self.envs[environmentId].maxArm - self.getReward(playerId, environmentId)

    def getCentralizedRegret(self, environmentId):
        meansBestArms = np.sort(np.array([arm.mean() for arm in self.envs[environmentId].arms]))[-self.nbPlayers:]
        averageBestRewards = np.arange(1, 1 + self.horizon) * np.sum(meansBestArms)
        actualRewards = sum([self.getReward(playerId, environmentId) for playerId in range(self.nbPlayers)])
        return averageBestRewards - actualRewards

    # Plotting decentralized (vectorial) rewards
    def plotRegrets(self, environmentId, savefig=None, semilogx=False):
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
        plt.ylabel(r"Cumulative Regret $R_t$ (personal, not centralized)")
        plt.title("Multi-players ({}): personal regret for each player, averaged ${}$ times\nArms: ${}${}".format(self.collisionModel.__name__, self.repetitions, repr(self.envs[environmentId].arms), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI)
        plt.show()

    # Plotting centralized rewards (sum)
    def plotRegretsCentralized(self, environmentId, savefig=None, semilogx=False):
        Y = np.zeros(self.horizon)
        Y = self.getCentralizedRegret(environmentId)
        # Start the figure
        plt.figure()
        if semilogx:
            plt.semilogx(Y)
        else:
            plt.plot(Y)
        plt.grid()
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}$".format(self.horizon))
        plt.ylabel(r"Cumulative Centralized Regret $R_t$")
        plt.title("Multi-players ({}): cumulated regret from each player, averaged ${}$ times\nArms: ${}${}".format(self.collisionModel.__name__, self.repetitions, repr(self.envs[environmentId].arms), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI)
        plt.show()

    def plotBestArmPulls(self, environmentId, savefig=None):
        plt.figure()
        for i, player in enumerate(self.players):
            Y = self.getBestArmPulls(i, environmentId)
            plt.plot(Y, label=str(player))
        plt.legend(loc='lower right')
        plt.grid()
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}$".format(self.horizon))
        plt.ylim(-0.03, 1.03)
        plt.ylabel(r"Frequency of pulls of the optimal arm")
        plt.title("Multi-players ({}): best arm pulls frequency for each players, averaged ${}$ times\nArms: ${}${}".format(self.collisionModel.__name__, self.cfg['repetitions'], repr(self.envs[environmentId].arms), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI)
        plt.show()

    def plotFreeTransmissions(self, environmentId, savefig=None):
        plt.figure()
        for i, player in enumerate(self.players):
            Y = self.getFreeTransmissions(i, environmentId)
            plt.plot(Y, '.', label=str(player))
            # TODO should only plot with markers
        plt.legend(loc='lower right')
        plt.grid()
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}$".format(self.horizon))
        plt.ylim(-0.03, 1.03)
        plt.ylabel(r"Transmission on a free channel")
        plt.title("Multi-players ({}): free transmission for each players, averaged ${}$ times\nArms: ${}${}".format(self.collisionModel.__name__, self.cfg['repetitions'], repr(self.envs[environmentId].arms), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI)
        plt.show()

    def plotFrequencyCollisions(self, environmentId, savefig=None, piechart=True):
        nbArms = self.envs[environmentId].nbArms
        Y = np.zeros(1 + nbArms)
        labels = [''] * (1 + nbArms)
        # All the other arms
        for armId, arm in enumerate(self.envs[environmentId].arms):
            # Y[armId] = np.sum(self.getFrequencyCollisions(armId, environmentId) >= 1)
            Y[armId] = np.sum(self.getFrequencyCollisions(armId, environmentId))
            labels[armId] = '#${}$: {}'.format(armId + 1, repr(arm))
        for armId, arm in enumerate(self.envs[environmentId].arms):
            print("  - For {},\tfrequency of collisions is {:.3f}  ...".format(labels[armId], Y[armId] / self.horizon))
        if np.isclose(np.sum(Y), 0):
            print("==> No collisions to plot ... Stopping now  ...")
            return
        Y /= self.horizon
        # FIXME how could the np.sum(Y) not be < 1 ???
        Y[-1] = 1 - np.sum(Y) if np.sum(Y) < 1 else 0
        # Special arm: no collision
        labels[-1] = 'No collision'
        # Start the figure
        plt.figure()
        if piechart:
            xlabel = ', '.join(str(player) for player in self.players)
            print("Using xlabel =", xlabel)  # DEBUG
            plt.xlabel(xlabel)
            plt.axis('equal')
            plt.pie(Y, labels=labels, colors=colors[:len(labels)], explode=[0.05] * len(Y))
        else:
            # Y /= np.sum(Y)  # XXX Should we feed a normalized vector to plt.pie or plt.hist ?
            plt.hist(Y, bins=len(Y))
            # XXX if this is not enough, do the histogram/bar plot manually, and add labels as texts
        plt.legend(loc='lower right')
        plt.title("Multi-players ({}): Frequency of collision for each arm, averaged ${}$ times\nArms: ${}${}".format(self.collisionModel.__name__, self.cfg['repetitions'], repr(self.envs[environmentId].arms), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI)
        plt.show()

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

    rewards = np.zeros(nbPlayers)
    choices = np.zeros(nbPlayers, dtype=int)
    pulls = np.zeros((nbPlayers, nbArms), dtype=int)
    collisions = np.zeros(nbArms, dtype=int)
    for t in range(horizon):
        # Reset the array, faster than reallocating them!
        rewards.fill(0)
        pulls.fill(0)
        collisions.fill(0)
        # Every player decides which arm to pull
        for i, player in enumerate(players):
            choices[i] = player.choice()
            # print(" Round t = \t{}, player \t#{}/{} ({}) \tchose : {} ...".format(t, i + 1, len(players), player, choices[i]))  # DEBUG
        # Then we decide if there is collisions and what to do why them
        collisionModel(t, env.arms, players, choices, rewards, pulls, collisions)
        # FIXME? Do not store the choices as good choices if they gave a collision
        for i, player in enumerate(players):
            if collisions[choices[i]] > 1:
                choices[i] = -1
        # Finally we store the results
        result.store(t, choices, rewards, pulls, collisions)
    return result


