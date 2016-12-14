# -*- coding: utf-8 -*-
""" Evaluator class to wrap and run the simulations, for the multi-players case. """
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.1"

# Generic imports
from copy import deepcopy
from textwrap import wrap
from re import search
import random
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
from .plotsettings import DPI, signature, maximizeWindow, palette, makemarkers
from .ResultMultiPlayers import ResultMultiPlayers
from .MAB import MAB
from .CollisionModels import defaultCollisionModel


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
        self.rewardsSquared = dict()
        self.pulls = dict()
        self.collisions = dict()
        self.BestArmPulls = dict()
        self.FreeTransmissions = dict()
        print("Number of environments to try:", len(self.envs))
        for envId in range(len(self.envs)):  # Zeros everywhere
            self.rewards[envId] = np.zeros((self.nbPlayers, self.horizon))
            self.rewardsSquared[envId] = np.zeros((self.nbPlayers, self.horizon))
            self.pulls[envId] = np.zeros((self.nbPlayers, self.envs[envId].nbArms))
            self.collisions[envId] = np.zeros((self.envs[envId].nbArms, self.horizon))
            self.BestArmPulls[envId] = np.zeros((self.nbPlayers, self.horizon))
            self.FreeTransmissions[envId] = np.zeros((self.nbPlayers, self.horizon))

    def __initEnvironments__(self):
        for armType in self.cfg['environment']:
            self.envs.append(MAB(armType))

    def __initPlayers__(self, env):
        for playerId, player in enumerate(self.cfg['players']):
            print("- Adding player #{} = {} ...".format(playerId + 1, player))  # DEBUG
            if isinstance(player, dict):  # Either the 'player' is a config dict
                print("  Creating this player from a dictionnary 'player' = {} ...".format(player))  # DEBUG
                self.players.append(player['archtype'](env.nbArms, **player['params']))
            else:  # Or already a player object
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
            seeds = np.random.randint(low=0, high=100 * self.repetitions, size=self.repetitions)
            results = joblib.Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
                joblib.delayed(delayed_play)(env, self.players, self.horizon, self.collisionModel, seed=seeds[i])
                for i in range(self.repetitions)
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
            self.rewardsSquared[envId] += np.cumsum(r.rewardsSquared, axis=1)
            self.pulls[envId] += r.pulls
            self.collisions[envId] += r.collisions
            for playerId in range(self.nbPlayers):
                self.BestArmPulls[envId][playerId, :] += np.cumsum(np.in1d(r.choices[playerId, :], index_bestarm))
                # FIXME there is probably a bug in this computation
                self.FreeTransmissions[envId][playerId, :] += np.array([r.choices[playerId, t] not in r.collisions[:, t] for t in range(self.horizon)])

    def getPulls(self, playerId, environmentId=0):
        return self.pulls[environmentId][playerId, :] / float(self.repetitions)

    def getBestArmPulls(self, playerId, environmentId=0):
        # We have to divide by a arange() = cumsum(ones) to get a frequency
        return self.BestArmPulls[environmentId][playerId, :] / (float(self.repetitions) * np.arange(1, 1 + self.horizon))

    def getFreeTransmissions(self, playerId, environmentId=0):
        return self.FreeTransmissions[environmentId][playerId, :] / float(self.repetitions)

    def getCollisions(self, armId, environmentId=0):
        return self.collisions[environmentId][armId, :] / float(self.repetitions)

    def getRewards(self, playerId, environmentId=0):
        return self.rewards[environmentId][playerId, :] / float(self.repetitions)

    def getRegretMean(self, playerId, environmentId=0):
        """ Warning: this is the centralized regret, for one arm, it does not make much sense in the multi-players setting!
        """
        return np.arange(1, 1 + self.horizon) * self.envs[environmentId].maxArm - self.getRewards(playerId, environmentId)

    def getCentralizedRegret(self, environmentId=0):
        meansArms = np.sort(np.array([arm.mean() for arm in self.envs[environmentId].arms]))
        meansBestArms = meansArms[-self.nbPlayers:]
        sumBestMeans = np.sum(meansBestArms)
        # FIXED how to count it when there is more players than arms ?
        # FIXME it depend on the collision model !
        if self.envs[environmentId].nbArms < self.nbPlayers:
            # sure to have collisions, then the best strategy is to put all the collisions in the worse arm
            worseArm = np.min(meansArms)
            sumBestMeans -= worseArm  # This count the collisions
        averageBestRewards = np.arange(1, 1 + self.horizon) * sumBestMeans
        # And for the actual rewards, the collisions are counted in the rewards logged in self.getRewards
        actualRewards = sum(self.getRewards(playerId, environmentId) for playerId in range(self.nbPlayers))
        return averageBestRewards - actualRewards

    # Plotting decentralized (vectorial) rewards
    def plotRewards(self, environmentId=0, savefig=None, semilogx=False):
        plt.figure()
        ymin = 0
        colors = palette(self.nbPlayers)
        markers = makemarkers(self.nbPlayers)
        markers_on = np.arange(0, self.horizon, int(self.horizon / 10.0))
        delta_marker = 1 + int(self.horizon / 200.0)  # XXX put back 0 if needed
        for i, player in enumerate(self.players):
            label = 'Player #{}: {}'.format(i + 1, str(player))
            Y = self.getRewards(i, environmentId)
            ymin = min(ymin, np.min(Y))  # XXX Should be smarter
            if semilogx:
                plt.semilogx(Y, label=label, color=colors[i], marker=markers[i], markevery=(delta_marker * (i % self.envs[environmentId].nbArms) + markers_on))
            else:
                plt.plot(Y, label=label, color=colors[i], marker=markers[i], markevery=(delta_marker * (i % self.envs[environmentId].nbArms) + markers_on))
        # TODO add std
        plt.legend(loc='upper left', numpoints=1, fancybox=True, framealpha=0.8)  # http://matplotlib.org/users/recipes.html#transparent-fancy-legends
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}$".format(self.horizon))
        ymax = max(plt.ylim()[1], 1)
        plt.ylim(ymin, ymax)
        plt.ylabel("Cumulative personal reward $r_t$ (not centralized)")
        plt.title("Multi-players $M = {}$ (collision model: {}): personal reward for each player, averaged ${}$ times\nArms: ${}${}".format(self.nbPlayers, self.collisionModel.__name__, self.repetitions, self.envs[environmentId].reprarms(self.nbPlayers), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI, bbox_inches='tight')
        plt.show()

    # Plotting centralized rewards (sum)
    def plotRegretsCentralized(self, environmentId=0, savefig=None, semilogx=False, normalizedRegret=False):
        X = np.arange(self.horizon)
        Y = self.getCentralizedRegret(environmentId)
        if normalizedRegret:
            Y /= np.log(2 + X)   # XXX prevent /0
        meanY = np.mean(Y)
        plt.figure()
        if semilogx:
            plt.semilogx(Y, label="Cumulated centralized regret")
            # We plot a horizontal line ----- at the best arm mean
            plt.semilogx(X, meanY * np.ones_like(X), 'k--', label="Mean cumulated centralized regret = ${:.3g}$".format(meanY))
        else:
            plt.plot(Y, label="Cumulated centralized regret")
            # We plot a horizontal line ----- at the best arm mean
            plt.plot(X, meanY * np.ones_like(X), 'k--', label="Mean cumulated centralized regret = ${:.3g}$".format(meanY))
        # TODO add std
        lowerbound, anandkumar_lowerbound = self.envs[environmentId].lowerbound_multiplayers(self.nbPlayers)
        # We also plot our lower bound
        if normalizedRegret:
            plt.plot(lowerbound * np.ones_like(X), 'k-', label="Kaufmann & Besson lower bound", lw=3)
            plt.plot(anandkumar_lowerbound * np.ones_like(X), 'k:', label="Anandkumar lower bound", lw=3)
        else:
            plt.plot(lowerbound * np.log(2 + X), 'k-', label="Kaufmann & Besson lower bound", lw=3)
            plt.plot(anandkumar_lowerbound * np.log(2 + X), 'k:', label="Anandkumar lower bound", lw=3)
        # Labels and legends
        plt.legend(loc='lower right', numpoints=1, fancybox=True, framealpha=0.8)  # http://matplotlib.org/users/recipes.html#transparent-fancy-legends
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}$\n{}".format(self.horizon, self.strPlayers()))
        plt.ylabel("{}Cumulative Centralized Regret $R_t$".format("Normalized " if normalizedRegret else " "))
        plt.title("Multi-players $M = {}$ (collision model: {}): {}cumulated centralized regret, averaged ${}$ times\nArms: ${}${}".format(self.nbPlayers, self.collisionModel.__name__, "normalized " if normalizedRegret else " ", self.repetitions, self.envs[environmentId].reprarms(self.nbPlayers), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI, bbox_inches='tight')
        plt.show()

    def plotBestArmPulls(self, environmentId=0, savefig=None):
        plt.figure()
        colors = palette(self.nbPlayers)
        for i, player in enumerate(self.players):
            Y = self.getBestArmPulls(i, environmentId)
            plt.plot(Y, label=str(player), color=colors[i])
        plt.legend(loc='upper right', numpoints=1, fancybox=True, framealpha=0.8)  # http://matplotlib.org/users/recipes.html#transparent-fancy-legends
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}$".format(self.horizon))
        plt.ylim(-0.03, 1.03)
        plt.ylabel("Frequency of pulls of the optimal arm")
        plt.title("Multi-players $M = {}$ (collision model: {}): best arm pulls frequency for each players, averaged ${}$ times\nArms: ${}${}".format(self.nbPlayers, self.collisionModel.__name__, self.cfg['repetitions'], self.envs[environmentId].reprarms(self.nbPlayers), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI, bbox_inches='tight')
        plt.show()

    def plotFreeTransmissions(self, environmentId=0, savefig=None):
        plt.figure()
        colors = palette(self.nbPlayers)
        for i, player in enumerate(self.players):
            Y = self.getFreeTransmissions(i, environmentId)
            plt.plot(Y, '.', label=str(player), color=colors[i], linewidth=1, markersize=1)
            # TODO should only plot with markers
        plt.legend(loc='upper left', numpoints=1, fancybox=True, framealpha=0.8)  # http://matplotlib.org/users/recipes.html#transparent-fancy-legends
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}$".format(self.horizon))
        plt.ylim(-0.03, 1.03)
        plt.ylabel("Transmission on a free channel")
        plt.title("Multi-players $M = {}$ (collision model: {}): free transmission for each players, averaged ${}$ times\nArms: ${}${}".format(self.nbPlayers, self.collisionModel.__name__, self.cfg['repetitions'], self.envs[environmentId].reprarms(self.nbPlayers), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI, bbox_inches='tight')
        plt.show()

    # TODO I should plot the evolution of the occupation ratio of each channel, as a function of time
    # Starting from the average occupation (by primary users), as given by [1 - arm.mean()], it should increase occupation[arm] when users chose it
    # The reason/idea is that good arms (low occupation ration) are pulled a lot, thus becoming not as available as they seemed

    def plotNbCollisions(self, environmentId=0, savefig=None, cumsum=False):
        Y = np.zeros(self.horizon)
        for armId in range(self.envs[environmentId].nbArms):
            # Y += (self.getCollisions(armId, environmentId) >= 1)
            Y += self.getCollisions(armId, environmentId)
        plt.figure()
        if cumsum:
            Y = np.cumsum(Y)
        else:
            plt.ylim([-0.03, 1.03])
        Y /= (self.nbPlayers)  # XXX To normalized the count?
        # Start the figure
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}$\n{}".format(self.horizon, self.strPlayers()))
        plt.ylabel("{} of collisions".format("Cumulated number" if cumsum else "Frequency"))
        plt.plot(Y, '-' if cumsum else '.')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8)  # http://matplotlib.org/users/recipes.html#transparent-fancy-legends
        plt.title("Multi-players $M = {}$ (collision model: {}):{} of collisions, averaged ${}$ times\nArms: ${}${}".format(self.nbPlayers, self.collisionModel.__name__, " cumulated number" if cumsum else " frequency", self.cfg['repetitions'], self.envs[environmentId].reprarms(self.nbPlayers), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI, bbox_inches='tight')
        plt.show()

    def plotFrequencyCollisions(self, environmentId=0, savefig=None, piechart=True):
        nbArms = self.envs[environmentId].nbArms
        Y = np.zeros(1 + nbArms)
        labels = [''] * (1 + nbArms)  # Empty labels
        colors = palette(1 + nbArms)  # Get colors
        # All the other arms
        for armId, arm in enumerate(self.envs[environmentId].arms):
            # Y[armId] = np.sum(self.getCollisions(armId, environmentId) >= 1)  # XXX no, we should not count just the fact that there were collisions, but instead count all collisions
            Y[armId] = np.sum(self.getCollisions(armId, environmentId))
            labels[armId] = "#${}$: ${}$".format(armId, repr(arm))
        Y /= (self.horizon * self.nbPlayers)
        assert 0 <= np.sum(Y) <= 1, "Error: the sum of collisions = {}, averaged by horizon and nbPlayers, cannot be outside of [0, 1] ...".format(np.sum(Y))
        for armId, arm in enumerate(self.envs[environmentId].arms):
            print("  - For {},\tfrequency of collisions is {:g}  ...".format(labels[armId], Y[armId]))
            if Y[armId] < 1e-4:  # Do not display small slices
                labels[armId] = ''
        if np.isclose(np.sum(Y), 0):
            print("==> No collisions to plot ... Stopping now  ...")
            # return  # XXX
        # Special arm: no collision
        Y[-1] = 1 - np.sum(Y) if np.sum(Y) < 1 else 0
        labels[-1] = "No collision (${:.1%}$$\%$)".format(Y[-1]) if Y[-1] > 1e-4 else ''
        colors[-1] = 'lightgrey'
        # Start the figure
        plt.figure()
        if piechart:
            plt.xlabel(self.strPlayers())  # DONE split this in new lines if it is too long!
            plt.axis('equal')
            plt.pie(Y, labels=labels, colors=colors, explode=[0.07] * len(Y), startangle=45)
            # , autopct='%.4g%%'
        else:  # TODO do an histogram instead of this piechart
            plt.hist(Y, bins=len(Y), colors=colors)
            # XXX if this is not enough, do the histogram/bar plot manually, and add labels as texts
        plt.legend(loc='center right', fancybox=True, framealpha=0.8)  # http://matplotlib.org/users/recipes.html#transparent-fancy-legends
        plt.title("Multi-players $M = {}$ (collision model: {}): Frequency of collision for each arm, averaged ${}$ times\nArms: ${}${}".format(self.nbPlayers, self.collisionModel.__name__, self.cfg['repetitions'], self.envs[environmentId].reprarms(self.nbPlayers), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI, bbox_inches='tight')
        plt.show()

    def printFinalRanking(self, environmentId=0):
        assert 0 < self.averageOn < 1, "Error, the parameter averageOn of a EvaluatorMultiPlayers classs has to be in (0, 1) strictly, but is = {} here ...".format(self.averageOn)
        print("\nFinal ranking for this environment #{} :".format(environmentId))
        lastY = np.zeros(self.nbPlayers)
        for i, player in enumerate(self.players):
            Y = self.getRewards(i, environmentId)
            if self.finalRanksOnAverage:
                lastY[i] = np.mean(Y[-int(self.averageOn * self.horizon)])   # get average value during the last averageOn% of the iterations
            else:
                lastY[i] = Y[-1]  # get the last value
        # Sort lastY and give ranking
        index_of_sorting = np.argsort(-lastY)  # Get them by INCREASING rewards, not decreasing regrets
        for i, k in enumerate(index_of_sorting):
            player = self.players[k]
            print("- Player #{}, '{}'\twas ranked\t{} / {} for this simulation (last rewards = {:g}).".format(k + 1, str(player), i + 1, self.nbPlayers, lastY[k]))
        return lastY, index_of_sorting

    def strPlayers(self, width=130):
        listStrPlayers = [extract(str(player)) for player in self.players]
        if len(set(listStrPlayers)) == 1:  # Unique user
            text = '{} x {}'.format(self.nbPlayers, listStrPlayers[0])
        else:
            text = ', '.join(listStrPlayers)
        text = '\n'.join(wrap(text, width=width))
        text = '{} players: {}'.format(self.nbPlayers, text)
        return text


# Helper function for the parallelization
# @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof
def delayed_play(env, players, horizon, collisionModel, seed=None):
    # XXX Try to give a unique seed to random & numpy.random for each call of this function
    random.seed(seed)
    np.random.seed(seed)
    # We have to deepcopy because this function is Parallel-ized
    env = deepcopy(env)
    nbArms = env.nbArms
    players = deepcopy(players)
    horizon = deepcopy(horizon)
    nbPlayers = len(players)
    # random_arm_orders = [np.random.permutation(nbArms) for i in range(nbPlayers)]
    # Start game
    for player in players:
        player.startGame()
    # Store results
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
            # FIXME here, the environment should apply a random permutation to each player, in order for the non-modified UCB-like algorithms to work fine in case of collisions (their initial exploration phase is non-random hence leading to only collisions in the first steps, and ruining the performance)
            # choices[i] = random_arm_orders[i][player.choice()]  # FIXME try it!
            choices[i] = player.choice()
            # print(" Round t = \t{}, player \t#{}/{} ({}) \tchose : {} ...".format(t, i + 1, len(players), player, choices[i]))  # DEBUG
        # Then we decide if there is collisions and what to do why them
        # XXX It is here that the player may receive a reward, if there is no collisions
        collisionModel(t, env.arms, players, choices, rewards, pulls, collisions)
        # Finally we store the results
        result.store(t, choices, rewards, pulls, collisions)

    # Prints the ranks
    ranks = [player.rank if hasattr(player, 'rank') else None for player in players]
    if len(set(ranks)) != nbPlayers:
        for (player, rank) in zip(players, ranks):
            if rank:
                print(" - End of one game, rhoRand player {} had rank {} ...".format(player, rank))
    else:
        if set(ranks) != {None}:
            print(" - End of one game, rhoRand found orthogonal ranks: ranks = {} ...".format(ranks))
    return result


def extract(text):
    """ Extract the str of a player, if it is a child, printed as '#[0-9]+<...>' --> ... """
    m = search('<[^>]+>', text).group(0)
    if m[0] == '<' and m[-1] == '>':
        return m[1:-1]
    else:
        return text
