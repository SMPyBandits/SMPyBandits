# -*- coding: utf-8 -*-
""" EvaluatorMultiPlayers class to wrap and run the simulations, for the multi-players case.
Lots of plotting methods, to have various visualizations.
"""
from __future__ import print_function, division

__author__ = "Lilian Besson"
__version__ = "0.5"

# Generic imports
from copy import deepcopy
from re import search
import random
# Scientific imports
import numpy as np
import matplotlib.pyplot as plt
# import h5py
# Local imports, libraries
from .usejoblib import USE_JOBLIB, Parallel, delayed
from .usetqdm import USE_TQDM, tqdm
# Local imports, tools and config
from .plotsettings import BBOX_INCHES, signature, maximizeWindow, palette, makemarkers, add_percent_formatter, wraptext, wraplatex, legend, show_and_save
from .sortedDistance import weightedDistance, manhattan, kendalltau, spearmanr, gestalt, meanDistance, sortedDistance
from .fairnessMeasures import amplitude_fairness, std_fairness, rajjain_fairness, mean_fairness, fairnessMeasure, fairness_mapping
# Local imports, objects and functions
from .CollisionModels import onlyUniqUserGetsReward, noCollision, closerUserGetsReward, rewardIsSharedUniformly, defaultCollisionModel, full_lost_if_collision
from .MAB import MAB, MarkovianMAB, DynamicMAB
from .ResultMultiPlayers import ResultMultiPlayers

REPETITIONS = 1  #: Default nb of repetitions
DELTA_T_SAVE = 1  #: Default sampling rate for saving
DELTA_T_PLOT = 50  #: Default sampling rate for plotting


# --- Class EvaluatorMultiPlayers

class EvaluatorMultiPlayers(object):
    """ Evaluator class to run the simulations, for the multi-players case.
    """

    def __init__(self, configuration):
        # Configuration
        self.cfg = configuration  #: Configuration dictionnary
        # Attributes
        self.nbPlayers = len(self.cfg['players'])  #: Number of policies
        print("Number of players in the multi-players game:", self.nbPlayers)
        self.horizon = self.cfg['horizon']  #: Horizon (number of time steps)
        print("Time horizon:", self.horizon)
        self.repetitions = self.cfg.get('repetitions', REPETITIONS)  #: Number of repetitions
        print("Number of repetitions:", self.repetitions)
        self.delta_t_save = self.cfg.get('delta_t_save', DELTA_T_SAVE)
        print("Sampling rate for saving, delta_t_save:", self.delta_t_save)  #: Sampling rate for saving
        self.delta_t_plot = 1 if self.horizon <= 10000 else self.cfg.get('delta_t_plot', DELTA_T_PLOT)
        print("Sampling rate for plotting, delta_t_plot:", self.delta_t_plot)  #: Sampling rate for plotting
        self.duration = int(self.horizon / self.delta_t_save)
        print("Number of jobs for parallelization:", self.cfg['n_jobs'])
        self.collisionModel = self.cfg.get('collisionModel', defaultCollisionModel)  #: Which collision model should be used
        self.full_lost_if_collision = full_lost_if_collision.get(self.collisionModel.__name__, True)  #: Is there a full loss of rewards if collision ? To compute the correct decomposition of regret
        print("Using collision model {} (function {}).\nMore details:\n{}".format(self.collisionModel.__name__, self.collisionModel, self.collisionModel.__doc__))
        # Flags
        self.finalRanksOnAverage = self.cfg.get('finalRanksOnAverage', True)  #: Final display of ranks are done on average rewards?
        self.averageOn = self.cfg.get('averageOn', 5e-3)  #: How many last steps for final rank average rewards
        self.useJoblib = USE_JOBLIB and self.cfg['n_jobs'] != 1  #: Use joblib to parallelize for loop on repetitions (useful)
        self.showplot = self.cfg.get('showplot', True)  #: Show the plot (interactive display or not)
        # Internal object memory
        self.envs = []  #: List of environments
        self.players = []  #: List of policies
        self.__initEnvironments__()
        # Internal vectorial memory
        self.rewards = dict()  #: For each env, history of rewards
        # self.rewardsSquared = dict()
        self.pulls = dict()  #: For each env, keep the history of best arm pulls
        self.allPulls = dict()  #: For each env, keep the full history of best arm pulls
        self.collisions = dict()  #: For each env, keep the history of collisions on all arms
        self.NbSwitchs = dict()  #: For each env, keep the history of switches (change of configuration of players)
        self.BestArmPulls = dict()  #: For each env, keep the history of best arm pulls
        self.FreeTransmissions = dict()  #: For each env, keep the history of succesful transmission (1 - collisions, basically)
        print("Number of environments to try:", len(self.envs))  # DEBUG
        for envId in range(len(self.envs)):  # Zeros everywhere
            self.rewards[envId] = np.zeros((self.nbPlayers, self.duration))
            # self.rewardsSquared[envId] = np.zeros((self.nbPlayers, self.duration))
            self.pulls[envId] = np.zeros((self.nbPlayers, self.envs[envId].nbArms))
            self.allPulls[envId] = np.zeros((self.nbPlayers, self.envs[envId].nbArms, self.duration))
            self.collisions[envId] = np.zeros((self.envs[envId].nbArms, self.duration))
            self.NbSwitchs[envId] = np.zeros((self.nbPlayers, self.duration))
            self.BestArmPulls[envId] = np.zeros((self.nbPlayers, self.duration))
            self.FreeTransmissions[envId] = np.zeros((self.nbPlayers, self.duration))
        # To speed up plotting
        self.times = np.arange(1, 1 + self.horizon, self.delta_t_save)

    # --- Init methods

    def __initEnvironments__(self):
        """ Create environments."""
        nbArms = []
        for configuration_arms in self.cfg['environment']:
            if isinstance(configuration_arms, dict) \
               and "arm_type" in configuration_arms and "params" in configuration_arms \
               and "function" in configuration_arms["params"] and "args" in configuration_arms["params"]:
                MB = DynamicMAB(configuration_arms)
            elif isinstance(configuration_arms, dict) \
               and "arm_type" in configuration_arms and configuration_arms["arm_type"] == "Markovian" \
               and "params" in configuration_arms \
               and "transitions" in configuration_arms["params"]:
                self.envs.append(MarkovianMAB(configuration_arms))
            else:
                MB = MAB(configuration_arms)
            self.envs.append(MB)
            nbArms.append(MB.nbArms)
        if len(set(nbArms)) != 1:  # FIXME
            raise ValueError("ERROR: right now, the multi-environments evaluator does not work well for MP policies, if there is a number different of arms in the scenarios. FIXME correct this point!")

    def __initPlayers__(self, env):
        """ Create or initialize policies."""
        for playerId, player in enumerate(self.cfg['players']):
            print("- Adding player #{} = {} ...".format(playerId + 1, player))  # DEBUG
            if isinstance(player, dict):  # Either the 'player' is a config dict
                print("  Creating this player from a dictionnary 'player' = {} ...".format(player))  # DEBUG
                self.players.append(player['archtype'](env.nbArms, **player['params']))
            else:  # Or already a player object
                print("  Using this already created player 'player' = {} ...".format(player))  # DEBUG
                self.players.append(player)

    # --- Start computation

    def startAllEnv(self):
        """Simulate all envs."""
        for envId, env in enumerate(self.envs):
            self.startOneEnv(envId, env)

    def startOneEnv(self, envId, env):
        """Simulate that env."""
        print("\nEvaluating environment:", repr(env))  # DEBUG
        self.players = []
        self.__initPlayers__(env)
        # Get the position of the best arms
        means = env.means
        bestarm = env.maxArm
        index_bestarm = np.nonzero(np.isclose(means, bestarm))[0]

        def store(r):
            """Store the result of the experiment r."""
            self.rewards[envId] += np.cumsum(r.rewards, axis=1)
            # self.rewardsSquared[envId] += np.cumsum(r.rewards ** 2, axis=1)
            # self.rewardsSquared[envId] += np.cumsum(r.rewardsSquared, axis=1)
            self.pulls[envId] += r.pulls
            self.allPulls[envId] += r.allPulls
            self.collisions[envId] += r.collisions
            for playerId in range(self.nbPlayers):
                self.NbSwitchs[envId][playerId, 1:] += (np.diff(r.choices[playerId, :]) != 0)
                self.BestArmPulls[envId][playerId, :] += np.cumsum(np.in1d(r.choices[playerId, :], index_bestarm))
                # FIXME there is probably a bug in this computation
                self.FreeTransmissions[envId][playerId, :] += np.array([r.choices[playerId, t] not in r.collisions[:, t] for t in range(self.duration)])

        # Start now
        if self.useJoblib:
            seeds = np.random.randint(low=0, high=100 * self.repetitions, size=self.repetitions)
            for r in Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
                delayed(delayed_play)(env, self.players, self.horizon, self.collisionModel, delta_t_save=self.delta_t_save, seed=seeds[repeatId], repeatId=repeatId)
                for repeatId in tqdm(range(self.repetitions), desc="Repeat||")
            ):
                store(r)
        else:
            for repeatId in tqdm(range(self.repetitions), desc="Repeat"):
                r = delayed_play(env, self.players, self.horizon, self.collisionModel, delta_t_save=self.delta_t_save, repeatId=repeatId)
                store(r)

    # --- Getter methods

    def getPulls(self, playerId, envId=0):
        """Extract mean pulls."""
        return self.pulls[envId][playerId, :] / float(self.repetitions)

    def getAllPulls(self, playerId, armId, envId=0):
        """Extract mean of all pulls."""
        return self.allPulls[envId][playerId, armId, :] / float(self.repetitions)

    def getNbSwitchs(self, playerId, envId=0):
        """Extract mean nb of switches."""
        return self.NbSwitchs[envId][playerId, :] / float(self.repetitions)

    def getCentralizedNbSwitchs(self, envId=0):
        """Extract average of mean nb of switches."""
        return np.sum(self.NbSwitchs[envId], axis=0) / (float(self.repetitions) * self.nbPlayers)

    def getBestArmPulls(self, playerId, envId=0):
        """Extract mean of best arms pulls."""
        # We have to divide by a arange() = cumsum(ones) to get a frequency
        return self.BestArmPulls[envId][playerId, :] / (float(self.repetitions) * self.times)

    def getFreeTransmissions(self, playerId, envId=0):
        """Extract mean of succesful transmission."""
        return self.FreeTransmissions[envId][playerId, :] / float(self.repetitions)

    def getCollisions(self, armId, envId=0):
        """Extract mean of number of collisions."""
        return self.collisions[envId][armId, :] / float(self.repetitions)

    def getRewards(self, playerId, envId=0):
        """Extract mean of rewards."""
        return self.rewards[envId][playerId, :] / float(self.repetitions)

    def getRegretMean(self, playerId, envId=0):
        """Extract mean of regret

        - Warning: this is the centralized regret, for one arm, it does not make much sense in the multi-players setting!
        """
        return (self.times - 1) * self.envs[envId].maxArm - self.getRewards(playerId, envId)

    def getCentralizedRegret(self, envId=0):
        """Compute the empirical centralized regret: cumsum on time of the mean rewards of the M best arms - cumsum on time of the empirical rewards obtained by the players."""
        meansArms = np.sort(self.envs[envId].means)
        meansBestArms = meansArms[-self.nbPlayers:]
        sumBestMeans = np.sum(meansBestArms)
        # FIXED how to count it when there is more players than arms ?
        # FIXME it depends on the collision model !
        if self.envs[envId].nbArms < self.nbPlayers:
            # sure to have collisions, then the best strategy is to put all the collisions in the worse arm
            worseArm = np.min(meansArms)
            sumBestMeans -= worseArm  # This count the collisions
        averageBestRewards = self.times * sumBestMeans
        # And for the actual rewards, the collisions are counted in the rewards logged in self.getRewards
        actualRewards = sum(self.getRewards(playerId, envId) for playerId in range(self.nbPlayers))
        return averageBestRewards - actualRewards

    # --- Three terms in the regret

    def getFirstRegretTerm(self, envId=0):
        """Extract and compute the first term in the centralized regret: losses due to pulling suboptimal arms."""
        means = self.envs[envId].means
        sortingIndex = np.argsort(means)
        means = np.sort(means)
        deltaMeansWorstArms = means[-self.nbPlayers] - means[:-self.nbPlayers]
        allPulls = self.allPulls[envId] / float(self.repetitions)  # Shape: (nbPlayers, nbArms, duration)
        allWorstPulls = allPulls[:, sortingIndex[:-self.nbPlayers], :]
        worstPulls = np.sum(allWorstPulls, axis=0)  # sum for all players
        losses = np.dot(deltaMeansWorstArms, worstPulls)  # Count and sum on k in Mworst
        firstRegretTerm = np.cumsum(losses)  # Accumulate losses
        return firstRegretTerm

    def getSecondRegretTerm(self, envId=0):
        """Extract and compute the second term in the centralized regret: losses due to not pulling optimal arms."""
        means = self.envs[envId].means
        sortingIndex = np.argsort(means)
        means = np.sort(means)
        deltaMeansBestArms = means[-self.nbPlayers:] - means[-self.nbPlayers]
        allPulls = self.allPulls[envId] / float(self.repetitions)  # Shape: (nbPlayers, nbArms, duration)
        allBestPulls = allPulls[:, sortingIndex[-self.nbPlayers:], :]
        bestMisses = 1 - np.sum(allBestPulls, axis=0)  # sum for all players
        losses = np.dot(deltaMeansBestArms, bestMisses)  # Count and sum on k in Mbest
        secondRegretTerm = np.cumsum(losses)  # Accumulate losses
        return secondRegretTerm

    def getThirdRegretTerm(self, envId=0):
        """Extract and compute the third term in the centralized regret: losses due to collisions."""
        means = self.envs[envId].means
        countCollisions = self.collisions[envId]   # Shape: (nbArms, duration)
        if not self.full_lost_if_collision:
            print("Warning: the collision model ({}) does *not* yield a loss in communication when colliding (one user can communicate, or in average one user can communicate), so countCollisions -= 1 for the 3rd regret term ...".format(self.collisionModel.__name__))  # DEBUG
            countCollisions = np.maximum(0, countCollisions - 1)
        losses = np.dot(means, countCollisions / float(self.repetitions))  # Count and sum on k in 1...K
        thirdRegretTerm = np.cumsum(losses)  # Accumulate losses
        return thirdRegretTerm

    # --- Plotting methods

    def plotRewards(self, envId=0, savefig=None, semilogx=False):
        """Plot the decentralized (vectorial) rewards, for each player."""
        fig = plt.figure()
        ymin = 0
        colors = palette(self.nbPlayers)
        markers = makemarkers(self.nbPlayers)
        X = self.times - 1
        cumRewards = np.zeros((self.nbPlayers, self.duration))
        for playerId, player in enumerate(self.players):
            label = 'Player #{}: {}'.format(playerId + 1, _extract(str(player)))
            Y = self.getRewards(playerId, envId)
            cumRewards[playerId, :] = Y
            ymin = min(ymin, np.min(Y))  # XXX Should be smarter
            if semilogx:
                plt.semilogx(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[playerId], marker=markers[playerId], markevery=(playerId / 50., 0.1))
            else:
                plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[playerId], marker=markers[playerId], markevery=(playerId / 50., 0.1))
        legend()
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}${}".format(self.horizon, signature))
        plt.ylabel(r"Cumulative personal reward $\mathbb{E}_{%d}[r_t]$" % self.repetitions)
        plt.title("Multi-players $M = {}$ (collision model: {}):\nPersonal reward for each player, averaged ${}$ times\n{} arms: {}".format(self.nbPlayers, self.collisionModel.__name__, self.repetitions, self.envs[envId].nbArms, self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig)
        return fig

    def plotFairness(self, envId=0, savefig=None, semilogx=False, fairness="default", evaluators=()):
        """Plot a certain measure of "fairness", from these personal rewards, support more than one environments (use evaluators to give a list of other environments)."""
        fig = plt.figure()
        X = self.times - 1
        evaluators = [self] + list(evaluators)  # Default to only [self]
        colors = palette(len(evaluators))
        markers = makemarkers(len(evaluators))
        plot_method = plt.semilogx if semilogx else plt.plot
        # Decide which fairness function to use
        fairnessFunction = fairness_mapping[fairness] if isinstance(fairness, str) else fairness
        fairnessName = fairness if isinstance(fairness, str) else fairness.__name__
        for evaId, eva in enumerate(evaluators):
            label = eva.strPlayers(short=True)
            cumRewards = np.zeros((eva.nbPlayers, eva.duration))
            for playerId, player in enumerate(eva.players):
                cumRewards[playerId, :] = eva.getRewards(playerId, envId)
            # # Print each fairness measure  # DEBUG
            # for fN, fF in fairness_mapping.items():
            #     f = fF(cumRewards)
            #     print("  - {} fairness index is = {} ...".format(fN, f))  # DEBUG
            # Plot only one fairness term
            fairness = fairnessFunction(cumRewards)
            plot_method(X[::self.delta_t_plot][2:], fairness[::self.delta_t_plot][2:], markers[evaId] + '-', label=label, markevery=(evaId / 50., 0.1), color=colors[evaId])
        if len(evaluators) > 1:
            legend()
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}${}{}".format(self.horizon, "\n" + self.strPlayers() if len(evaluators) == 1 else "", signature))
        add_percent_formatter("yaxis", 1.0)
        # plt.ylim(0, 1)
        plt.ylabel("Centralized measure of fairness for cumulative rewards ({})".format(fairnessName.title()))
        plt.title("Multi-players $M = {}$ (collision model: {}):\nCentralized measure of fairness, averaged ${}$ times\n{} arms: {}".format(self.nbPlayers, self.collisionModel.__name__, self.repetitions, self.envs[envId].nbArms, self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig)
        return fig

    def plotRegretCentralized(self, envId=0, savefig=None,
                              semilogx=False, semilogy=False, loglog=False,
                              normalized=False, evaluators=(), subTerms=False):
        """Plot the centralized cumulated regret, support more than one environments (use evaluators to give a list of other environments).

        - The lower bounds are also plotted (Besson & Kaufmann, and Anandkumar et al).
        - The three terms of the regret are also plotting if evaluators = () (that's the default).
        """
        X0 = X = self.times - 1
        fig = plt.figure()
        evaluators = [self] + list(evaluators)  # Default to only [self]
        colors = palette(5 if len(evaluators) == 1 and subTerms else len(evaluators))
        markers = makemarkers(5 if len(evaluators) == 1 and subTerms else len(evaluators))
        plot_method = plt.loglog if loglog else plt.plot
        plot_method = plt.semilogy if semilogy else plot_method
        plot_method = plt.semilogx if semilogx else plot_method
        # Loop
        for evaId, eva in enumerate(evaluators):
            if subTerms:
                Ys = [None] * 3
                labels = [""] * 3
                Ys[0] = eva.getFirstRegretTerm(envId)
                labels[0] = " 1st term: Pulls of {} suboptimal arms (lower-bounded)".format(self.envs[envId].nbArms - self.nbPlayers)
                Ys[1] = eva.getSecondRegretTerm(envId)
                labels[1] = " 2nd term: Non-pulls of {} optimal arms".format(self.nbPlayers)
                Ys[2] = eva.getThirdRegretTerm(envId)
                labels[2] = " 3rd term: Weighted count of collisions"
            Y = eva.getCentralizedRegret(envId)
            label = "{}umulated centralized regret".format("Normalized c" if normalized else "C") if len(evaluators) == 1 else eva.strPlayers(short=True)
            if semilogx or loglog:  # FIXED for semilogx plots, truncate to only show t >= 100
                X, Y = X0[X0 >= 100], Y[X0 >= 100]
                if subTerms:
                    for i in range(len(Ys)):
                            Ys[i] = Ys[i][X0 >= 100]
            if normalized:
                Y /= np.log(2 + X)   # XXX prevent /0
                if subTerms:
                    for i in range(len(Ys)):
                        Ys[i] /= np.log(2 + X)  # XXX prevent /0
            meanY = np.mean(Y)
            # Now plot
            plot_method(X[::self.delta_t_plot], Y[::self.delta_t_plot], (markers[evaId] + '-'), markevery=(evaId / 50., 0.1), label=label, color=colors[evaId])
            if len(evaluators) == 1:
                if not semilogx and not loglog and not semilogy:
                    # We plot a horizontal line ----- at the mean regret
                    plot_method(X[::self.delta_t_plot], meanY * np.ones_like(X)[::self.delta_t_plot], '--', label="Mean cumulated centralized regret", color=colors[evaId])
                # " = ${:.3g}$".format(meanY)
                if subTerms:
                    Ys.append(Ys[0] + Ys[1] + Ys[2])
                    labels.append("Sum of 3 terms ($= R_t$)")
                    # print("Difference between regret and sum of three terms:", Y - np.array(Ys[-1]))  # DEBUG
                    for i, (Y, label) in enumerate(zip(Ys, labels)):
                        plot_method(X[::self.delta_t_plot], Y[::self.delta_t_plot], (markers[i + 1] + '-'), markevery=((i + 1) / 50., 0.1), label=label, color=colors[i + 1])
                        if semilogx or loglog:  # Manual fix for issue https://github.com/Naereen/AlgoBandits/issues/38
                            plt.xscale('log')
                        if semilogy or loglog:  # Manual fix for issue https://github.com/Naereen/AlgoBandits/issues/38
                            plt.yscale('log')
        # We also plot our lower bound
        lowerbound, anandkumar_lowerbound, centralized_lowerbound = self.envs[envId].lowerbound_multiplayers(self.nbPlayers)
        print("\nThis MAB problem has: \n - a [Lai & Robbins] complexity constant C(mu) = {:.3g} for 1-player problem ... \n - a Optimal Arm Identification factor H_OI(mu) = {:.2%} ...".format(self.envs[envId].lowerbound(), self.envs[envId].hoifactor()))  # DEBUG
        print(" - [Anandtharam et al] centralized lower-bound = {:.3g},\n - [Anandkumar et al] decentralized lower-bound = {:.3g}\n - Our better (larger) decentralized lower-bound = {:.3g},".format(centralized_lowerbound, anandkumar_lowerbound, lowerbound))  # DEBUG
        T = np.ones_like(X) if normalized else np.log(2 + X)
        plot_method(X[::self.delta_t_plot], lowerbound * T[::self.delta_t_plot], 'k-', label="Kaufmann & Besson lower bound = ${:.3g}$".format(lowerbound), lw=3)
        plot_method(X[::self.delta_t_plot], anandkumar_lowerbound * T[::self.delta_t_plot], 'k--', label="Anandkumar et al lower bound = ${:.3g}$".format(anandkumar_lowerbound), lw=2)
        plot_method(X[::self.delta_t_plot], centralized_lowerbound * T[::self.delta_t_plot], 'k:', label="Centralized lower bound = ${:.3g}$".format(centralized_lowerbound), lw=1)
        # Labels and legends
        legend()
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}${}{}".format(self.horizon, "\n" + self.strPlayers() if len(evaluators) == 1 else "", signature))
        plt.ylabel("{}umulative centralized regret {}".format("Normalized c" if normalized else "C", r"$\mathbb{E}_{%d}[R_t]$" % self.repetitions))
        plt.title("Multi-players $M = {}$ (collision model: {}):\n{}umulated centralized regret, averaged ${}$ times\n{} arms: {}".format(self.nbPlayers, self.collisionModel.__name__, "Normalized c" if normalized else "C", self.repetitions, self.envs[envId].nbArms, self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig)
        return fig

    def plotNbSwitchs(self, envId=0, savefig=None, semilogx=False, cumulated=False):
        """Plot cumulated number of switchs (to evaluate the switching costs), comparing each player."""
        X = self.times - 1
        fig = plt.figure()
        ymin = 0
        colors = palette(self.nbPlayers)
        markers = makemarkers(self.nbPlayers)
        plot_method = plt.semilogx if semilogx else plt.plot
        for playerId, player in enumerate(self.players):
            label = 'Player #{}: {}'.format(playerId + 1, _extract(str(player)))
            Y = self.getNbSwitchs(playerId, envId)
            if cumulated:
                Y = np.cumsum(Y)
            ymin = min(ymin, np.min(Y))  # XXX Should be smarter
            plot_method(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[playerId], marker=markers[playerId], markevery=(playerId / 50., 0.1), linestyle='-' if cumulated else '')
        legend()
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}${}".format(self.horizon, signature))
        ymax = max(plt.ylim()[1], 1)
        plt.ylim(ymin, ymax)
        if not cumulated:
            add_percent_formatter("yaxis", 1.0)
        plt.ylabel("{} of switches by player".format("Cumulated number" if cumulated else "Frequency"))
        plt.title("Multi-players $M = {}$ (collision model: {}):\n{}umber of switches for each player, averaged ${}$ times\n{} arms: {}".format(self.nbPlayers, self.collisionModel.__name__, "Cumulated n" if cumulated else "N", self.repetitions, self.envs[envId].nbArms, self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig)
        return fig

    def plotNbSwitchsCentralized(self, envId=0, savefig=None, semilogx=False, cumulated=False, evaluators=()):
        """Plot the centralized cumulated number of switchs (to evaluate the switching costs), support more than one environments (use evaluators to give a list of other environments)."""
        X = self.times - 1
        fig = plt.figure()
        ymin = 0
        evaluators = [self] + list(evaluators)  # Default to only [self]
        colors = palette(len(evaluators))
        markers = makemarkers(len(evaluators))
        plot_method = plt.semilogx if semilogx else plt.plot
        for evaId, eva in enumerate(evaluators):
            label = "" if len(evaluators) == 1 else eva.strPlayers(short=True)
            Y = eva.getCentralizedNbSwitchs(envId)
            if cumulated:
                Y = np.cumsum(Y)
            ymin = min(ymin, np.min(Y))  # XXX Should be smarter
            plot_method(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[evaId], marker=markers[evaId], markevery=(evaId / 50., 0.1), linestyle='-' if cumulated else '')
        if len(evaluators) > 1:
            legend()
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}${}{}".format(self.horizon, "\n" + self.strPlayers() if len(evaluators) == 1 else "", signature))
        if not cumulated:
            add_percent_formatter("yaxis", 1.0)
        plt.ylabel("{} of switches by player".format("Cumulated number" if cumulated else "Frequency"))
        plt.title("Multi-players $M = {}$ (collision model: {}):\nCentralized {}number of switches, averaged ${}$ times\n{} arms: {}".format(self.nbPlayers, self.collisionModel.__name__, "cumulated " if cumulated else "", self.repetitions, self.envs[envId].nbArms, self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig)
        return fig

    def plotBestArmPulls(self, envId=0, savefig=None):
        """Plot the frequency of pulls of the best channel.

        - Warning: does not adapt to dynamic settings!
        """
        X = self.times - 1
        fig = plt.figure()
        colors = palette(self.nbPlayers)
        markers = makemarkers(self.nbPlayers)
        for playerId, player in enumerate(self.players):
            label = 'Player #{}: {}'.format(playerId + 1, _extract(str(player)))
            Y = self.getBestArmPulls(playerId, envId)
            plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[playerId], marker=markers[playerId], markevery=(playerId / 50., 0.1))
        legend()
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}${}".format(self.horizon, signature))
        add_percent_formatter("yaxis", 1.0)
        plt.ylabel("Frequency of pulls of the optimal arm")
        plt.title("Multi-players $M = {}$ (collision model: {}):\nBest arm pulls frequency for each players, averaged ${}$ times\n{} arms: {}".format(self.nbPlayers, self.collisionModel.__name__, self.cfg['repetitions'], self.envs[envId].nbArms, self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig)
        return fig

    def plotAllPulls(self, envId=0, savefig=None, cumulated=True, normalized=False):
        """Plot the frequency of use of every channels, one figure for each channel. Not so useful."""
        X = self.times - 1
        mainfig = savefig
        colors = palette(self.nbPlayers)
        markers = makemarkers(self.nbPlayers)
        figs = []
        for armId in range(self.envs[envId].nbArms):
            figs.append(plt.figure())
            for playerId, player in enumerate(self.players):
                Y = self.getAllPulls(playerId, armId, envId)
                if cumulated:
                    Y = np.cumsum(Y)
                if normalized:
                    Y /= 1 + X
                plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=str(player), color=colors[playerId], linestyle='', marker=markers[playerId], markevery=(playerId / 50., 0.1))
            legend()
            plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}${}".format(self.horizon, signature))
            s = ("Normalized " if normalized else "") + ("Cumulated number" if cumulated else "Frequency")
            plt.ylabel("{} of pulls of the arm #{}".format(s, armId + 1))
            plt.title("Multi-players $M = {}$ (collision model: {}):\n{} of pulls of the arm #{} for each players, averaged ${}$ times\n{} arms: {}".format(self.nbPlayers, self.collisionModel.__name__, s.lower(), armId + 1, self.cfg['repetitions'], self.envs[envId].nbArms, self.envs[envId].reprarms(self.nbPlayers, latex=True)))
            maximizeWindow()
            if savefig is not None:
                savefig = mainfig.replace("AllPulls", "AllPulls_Arm{}".format(armId + 1))
                print("Saving to", savefig, "...")  # DEBUG
                plt.savefig(savefig, bbox_inches=BBOX_INCHES)
            plt.show() if self.showplot else plt.close()
        return figs

    def plotFreeTransmissions(self, envId=0, savefig=None, cumulated=False):
        """Plot the frequency free transmission."""
        X = self.times - 1
        fig = plt.figure()
        colors = palette(self.nbPlayers)
        for playerId, player in enumerate(self.players):
            Y = self.getFreeTransmissions(playerId, envId)
            if cumulated:
                Y = np.cumsum(Y)
            plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], '.', label=str(player), color=colors[playerId], linewidth=1, markersize=1)
            # should only plot with markers
        legend()
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}${}".format(self.horizon, signature))
        add_percent_formatter("yaxis", 1.0)
        plt.ylabel("{}ransmission on a free channel".format("Cumulated T" if cumulated else "T"))
        plt.title("Multi-players $M = {}$ (collision model: {}):\n{}free transmission for each players, averaged ${}$ times\n{} arms: {}".format(self.nbPlayers, self.collisionModel.__name__, "Cumulated " if cumulated else "", self.cfg['repetitions'], self.envs[envId].nbArms, self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig)
        return fig

    # TODO I should plot the evolution of the occupation ratio of each channel, as a function of time
    # Starting from the average occupation (by primary users), as given by [1 - arm.mean], it should increase occupation[arm] when users chose it
    # The reason/idea is that good arms (low occupation ration) are pulled a lot, thus becoming not as available as they seemed

    def plotNbCollisions(self, envId=0, savefig=None,
                         semilogx=False, semilogy=False, loglog=False,
                         cumulated=False, upperbound=False, evaluators=()):
        """Plot the frequency or cum number of collisions, support more than one environments (use evaluators to give a list of other environments)."""
        X = self.times - 1
        fig = plt.figure()
        evaluators = [self] + list(evaluators)  # Default to only [self]
        colors = palette(len(evaluators))
        markers = makemarkers(len(evaluators))
        plot_method = plt.loglog if loglog else plt.plot
        plot_method = plt.semilogy if semilogy else plot_method
        plot_method = plt.semilogx if semilogx else plot_method
        for evaId, eva in enumerate(evaluators):
            Y = np.zeros(eva.duration)
            for armId in range(eva.envs[envId].nbArms):
                Y += eva.getCollisions(armId, envId)
            if cumulated:
                Y = np.cumsum(Y)
            Y /= eva.nbPlayers  # To normalized the count?
            plot_method(X[::self.delta_t_plot], Y[::self.delta_t_plot], (markers[evaId] + '-') if cumulated else '.', markevery=((evaId / 50., 0.1) if cumulated else None), label=eva.strPlayers(short=True), color=colors[evaId], alpha=1. if cumulated else 0.7)
        if not cumulated:
            add_percent_formatter("yaxis", 1.0)
        # We also plot our lower bound
        if upperbound and cumulated:
            upperboundLog = self.envs[envId].upperbound_collisions(self.nbPlayers, X)
            print("Anandkumar et al. upper bound for the non-cumulated number of collisions is {:.3g} * log(t) here ...".format(upperboundLog[-1]))  # DEBUG
            plot_method(X, upperboundLog, 'k-', label="Anandkumar et al. upper bound", lw=3)
        else:
            print("No upper bound for the non-cumulated number of collisions...")  # DEBUG
        # Start the figure
        plt.xlabel("Time steps $t = 1 .. T$, horizon $T = {}${}".format(self.horizon, signature))
        plt.ylabel("{} of collisions".format("Cumulated number" if cumulated else "Frequency"))
        legend()
        plt.title("Multi-players $M = {}$ (collision model: {}):\n{}of collisions, averaged ${}$ times\n{} arms: {}".format(self.nbPlayers, self.collisionModel.__name__, "Cumulated number " if cumulated else "Frequency ", self.cfg['repetitions'], self.envs[envId].nbArms, self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig)
        return fig

    def plotFrequencyCollisions(self, envId=0, savefig=None, piechart=True, semilogy=False):
        """Plot the frequency of collision, in a pie chart (histogram not supported yet)."""
        nbArms = self.envs[envId].nbArms
        Y = np.zeros(1 + nbArms)  # One extra arm for "no collision"
        labels = [''] * (1 + nbArms)  # Empty labels
        colors = palette(1 + nbArms)  # Get colors
        # All the other arms
        for armId, arm in enumerate(self.envs[envId].arms):
            # Y[armId] = np.sum(self.getCollisions(armId, envId) >= 1)  # XXX no, we should not count just the fact that there were collisions, but instead count all collisions
            Y[armId] = np.sum(self.getCollisions(armId, envId))
        Y /= (self.horizon * self.nbPlayers)
        assert 0 <= np.sum(Y) <= 1, "Error: the sum of collisions = {}, averaged by horizon and nbPlayers, cannot be outside of [0, 1] ...".format(np.sum(Y))
        for armId, arm in enumerate(self.envs[envId].arms):
            labels[armId] = "#${}$: ${}$ (${:.1%}$$\%$)".format(armId, repr(arm), Y[armId])
            print("  - For {},\tfrequency of collisions is {:g}  ...".format(labels[armId], Y[armId]))  # DEBUG
            if Y[armId] < 1e-4:  # Do not display small slices
                labels[armId] = ''
        if np.isclose(np.sum(Y), 0):
            print("==> No collisions to plot ... Stopping now  ...")  # DEBUG
            return
        # Special arm: no collision
        Y[-1] = 1 - np.sum(Y) if np.sum(Y) < 1 else 0
        labels[-1] = "No collision (${:.1%}$$\%$)".format(Y[-1]) if Y[-1] > 1e-4 else ''
        colors[-1] = 'lightgrey'
        # Start the figure
        fig = plt.figure()
        plt.xlabel("{}{}".format(self.strPlayers(), signature))
        if piechart:
            plt.axis('equal')
            plt.pie(Y, labels=labels, colors=colors, explode=[0.07] * len(Y), startangle=45)
        else:
            if semilogy:  # FIXME is it perfectly working?
                Y = np.log10(Y)  # use semilogy scale!
                Y -= np.min(Y)   # project back to [0, oo)
                Y /= np.sum(Y)   # project back to [0, 1)
            for i in range(len(Y)):
                plt.axvspan(i - 0.25, i + 0.25, 0, Y[i], label=labels[i], color=colors[i])
            plt.xticks(np.arange(len(Y)), ['Collision \non arm #$%i$' % i for i in range(nbArms)] + ['No collision'])
            plt.ylabel("Frequency of collision, in logarithmic scale" if semilogy else "Frequency of collision")
            if not semilogy:
                add_percent_formatter("yaxis", 1.0)
        legend()
        plt.title("Multi-players $M = {}$ (collision model: {}):\nFrequency of collision for each arm, averaged ${}$ times\n{} arms: {}".format(self.nbPlayers, self.collisionModel.__name__, self.cfg['repetitions'], self.envs[envId].nbArms, self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig)
        return fig

    def printFinalRanking(self, envId=0):
        """Compute and print the ranking of the different players."""
        assert 0 < self.averageOn < 1, "Error, the parameter averageOn of a EvaluatorMultiPlayers classs has to be in (0, 1) strictly, but is = {} here ...".format(self.averageOn)
        print("\nFinal ranking for this environment #{} :".format(envId))  # DEBUG
        lastY = np.zeros(self.nbPlayers)
        for playerId, player in enumerate(self.players):
            Y = self.getRewards(playerId, envId)
            if self.finalRanksOnAverage:
                lastY[playerId] = np.mean(Y[-int(self.averageOn * self.duration)])   # get average value during the last averageOn% of the iterations
            else:
                lastY[playerId] = Y[-1]  # get the last value
        # Sort lastY and give ranking
        index_of_sorting = np.argsort(-lastY)  # Get them by INCREASING rewards, not decreasing regrets
        for i, k in enumerate(index_of_sorting):
            player = self.players[k]
            print("- Player #{}, '{}'\twas ranked\t{} / {} for this simulation (last rewards = {:g}).".format(k + 1, str(player), i + 1, self.nbPlayers, lastY[k]))  # DEBUG
        return lastY, index_of_sorting

    def strPlayers(self, short=False):
        """Get a string of the players for this environment."""
        listStrPlayers = [_extract(str(player)) for player in self.players]
        if len(set(listStrPlayers)) == 1:  # Unique user
            text = r'${} \times$ {}'.format(self.nbPlayers, listStrPlayers[0])
        else:
            text = ', '.join(listStrPlayers)
        text = wraptext(text)
        if not short:
            text = '{} players: {}'.format(self.nbPlayers, text)
        return text


def delayed_play(env, players, horizon, collisionModel,
                 delta_t_save=1, seed=None, repeatId=0):
    """Helper function for the parallelization."""
    # Give a unique seed to random & numpy.random for each call of this function
    try:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    except (ValueError, SystemError):
        print("Warning: setting random.seed and np.random.seed seems to not be available. Are you using Windows?")  # XXX
    # We have to deepcopy because this function is Parallel-ized
    # XXX this uses a LOT of RAM memory!!!
    # env = deepcopy(env)
    if env.isDynamic:  # FIXME
        env.newRandomArms()
    players = deepcopy(players)
    nbArms = env.nbArms
    nbPlayers = len(players)
    # random_arm_orders = [np.random.permutation(nbArms) for i in range(nbPlayers)]
    # Start game
    for player in players:
        player.startGame()
    # Store results
    result = ResultMultiPlayers(env.nbArms, horizon, nbPlayers)
    # , delta_t_save=delta_t_save
    rewards = np.zeros(nbPlayers)
    choices = np.zeros(nbPlayers, dtype=int)
    pulls = np.zeros((nbPlayers, nbArms), dtype=int)
    collisions = np.zeros(nbArms, dtype=int)

    prettyRange = tqdm(range(horizon), desc="Time t") if repeatId == 0 else range(horizon)
    for t in prettyRange:
        # Reset the array, faster than reallocating them!
        rewards.fill(0)
        pulls.fill(0)
        collisions.fill(0)
        # Every player decides which arm to pull
        for playerId, player in enumerate(players):
            # FIXME here, the environment should apply ONCE a random permutation to each player, in order for the non-modified UCB-like algorithms to work fine in case of collisions (their initial exploration phase is non-random hence leading to only collisions in the first steps, and ruining the performance)
            # choices[i] = random_arm_orders[i][player.choice()]
            choices[playerId] = player.choice()
            # print(" Round t = \t{}, player \t#{}/{} ({}) \tchose : {} ...".format(t, i + 1, len(players), player, choices[i]))  # DEBUG

        # Then we decide if there is collisions and what to do why them
        # XXX It is here that the player may receive a reward, if there is no collisions
        collisionModel(t, env.arms, players, choices, rewards, pulls, collisions)

        # if t % delta_t_save == 0:  # XXX inefficient and does not work yet
        #     if delta_t_save > 1: print("t =", t, "delta_t_save =", delta_t_save, " : saving ...")  # DEBUG
        # Finally we store the results
        result.store(t, choices, rewards, pulls, collisions)

    # Print the quality of estimation of arm ranking for this policy, just for 1st repetition
    if repeatId == 0:
        for playerId, player in enumerate(players):
            try:
                order = player.estimatedOrder()
                print("\nEstimated order by the policy {} after {} steps: {} ...".format(player, horizon, order))
                print("  ==> Optimal arm identification: {:.2%} (relative success)...".format(weightedDistance(order, env.means, n=nbPlayers)))
                print("  ==> Manhattan   distance from optimal ordering: {:.2%} (relative success)...".format(manhattan(order)))
                print("  ==> Kendell Tau distance from optimal ordering: {:.2%} (relative success)...".format(kendalltau(order)))
                print("  ==> Spearman    distance from optimal ordering: {:.2%} (relative success)...".format(spearmanr(order)))
                print("  ==> Gestalt     distance from optimal ordering: {:.2%} (relative success)...".format(gestalt(order)))
                print("  ==> Mean distance from optimal ordering: {:.2%} (relative success)...".format(meanDistance(order)))
            except AttributeError:
                print("Unable to print the estimated ordering, no method estimatedOrder was found!")

    return result


def _extract(text):
    """ Extract the str of a player, if it is a child, printed as '#[0-9]+<...>' --> ... """
    try:
        m = search("<[^>]+>", text).group(0)
        if m[0] == '<' and m[-1] == '>':
            return m[1:-1]  # Extract text between < ... >
        else:
            return text
    except AttributeError:
        return text
