# -*- coding: utf-8 -*-
""" EvaluatorMultiPlayers class to wrap and run the simulations, for the multi-players case.
Lots of plotting methods, to have various visualizations. See documentation.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

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
from .plotsettings import BBOX_INCHES, signature, maximizeWindow, palette, makemarkers, add_percent_formatter, wraptext, wraplatex, legend, show_and_save, nrows_ncols, addTextForWorstCases
from .sortedDistance import weightedDistance, manhattan, kendalltau, spearmanr, gestalt, meanDistance, sortedDistance
from .fairnessMeasures import amplitude_fairness, std_fairness, rajjain_fairness, mean_fairness, fairnessMeasure, fairness_mapping
# Local imports, objects and functions
from .CollisionModels import onlyUniqUserGetsReward, noCollision, closerUserGetsReward, rewardIsSharedUniformly, defaultCollisionModel, full_lost_if_collision
from .MAB import MAB, MarkovianMAB, DynamicMAB
from .ResultMultiPlayers import ResultMultiPlayers

REPETITIONS = 1  #: Default nb of repetitions
DELTA_T_PLOT = 50  #: Default sampling rate for plotting
COUNT_RANKS_MARKOV_CHAIN = False  #: If true, count and then print a lot of statistics for the Markov Chain of the underlying configurations on ranks

MORE_ACCURATE = False          #: Use the count of selections instead of rewards for a more accurate mean/std reward measure.
MORE_ACCURATE = True           #: Use the count of selections instead of rewards for a more accurate mean/std reward measure.

plot_lowerbounds = True  #: Default is to plot the lower-bounds

FINAL_RANKS_ON_AVERAGE = True  #: Default value for ``finalRanksOnAverage``
USE_JOBLIB_FOR_POLICIES = False  #: Default value for ``useJoblibForPolicies``. Does not speed up to use it (too much overhead in using too much threads); so it should really be disabled.
PICKLE_IT = True  #: Default value for ``pickleit`` for saving the figures. If True, then all ``plt.figure`` object are saved (in pickle format).

# --- Class EvaluatorMultiPlayers

class EvaluatorMultiPlayers(object):
    """ Evaluator class to run the simulations, for the multi-players case.
    """

    def __init__(self, configuration,
                 moreAccurate=MORE_ACCURATE):
        # Configuration
        self.cfg = configuration  #: Configuration dictionnary
        # Attributes
        self.nbPlayers = len(self.cfg['players'])  #: Number of policies
        print("Number of players in the multi-players game:", self.nbPlayers)
        self.horizon = self.cfg['horizon']  #: Horizon (number of time steps)
        print("Time horizon:", self.horizon)
        self.repetitions = self.cfg.get('repetitions', REPETITIONS)  #: Number of repetitions
        print("Number of repetitions:", self.repetitions)
        self.delta_t_plot = 1 if self.horizon <= 10000 else self.cfg.get('delta_t_plot', DELTA_T_PLOT)
        print("Sampling rate for plotting, delta_t_plot:", self.delta_t_plot)  #: Sampling rate for plotting
        self.horizon = int(self.horizon)
        print("Number of jobs for parallelization:", self.cfg['n_jobs'])
        self.collisionModel = self.cfg.get('collisionModel', defaultCollisionModel)  #: Which collision model should be used
        self.full_lost_if_collision = full_lost_if_collision.get(self.collisionModel.__name__, True)  #: Is there a full loss of rewards if collision ? To compute the correct decomposition of regret
        print("Using collision model {} (function {}).\nMore details:\n{}".format(self.collisionModel.__name__, self.collisionModel, self.collisionModel.__doc__))
        self.signature = signature
        # Flags
        self.moreAccurate = moreAccurate  #: Use the count of selections instead of rewards for a more accurate mean/std reward measure.
        print("Using accurate regrets and last regrets ? {}".format(moreAccurate))
        self.finalRanksOnAverage = self.cfg.get('finalRanksOnAverage', FINAL_RANKS_ON_AVERAGE)  #: Final display of ranks are done on average rewards?
        self.averageOn = self.cfg.get('averageOn', 5e-3)  #: How many last steps for final rank average rewards
        self.plot_lowerbounds = self.cfg.get('plot_lowerbounds', plot_lowerbounds)  #: Should we plot the lower-bounds?
        self.useJoblib = USE_JOBLIB and self.cfg['n_jobs'] != 1  #: Use joblib to parallelize for loop on repetitions (useful)
        self.showplot = self.cfg.get('showplot', True)  #: Show the plot (interactive display or not)
        self.count_ranks_markov_chain = self.cfg.get('count_ranks_markov_chain', COUNT_RANKS_MARKOV_CHAIN)#: If true, count and then print a lot of statistics for the Markov Chain of the underlying configurations on ranks

        self.change_labels = self.cfg.get('change_labels', {})  #: Possibly empty dictionary to map 'policyId' to new labels (overwrite their name).
        self.append_labels = self.cfg.get('append_labels', {})  #: Possibly empty dictionary to map 'policyId' to new labels (by appending the result from 'append_labels').

        # Internal object memory
        self.envs = []  #: List of environments
        self.players = []  #: List of policies
        self.__initEnvironments__()
        # Internal vectorial memory
        self.rewards = dict()  #: For each env, history of rewards
        # self.rewardsSquared = dict()
        self.pulls = dict()  #: For each env, keep the history of arm pulls (mean)
        self.lastPulls = dict()  #: For each env, keep the distribution of arm pulls
        self.allPulls = dict()  #: For each env, keep the full history of arm pulls
        self.collisions = dict()  #: For each env, keep the history of collisions on all arms
        self.lastCumCollisions = dict()  #: For each env, last count of collisions on all arms
        self.nbSwitchs = dict()  #: For each env, keep the history of switches (change of configuration of players)
        self.bestArmPulls = dict()  #: For each env, keep the history of best arm pulls
        self.freeTransmissions = dict()  #: For each env, keep the history of successful transmission (1 - collisions, basically)
        self.lastCumRewards = dict()  #: For each env, last accumulated rewards, to compute variance and histogram of whole regret R_T

        print("Number of environments to try:", len(self.envs))  # DEBUG
        # XXX: WARNING no memorized vectors should have dimension duration * repetitions, that explodes the RAM consumption!
        for envId in range(len(self.envs)):  # Zeros everywhere
            self.rewards[envId] = np.zeros((self.nbPlayers, self.horizon))
            # self.rewardsSquared[envId] = np.zeros((self.nbPlayers, self.horizon))
            self.lastCumRewards[envId] = np.zeros(self.repetitions)
            self.pulls[envId] = np.zeros((self.nbPlayers, self.envs[envId].nbArms))
            self.lastPulls[envId] = np.zeros((self.nbPlayers, self.envs[envId].nbArms, self.repetitions))
            self.allPulls[envId] = np.zeros((self.nbPlayers, self.envs[envId].nbArms, self.horizon))
            self.collisions[envId] = np.zeros((self.envs[envId].nbArms, self.horizon))
            self.lastCumCollisions[envId] = np.zeros((self.envs[envId].nbArms, self.repetitions))
            self.nbSwitchs[envId] = np.zeros((self.nbPlayers, self.horizon))
            self.bestArmPulls[envId] = np.zeros((self.nbPlayers, self.horizon))
            self.freeTransmissions[envId] = np.zeros((self.nbPlayers, self.horizon))
        # To speed up plotting
        self._times = np.arange(1, 1 + self.horizon)

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
            print("- Adding player #{:>2} = {} ...".format(playerId + 1, player))  # DEBUG
            if isinstance(player, dict):  # Either the 'player' is a config dict
                print("  Creating this player from a dictionnary 'player' = {} ...".format(player))  # DEBUG
                self.players.append(player['archtype'](env.nbArms, **player['params']))
            else:  # Or already a player object
                print("  Using this already created player 'player' = {} ...".format(player))  # DEBUG
                self.players.append(player)
        for playerId in range(len(self.players)):
            self.players[playerId].__cachedstr__ = str(self.players[playerId])
            if playerId in self.append_labels:
                self.players[playerId].__cachedstr__ += self.append_labels[playerId]
            if playerId in self.change_labels:
                self.players[playerId].__cachedstr__ = self.append_labels[playerId]

    # --- Start computation

    def startAllEnv(self):
        """Simulate all envs."""
        for envId, env in enumerate(self.envs):
            self.startOneEnv(envId, env)

    def startOneEnv(self, envId, env):
        """Simulate that env."""
        print("\n\nEvaluating environment:", repr(env))  # DEBUG
        self.players = []
        self.__initPlayers__(env)
        # Get the position of the best arms
        means = env.means
        bestarm = env.maxArm
        indexes_bestarm = np.nonzero(np.isclose(means, bestarm))[0]

        def store(r, repeatId):
            """Store the result of the experiment r."""
            self.rewards[envId] += np.cumsum(r.rewards, axis=1)  # cumsum on time
            # self.rewardsSquared[envId] += np.cumsum(r.rewards ** 2, axis=1)  # cumsum on time
            # self.rewardsSquared[envId] += np.cumsum(r.rewardsSquared, axis=1)  # cumsum on time
            self.lastCumRewards[envId][repeatId] = np.sum(r.rewards)  # sum on time and sum on policies
            self.pulls[envId] += r.pulls
            self.lastPulls[envId][:, :, repeatId] = r.pulls
            self.allPulls[envId] += r.allPulls
            self.collisions[envId] += r.collisions
            self.lastCumCollisions[envId][:, repeatId] = np.sum(r.collisions, axis=1)  # sum on time
            for playerId in range(self.nbPlayers):
                self.nbSwitchs[envId][playerId, 1:] += (np.diff(r.choices[playerId, :]) != 0)
                self.bestArmPulls[envId][playerId, :] += np.cumsum(np.in1d(r.choices[playerId, :], indexes_bestarm))
                # FIXME there is probably a bug in this computation
                self.freeTransmissions[envId][playerId, :] += np.array([r.choices[playerId, t] not in r.collisions[:, t] for t in range(self.horizon)])

        # Start now
        if self.useJoblib:
            seeds = np.random.randint(low=0, high=100 * self.repetitions, size=self.repetitions)
            repeatIdout = 0
            historyOfMeans = []
            for r in Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
                delayed(delayed_play)(env, self.players, self.horizon, self.collisionModel, seed=seeds[repeatId], repeatId=repeatId, count_ranks_markov_chain=self.count_ranks_markov_chain)
                for repeatId in tqdm(range(self.repetitions), desc="Repeat||")
            ):
                historyOfMeans.append(r._means)
                store(r, repeatIdout)
                repeatIdout += 1
            if env.isDynamic:
                env._t += self.repetitions  # new self.repetitions draw!
                env._historyOfMeans = historyOfMeans
        else:
            for repeatId in tqdm(range(self.repetitions), desc="Repeat"):
                r = delayed_play(env, self.players, self.horizon, self.collisionModel, repeatId=repeatId, count_ranks_markov_chain=self.count_ranks_markov_chain)
                store(r, repeatId)

    # --- Getter methods

    def getPulls(self, playerId, envId=0):
        """Extract mean pulls."""
        return self.pulls[envId][playerId, :] / float(self.repetitions)

    def getAllPulls(self, playerId, armId, envId=0):
        """Extract mean of all pulls."""
        return self.allPulls[envId][playerId, armId, :] / float(self.repetitions)

    def getNbSwitchs(self, playerId, envId=0):
        """Extract mean nb of switches."""
        return self.nbSwitchs[envId][playerId, :] / float(self.repetitions)

    def getCentralizedNbSwitchs(self, envId=0):
        """Extract average of mean nb of switches."""
        return np.sum(self.nbSwitchs[envId], axis=0) / (float(self.repetitions) * self.nbPlayers)

    def getBestArmPulls(self, playerId, envId=0):
        """Extract mean of best arms pulls."""
        # We have to divide by a arange() = cumsum(ones) to get a frequency
        return self.bestArmPulls[envId][playerId, :] / (float(self.repetitions) * self._times)

    def getfreeTransmissions(self, playerId, envId=0):
        """Extract mean of successful transmission."""
        return self.freeTransmissions[envId][playerId, :] / float(self.repetitions)

    def getCollisions(self, armId, envId=0):
        """Extract mean of number of collisions."""
        return self.collisions[envId][armId, :] / float(self.repetitions)

    def getRewards(self, playerId, envId=0):
        """Extract mean of rewards."""
        return self.rewards[envId][playerId, :] / float(self.repetitions)

    def getRegretMean(self, playerId, envId=0):
        """Extract mean of regret

        .. warning:: This is the centralized regret, for one arm, it does not make much sense in the multi-players setting!
        """
        return (self._times - 1) * self.envs[envId].maxArm - self.getRewards(playerId, envId)

    def getCentralizedRegret_LessAccurate(self, envId=0):
        """Compute the empirical centralized regret: cumsum on time of the mean rewards of the M best arms - cumsum on time of the empirical rewards obtained by the players, based on accumulated rewards."""
        meansArms = np.sort(self.envs[envId].means)
        sumBestMeans = self.envs[envId].sumBestMeans(self.nbPlayers)
        # FIXED how to count it when there is more players than arms ?
        # FIXME it depends on the collision model !
        if self.envs[envId].nbArms < self.nbPlayers:
            # sure to have collisions, then the best strategy is to put all the collisions in the worse arm
            worseArm = np.min(meansArms)
            sumBestMeans -= worseArm  # This count the collisions
        averageBestRewards = self._times * sumBestMeans
        # And for the actual rewards, the collisions are counted in the rewards logged in self.getRewards
        actualRewards = np.sum(self.rewards[envId][:, :], axis=0) / float(self.repetitions)
        return averageBestRewards - actualRewards

    # --- Three terms in the regret

    def getFirstRegretTerm(self, envId=0):
        """Extract and compute the first term :math:`(a)` in the centralized regret: losses due to pulling suboptimal arms."""
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
        """Extract and compute the second term :math:`(b)` in the centralized regret: losses due to not pulling optimal arms."""
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
        """Extract and compute the third term :math:`(c)` in the centralized regret: losses due to collisions."""
        means = self.envs[envId].means
        countCollisions = self.collisions[envId]   # Shape: (nbArms, duration)
        if not self.full_lost_if_collision:
            print("Warning: the collision model ({}) does *not* yield a loss in communication when colliding (one user can communicate, or in average one user can communicate), so countCollisions -= 1 for the 3rd regret term ...".format(self.collisionModel.__name__))  # DEBUG
            countCollisions = np.maximum(0, countCollisions - 1)
        losses = np.dot(means, countCollisions / float(self.repetitions))  # Count and sum on k in 1...K
        thirdRegretTerm = np.cumsum(losses)  # Accumulate losses
        return thirdRegretTerm

    def getCentralizedRegret_MoreAccurate(self, envId=0):
        """Compute the empirical centralized regret, based on counts of selections and not actual rewards."""
        return self.getFirstRegretTerm(envId=envId) + self.getSecondRegretTerm(envId=envId) + self.getThirdRegretTerm(envId=envId)

    def getCentralizedRegret(self, envId=0, moreAccurate=None):
        """Using either the more accurate or the less accurate regret count."""
        moreAccurate = moreAccurate if moreAccurate is not None else self.moreAccurate
        # print("Computing the vector of mean cumulated regret with '{}' accurate method...".format("more" if moreAccurate else "less"))  # DEBUG
        if moreAccurate:
            return self.getCentralizedRegret_MoreAccurate(envId=envId)
        else:
            return self.getCentralizedRegret_LessAccurate(envId=envId)

    # --- Last regrets

    def getLastRegrets_LessAccurate(self, envId=0):
        """Extract last regrets, based on accumulated rewards."""
        meansArms = np.sort(self.envs[envId].means)
        sumBestMeans = self.envs[envId].sumBestMeans(self.nbPlayers)
        # FIXED how to count it when there is more players than arms ?
        # FIXME it depends on the collision model !
        if self.envs[envId].nbArms < self.nbPlayers:
            # sure to have collisions, then the best strategy is to put all the collisions in the worse arm
            worseArm = np.min(meansArms)
            sumBestMeans -= worseArm  # This count the collisions
        return self.horizon * sumBestMeans - self.lastCumRewards[envId]

    def getAllLastWeightedSelections(self, envId=0):
        """Extract weighted count of selections."""
        all_last_weighted_selections = np.zeros(self.repetitions)
        lastCumCollisions = self.lastCumCollisions[envId]
        for armId, mean in enumerate(self.envs[envId].means):
            last_selections = np.sum(self.lastPulls[envId][:, armId, :], axis=0)  # sum on players
            all_last_weighted_selections += mean * (last_selections - lastCumCollisions[armId, :])
        return all_last_weighted_selections

    def getLastRegrets_MoreAccurate(self, envId=0):
        """Extract last regrets, based on counts of selections and not actual rewards."""
        meansArms = np.sort(self.envs[envId].means)
        sumBestMeans = self.envs[envId].sumBestMeans(self.nbPlayers)
        # FIXED how to count it when there is more players than arms ?
        # FIXME it depends on the collision model !
        if self.envs[envId].nbArms < self.nbPlayers:
            # sure to have collisions, then the best strategy is to put all the collisions in the worse arm
            worseArm = np.min(meansArms)
            sumBestMeans -= worseArm  # This count the collisions
        return self.horizon * sumBestMeans - self.getAllLastWeightedSelections(envId=envId)

    def getLastRegrets(self, envId=0, moreAccurate=None):
        """Using either the more accurate or the less accurate regret count."""
        moreAccurate = moreAccurate if moreAccurate is not None else self.moreAccurate
        # print("Computing the vector of last cumulated regrets (on repetitions) with '{}' accurate method...".format("more" if moreAccurate else "less"))  # DEBUG
        if moreAccurate:
            return self.getLastRegrets_MoreAccurate(envId=envId)
        else:
            return self.getLastRegrets_LessAccurate(envId=envId)

    # --- Plotting methods

    def plotRewards(self, envId=0, savefig=None, semilogx=False, moreAccurate=None):
        """Plot the decentralized (vectorial) rewards, for each player."""
        moreAccurate = moreAccurate if moreAccurate is not None else self.moreAccurate
        fig = plt.figure()
        ymin = 0
        colors = palette(self.nbPlayers)
        markers = makemarkers(self.nbPlayers)
        X = self._times - 1
        cumRewards = np.zeros((self.nbPlayers, self.horizon))
        for playerId, player in enumerate(self.players):
            label = 'Player #{:>2}: {}'.format(playerId + 1, _extract(player.__cachedstr__))
            Y = self.getRewards(playerId, envId)
            cumRewards[playerId, :] = Y
            ymin = min(ymin, np.min(Y))  # XXX Should be smarter
            if semilogx:
                plt.semilogx(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[playerId], marker=markers[playerId], markevery=(playerId / 50., 0.1), lw=2)
            else:
                plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[playerId], marker=markers[playerId], markevery=(playerId / 50., 0.1), lw=2)
        legend()
        plt.xlabel("Time steps $t = 1...T$, horizon $T = {}${}".format(self.horizon, self.signature))
        plt.ylabel("Cumulative personal reward {}".format(r"$\sum_{k=1}^{%d} \mu_k\mathbb{E}_{%d}[T_k(t)]$" % (self.envs[envId].nbArms, self.repetitions) if moreAccurate else r"$\mathbb{E}_{%d}[r_t]$" % self.repetitions))
        plt.title("Multi-players $M = {}$ : Personal reward for each player, averaged ${}$ times\n${}$ arms{}: {}".format(self.nbPlayers, self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig, fig=fig, pickleit=PICKLE_IT)
        return fig

    def plotFairness(self, envId=0, savefig=None, semilogx=False, fairness="default", evaluators=()):
        """Plot a certain measure of "fairness", from these personal rewards, support more than one environments (use evaluators to give a list of other environments)."""
        fig = plt.figure()
        X = self._times - 1
        evaluators = [self] + list(evaluators)  # Default to only [self]
        colors = palette(len(evaluators))
        markers = makemarkers(len(evaluators))
        plot_method = plt.semilogx if semilogx else plt.plot
        # Decide which fairness function to use
        fairnessFunction = fairness_mapping[fairness] if isinstance(fairness, str) else fairness
        fairnessName = fairness if isinstance(fairness, str) else getattr(fairness, '__name__', "std_fairness")
        for evaId, eva in enumerate(evaluators):
            label = eva.strPlayers(short=True)
            cumRewards = np.zeros((eva.nbPlayers, eva.horizon))
            for playerId, player in enumerate(eva.players):
                cumRewards[playerId, :] = eva.getRewards(playerId, envId)
            # # Print each fairness measure  # DEBUG
            # for fN, fF in fairness_mapping.items():
            #     f = fF(cumRewards)
            #     print("  - {} fairness index is = {} ...".format(fN, f))  # DEBUG
            # Plot only one fairness term
            fairness = fairnessFunction(cumRewards)
            plot_method(X[::self.delta_t_plot][2:], fairness[::self.delta_t_plot][2:], markers[evaId] + '-', label=label, markevery=(evaId / 50., 0.1), color=colors[evaId], lw=2)
        if len(evaluators) > 1:
            legend()
        plt.xlabel("Time steps $t = 1...T$, horizon $T = {}$, {}{}".format(self.horizon, self.strPlayers() if len(evaluators) == 1 else "", self.signature))
        add_percent_formatter("yaxis", 1.0)
        # plt.ylim(0, 1)
        plt.ylabel("Centralized measure of fairness for cumulative rewards ({})".format(fairnessName.title()))
        plt.title("Multi-players $M = {}$ : Centralized measure of fairness, averaged ${}$ times\n${}$ arms{}: {}".format(self.nbPlayers, self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig, fig=fig, pickleit=PICKLE_IT)
        return fig

    def plotRegretCentralized(self, envId=0, savefig=None,
                              semilogx=False, semilogy=False, loglog=False,
                              normalized=False, evaluators=(),
                              subTerms=False, sumofthreeterms=False, moreAccurate=None):
        """Plot the centralized cumulated regret, support more than one environments (use evaluators to give a list of other environments).

        - The lower bounds are also plotted (Besson & Kaufmann, and Anandkumar et al).
        - The three terms of the regret are also plotting if evaluators = () (that's the default).
        """
        moreAccurate = moreAccurate if moreAccurate is not None else self.moreAccurate
        X0 = X = self._times - 1
        fig = plt.figure()
        # if subTerms or len(list(evaluators)) == 0:  # XXX
        #     moreAccurate = False  # if no other guys, the three terms are also plotted, and their sum also, so we use the "real empirical regret"
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
                labels[0] = "$(a)$ term: Pulls of {} suboptimal arms (lower-bounded)".format(max(0, self.envs[envId].nbArms - self.nbPlayers))
                Ys[1] = eva.getSecondRegretTerm(envId)
                labels[1] = "$(b)$ term: Non-pulls of {} optimal arms".format(min(self.nbPlayers, self.envs[envId].nbArms))
                Ys[2] = eva.getThirdRegretTerm(envId)
                labels[2] = "$(c)$ term: Weighted count of collisions"
            Y = eva.getCentralizedRegret(envId, moreAccurate=moreAccurate)
            label = "{}umulated centralized regret".format("Normalized c" if normalized else "C") if len(evaluators) == 1 else eva.strPlayers(short=True)
            if semilogx or loglog:  # FIXED for semilogx plots, truncate to only show t >= 100
                X, Y = X0[X0 >= 100], Y[X0 >= 100]
                if subTerms:
                    for i in range(len(Ys)):
                            Ys[i] = Ys[i][X0 >= 100]
            if normalized:
                Y = Y[X >= 1] / np.log(X[X >= 1])   # XXX prevent /0
                if subTerms:
                    for i in range(len(Ys)):
                        Ys[i] = Ys[i][X >= 1] / np.log(X[X >= 1])  # XXX prevent /0
            meanY = np.mean(Y)
            # Now plot
            plot_method(X[::self.delta_t_plot], Y[::self.delta_t_plot], (markers[evaId] + '-'), markevery=(evaId / 50., 0.1), label=label, color=colors[evaId], lw=2)
            if len(evaluators) == 1:
                # if not semilogx and not loglog and not semilogy:
                #     # We plot a horizontal line ----- at the mean regret
                #     plot_method(X[::self.delta_t_plot], meanY * np.ones_like(X)[::self.delta_t_plot], '--', label="Mean cumulated centralized regret", color=colors[evaId], lw=2)
                # " = ${:.3g}$".format(meanY)
                if subTerms:
                    if sumofthreeterms:
                        Ys.append(Ys[0] + Ys[1] + Ys[2])
                        labels.append("Sum of 3 terms (= regret)")
                    # print("Difference between regret and sum of three terms:", Y - np.array(Ys[-1]))  # DEBUG
                    for i, (Y, label) in enumerate(zip(Ys, labels)):
                        plot_method(X[::self.delta_t_plot], Y[::self.delta_t_plot], (markers[i + 1] + '-'), markevery=((i + 1) / 50., 0.1), label=label, color=colors[i + 1], lw=2)
                        if semilogx or loglog:  # Manual fix for issue https://github.com/SMPyBandits/SMPyBandits/issues/38
                            plt.xscale('log')
                        if semilogy or loglog:  # Manual fix for issue https://github.com/SMPyBandits/SMPyBandits/issues/38
                            plt.yscale('log')
        # We also plot our lower bound
        if not self.envs[envId].isDynamic:
            try:
                # XXX In fact, the lower-bound is also true for Bayesian policies! Finite means ARE ALWAYS linear! I should write the proof, but I convinced myself that the lower-bound is still correct (in a certain sense) and at least it gives an overview of the (average) complexity of the problem (randomly drawn and) used for the experiments.
                lowerbound, anandkumar_lowerbound, centralized_lowerbound = self.envs[envId].lowerbound_multiplayers(self.nbPlayers)
                if not (semilogx or semilogy or loglog):
                    print("\nThis MAB problem has: \n - a [Lai & Robbins] complexity constant C(mu) = {:.3g} for 1-player problem ... \n - a Optimal Arm Identification factor H_OI(mu) = {:.2%} ...".format(self.envs[envId].lowerbound(), self.envs[envId].hoifactor()))  # DEBUG
                if self.envs[envId].isDynamic:
                    print("WARNING this env is in fact dynamic, this complexity term and H_OI factor do not have much sense... (they are computed from the average of the complexity for all mean vectors drawn in the repeted experiments...)")  # DEBUG
                print(" - [Anandtharam et al] centralized lower-bound = {:.3g},\n - [Anandkumar et al] decentralized lower-bound = {:.3g}\n - Our better (larger) decentralized lower-bound = {:.3g},".format(centralized_lowerbound, anandkumar_lowerbound, lowerbound))  # DEBUG
                if normalized:
                    T = np.ones_like(X)
                else:
                    X = X[X >= 1]
                    T = np.log(X)
                if self.plot_lowerbounds:
                    plot_method(X[::self.delta_t_plot], lowerbound * T[::self.delta_t_plot], 'k-', label="Besson & Kaufmann lower-bound = ${:.3g} \; \log(t)$".format(lowerbound), lw=3)
                    plot_method(X[::self.delta_t_plot], anandkumar_lowerbound * T[::self.delta_t_plot], 'k--', label="Anandkumar et al.'s lower-bound = ${:.3g} \; \log(t)$".format(anandkumar_lowerbound), lw=2)
                    plot_method(X[::self.delta_t_plot], centralized_lowerbound * T[::self.delta_t_plot], 'k:', label="Centralized lower-bound = ${:.3g} \; \log(t)$".format(centralized_lowerbound), lw=2)
            except AssertionError:
                print("Error: Unable to compute and display the lower-bound...")  # DEBUG
        # Labels and legends
        legend()
        plt.xlabel("Time steps $t = 1...T$, horizon $T = {}$, {}{}".format(self.horizon, self.strPlayers() if len(evaluators) == 1 else "", self.signature))
        plt.ylabel("{}umulative centralized regret {}".format("Normalized c" if normalized else "C", r"$\sum_{k=1}^{%d}\mu_k^* t - \sum_{k=1}^{%d} \mu_k\mathbb{E}_{%d}[T_k(t)]$" % (self.nbPlayers, self.envs[envId].nbArms, self.repetitions) if moreAccurate else r"$\mathbb{E}_{%d}[R_t]$" % self.repetitions))
        plt.title("Multi-players $M = {}$ : {}umulated centralized regret, averaged ${}$ times\n${}$ arms{}: {}".format(self.nbPlayers, "Normalized c" if normalized else "C", self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig, fig=fig, pickleit=PICKLE_IT)
        return fig

    def plotNbSwitchs(self, envId=0, savefig=None, semilogx=False, cumulated=False):
        """Plot cumulated number of switchs (to evaluate the switching costs), comparing each player."""
        X = self._times - 1
        fig = plt.figure()
        ymin = 0
        colors = palette(self.nbPlayers)
        markers = makemarkers(self.nbPlayers)
        plot_method = plt.semilogx if semilogx else plt.plot
        for playerId, player in enumerate(self.players):
            label = 'Player #{:>2}: {}'.format(playerId + 1, _extract(player.__cachedstr__))
            Y = self.getNbSwitchs(playerId, envId)
            if cumulated:
                Y = np.cumsum(Y)
            ymin = min(ymin, np.min(Y))  # XXX Should be smarter
            plot_method(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[playerId], marker=markers[playerId], markevery=(playerId / 50., 0.1), linestyle='-' if cumulated else '', lw=2)
        legend()
        plt.xlabel("Time steps $t = 1...T$, horizon $T = {}${}".format(self.horizon, self.signature))
        plt.ylim(ymin, max(plt.ylim()[1], 1))
        if not cumulated: add_percent_formatter("yaxis", 1.0)
        plt.ylabel("{} of switches by player".format("Cumulated number" if cumulated else "Frequency"))
        plt.title("Multi-players $M = {}$ : {}umber of switches for each player, averaged ${}$ times\n{} arm{}s: {}".format(self.nbPlayers, "Cumulated n" if cumulated else "N", self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig, fig=fig, pickleit=PICKLE_IT)
        return fig

    def plotNbSwitchsCentralized(self, envId=0, savefig=None, semilogx=False, cumulated=False, evaluators=()):
        """Plot the centralized cumulated number of switchs (to evaluate the switching costs), support more than one environments (use evaluators to give a list of other environments)."""
        X = self._times - 1
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
            plot_method(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[evaId], marker=markers[evaId], markevery=(evaId / 50., 0.1), linestyle='-' if cumulated else '', lw=2)
        if len(evaluators) > 1:
            legend()
        plt.xlabel("Time steps $t = 1...T$, horizon $T = {}$, {}{}".format(self.horizon, self.strPlayers() if len(evaluators) == 1 else "", self.signature))
        if not cumulated: add_percent_formatter("yaxis", 1.0)
        plt.ylabel("{} of switches (changes of arms)".format("Cumulated number" if cumulated else "Frequency"))
        plt.title("Multi-players $M = {}$ : Total {}number of switches, averaged ${}$ times\n${}$ arms{}: {}".format(self.nbPlayers, "cumulated " if cumulated else "", self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig, fig=fig, pickleit=PICKLE_IT)
        return fig

    def plotBestArmPulls(self, envId=0, savefig=None):
        """Plot the frequency of pulls of the best channel.

        - Warning: does not adapt to dynamic settings!
        """
        X = self._times - 1
        fig = plt.figure()
        colors = palette(self.nbPlayers)
        markers = makemarkers(self.nbPlayers)
        for playerId, player in enumerate(self.players):
            label = 'Player #{:>2}: {}'.format(playerId + 1, _extract(player.__cachedstr__))
            Y = self.getBestArmPulls(playerId, envId)
            plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[playerId], marker=markers[playerId], markevery=(playerId / 50., 0.1), lw=2)
        legend()
        plt.xlabel("Time steps $t = 1...T$, horizon $T = {}${}".format(self.horizon, self.signature))
        add_percent_formatter("yaxis", 1.0)
        plt.ylabel("Frequency of pulls of the optimal arm")
        plt.title("Multi-players $M = {}$ : Best arm pulls frequency for each players, averaged ${}$ times\n{} arm{}s: {}".format(self.nbPlayers, self.cfg['repetitions'], self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig, fig=fig, pickleit=PICKLE_IT)
        return fig

    def plotAllPulls(self, envId=0, savefig=None, cumulated=True, normalized=False):
        """Plot the frequency of use of every channels, one figure for each channel. Not so useful."""
        X = self._times - 1
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
                plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=player.__cachedstr__, color=colors[playerId], linestyle='', marker=markers[playerId], markevery=(playerId / 50., 0.1), lw=2)
            legend()
            plt.xlabel("Time steps $t = 1...T$, horizon $T = {}${}".format(self.horizon, self.signature))
            s = ("Normalized " if normalized else "") + ("Cumulated number" if cumulated else "Frequency")
            plt.ylabel("{} of pulls of the arm #{}".format(s, armId + 1))
            plt.title("Multi-players $M = {}$ : {} of pulls of the arm #{} for each players, averaged ${}$ times\n{} arm{}s: {}".format(self.nbPlayers, s.lower(), armId + 1, self.cfg['repetitions'], self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(self.nbPlayers, latex=True)))
            maximizeWindow()
            if savefig is not None:
                savefig = mainfig.replace("allPulls", "allPulls_Arm{}".format(armId + 1))
                print("Saving to", savefig, "...")  # DEBUG
                plt.savefig(savefig, bbox_inches=BBOX_INCHES)
            plt.show() if self.showplot else plt.close()
        return figs

    def plotFreeTransmissions(self, envId=0, savefig=None, cumulated=False):
        """Plot the frequency free transmission."""
        X = self._times - 1
        fig = plt.figure()
        colors = palette(self.nbPlayers)
        for playerId, player in enumerate(self.players):
            Y = self.getfreeTransmissions(playerId, envId)
            if cumulated:
                Y = np.cumsum(Y)
            plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], '.', label=player.__cachedstr__, color=colors[playerId], markersize=1, lw=2)
            # should only plot with markers
        legend()
        plt.xlabel("Time steps $t = 1...T$, horizon $T = {}${}".format(self.horizon, self.signature))
        add_percent_formatter("yaxis", 1.0)
        plt.ylabel("{}ransmission on a free channel".format("Cumulated T" if cumulated else "T"))
        plt.title("Multi-players $M = {}$ : {}free transmission for each players, averaged ${}$ times\n{} arm{}s: {}".format(self.nbPlayers, "Cumulated " if cumulated else "", self.cfg['repetitions'], self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig, fig=fig, pickleit=PICKLE_IT)
        return fig

    # TODO I should plot the evolution of the occupation ratio of each channel, as a function of time
    # Starting from the average occupation (by primary users), as given by [1 - arm.mean], it should increase occupation[arm] when users chose it
    # The reason/idea is that good arms (low occupation ration) are pulled a lot, thus becoming not as available as they seemed

    def plotNbCollisions(self, envId=0, savefig=None,
                         semilogx=False, semilogy=False, loglog=False,
                         cumulated=False, upperbound=False, evaluators=()):
        """Plot the frequency or cum number of collisions, support more than one environments (use evaluators to give a list of other environments)."""
        X = self._times - 1
        fig = plt.figure()
        evaluators = [self] + list(evaluators)  # Default to only [self]
        colors = palette(len(evaluators))
        markers = makemarkers(len(evaluators))
        plot_method = plt.loglog if loglog else plt.plot
        plot_method = plt.semilogy if semilogy else plot_method
        plot_method = plt.semilogx if semilogx else plot_method
        for evaId, eva in enumerate(evaluators):
            Y = np.zeros(eva.horizon)
            for armId in range(eva.envs[envId].nbArms):
                Y += eva.getCollisions(armId, envId)
            if cumulated:
                Y = np.cumsum(Y)
            Y /= eva.nbPlayers  # To normalized the count?
            plot_method(X[::self.delta_t_plot], Y[::self.delta_t_plot], (markers[evaId] + '-') if cumulated else '.', markevery=((evaId / 50., 0.1) if cumulated else None), label=eva.strPlayers(short=True), color=colors[evaId], alpha=1. if cumulated else 0.7, lw=2)
        if not cumulated: add_percent_formatter("yaxis", 1.0)
        # We also plot our lower bound
        if upperbound and cumulated:
            upperboundLog = self.envs[envId].upperbound_collisions(self.nbPlayers, X)
            print("Anandkumar et al. upper bound for the non-cumulated number of collisions is {:.3g} * log(t) here ...".format(upperboundLog[-1]))  # DEBUG
            plot_method(X, upperboundLog, 'k-', label="Anandkumar et al. upper bound", lw=3)
        else:
            print("No upper bound for the non-cumulated number of collisions...")  # DEBUG
        # Start the figure
        plt.xlabel("Time steps $t = 1...T$, horizon $T = {}${}".format(self.horizon, self.signature))
        plt.ylabel("{} of collisions on all arms".format("Cumulated number" if cumulated else "Frequency"))
        legend()
        plt.title("Multi-players $M = {}$ : {}of collisions, averaged ${}$ times\n{} arm{}s: {}".format(self.nbPlayers, "Cumulated number " if cumulated else "Frequency ", self.cfg['repetitions'], self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig, fig=fig, pickleit=PICKLE_IT)
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
        assert 0 <= np.sum(Y) <= 1, "Error: the sum of collisions = {}, averaged by horizon and nbPlayers, cannot be outside of [0, 1] ...".format(np.sum(Y))  # DEBUG
        for armId, arm in enumerate(self.envs[envId].arms):
            labels[armId] = "#${}$: ${}$ (${:.1%}$$\%$)".format(armId, repr(arm), Y[armId])
            print("  - For {},\tfrequency of collisions is {:.5g}  ...".format(labels[armId], Y[armId]))  # DEBUG
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
        plt.xlabel("{}{}".format(self.strPlayers(), self.signature))
        if piechart:
            plt.axis('equal')
            plt.pie(Y, labels=labels, colors=colors, explode=[0.07] * len(Y), startangle=45)
        else:
            if semilogy:  # XXX is it perfectly working?
                Y = np.log10(Y)  # use semilogy scale!
                Y -= np.min(Y)   # project back to [0, oo)
                Y /= np.sum(Y)   # project back to [0, 1)
            for i in range(len(Y)):
                plt.axvspan(i - 0.25, i + 0.25, 0, Y[i], label=labels[i], color=colors[i])
            plt.xticks(np.arange(len(Y)), ["Arm #$%i$" % i for i in range(nbArms)] + ["No collision"])
            plt.ylabel("Frequency of collision, in logarithmic scale" if semilogy else "Frequency of collision")
            if not semilogy:
                add_percent_formatter("yaxis", 1.0)
        legend()
        plt.title("Multi-players $M = {}$ : Frequency of collision for each arm, averaged ${}$ times\n{} arm{}s: {}".format(self.nbPlayers, self.cfg['repetitions'], self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(self.nbPlayers, latex=True)))
        show_and_save(self.showplot, savefig, fig=fig, pickleit=PICKLE_IT)
        return fig

    def printFinalRanking(self, envId=0, verb=True):
        """Compute and print the ranking of the different players."""
        assert 0 < self.averageOn < 1, "Error, the parameter averageOn of a EvaluatorMultiPlayers class has to be in (0, 1) strictly, but is = {} here ...".format(self.averageOn)  # DEBUG
        if verb: print("\nFinal ranking for this environment #{:>2} : {} ...".format(envId, self.strPlayers(latex=False, short=True)))  # DEBUG
        lastY = np.zeros(self.nbPlayers)
        for playerId, player in enumerate(self.players):
            Y = self.getRewards(playerId, envId)
            if self.finalRanksOnAverage:
                lastY[playerId] = np.mean(Y[-int(self.averageOn * self.horizon)])   # get average value during the last averageOn% of the iterations
            else:
                lastY[playerId] = Y[-1]  # get the last value
        # Sort lastY and give ranking
        index_of_sorting = np.argsort(-lastY)  # Get them by INCREASING rewards, not decreasing regrets
        if verb:
            for i, k in enumerate(index_of_sorting):
                player = self.players[k]
                print("- Player #{:>2} / {}, {}\twas ranked\t{} / {} for this simulation (last rewards = {:.5g}).".format(k + 1, self.nbPlayers, _extract(player.__cachedstr__), i + 1, self.nbPlayers, lastY[k]))  # DEBUG
        return lastY, index_of_sorting

    def printFinalRankingAll(self, envId=0, evaluators=()):
        """Compute and print the ranking of the different players."""
        evaluators = [self] + list(evaluators)  # Default to only [self]
        allLastY = np.zeros(len(evaluators))
        for evaId, eva in enumerate(evaluators):
            lastY, _ = eva.printFinalRanking(envId=envId, verb=False)
            allLastY[evaId] = np.sum(lastY)
        # Sort allLastY and give ranking
        index_of_sorting = np.argsort(-allLastY)  # Get them by INCREASING rewards, not decreasing regrets
        for i, k in enumerate(index_of_sorting):
            print("- Group of players #{:>2} / {}, {}\twas ranked\t{} / {} for this simulation (last rewards = {:.5g}).".format(k + 1, len(evaluators), evaluators[k].strPlayers(latex=False, short=True), i + 1, len(evaluators), allLastY[k]))  # DEBUG
        return allLastY, index_of_sorting

    def printLastRegrets(self, envId=0, evaluators=(), moreAccurate=None):
        """Print the last regrets of the different evaluators."""
        evaluators = [self] + list(evaluators)  # Default to only [self]
        for evaId, eva in enumerate(evaluators):
            print("\nFor evaluator #{:>2}/{} : {} (players {}) ...".format(1 + evaId, len(evaluators), eva, eva.strPlayers(latex=False, short=True)))
            last_regrets = eva.getLastRegrets(envId=envId, moreAccurate=moreAccurate)
            print("  Last regrets vector (for all repetitions) is:")
            print("Shape of  last regrets R_T =", np.shape(last_regrets))
            print("Min of    last regrets R_T =", np.min(last_regrets))
            print("Mean of   last regrets R_T =", np.mean(last_regrets))
            print("Median of last regrets R_T =", np.median(last_regrets))
            print("Max of    last regrets R_T =", np.max(last_regrets))
            print("STD of    last regrets R_T =", np.std(last_regrets))

    def plotLastRegrets(self, envId=0,
                        normed=False, subplots=True, nbbins=20, log=False,
                        all_on_separate_figures=False, sharex=False, sharey=False,
                        savefig=None, moreAccurate=None,
                        evaluators=()):
        """Plot histogram of the regrets R_T for all evaluators."""
        moreAccurate = moreAccurate if moreAccurate is not None else self.moreAccurate
        if len(evaluators) == 0:  # no need for a subplot
            subplots = False
        evaluators = [self] + list(evaluators)  # Default to only [self]
        N = len(evaluators)
        colors = palette(N)
        if all_on_separate_figures:
            figs = []
            for evaId, eva in enumerate(evaluators):
                fig = plt.figure()
                plt.title("Multi-players $M = {}$ : Histogram of regrets for {}\n${}$ arms{}: {}".format(self.nbPlayers, eva.strPlayers(short=True), self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(self.nbPlayers, latex=True)))
                plt.xlabel("Regret value $R_T$ at the end of simulation, for $T = {}${}".format(self.horizon, self.signature))
                plt.ylabel("{} of observations, ${}$ repetitions".format("Frequency" if normed else "Number", self.repetitions))
                n, returned_bins, patches = plt.hist(eva.getLastRegrets(envId=envId, moreAccurate=moreAccurate), normed=normed, color=colors[evaId], bins=nbbins)
                addTextForWorstCases(plt, n, returned_bins, patches, normed=normed)
                legend()
                show_and_save(self.showplot, None if savefig is None else "{}__Algo_{}_{}".format(savefig, 1 + evaId, 1 + N), fig=fig, pickleit=PICKLE_IT)
                figs.append(fig)
            return figs
        elif subplots:
            nrows, ncols = nrows_ncols(N)
            fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey)
            # now for the figure
            fig.suptitle("Histogram of regrets for different multi-players bandit algorithms\n${}$ arms{}: {}".format(self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(nbPlayers=self.nbPlayers, latex=True)))
            # XXX See https://stackoverflow.com/a/36542971/
            ax0 = fig.add_subplot(111, frame_on=False)  # add a big axes, hide frame
            ax0.grid(False)  # hide grid
            ax0.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')  # hide tick and tick label of the big axes
            # Add only once the ylabel, xlabel, in the middle
            ax0.set_ylabel("{} of observations, ${}$ repetitions".format("Frequency" if normed else "Number", self.repetitions))
            ax0.set_xlabel("Regret value $R_T$ at the end of simulation, for $T = {}${}".format(self.horizon, self.signature))
            # now for the subplots
            for evaId, eva in enumerate(evaluators):
                i, j = evaId % nrows, evaId // nrows
                ax = axes[i, j] if ncols > 1 else axes[i]
                # print("evaId = {}, i = {}, j = {}, nrows = {}, ncols = {}, ax = {} ...".format(evaId, i, j, nrows, ncols, ax))  # DEBUG
                last_regrets = eva.getLastRegrets(envId=envId, moreAccurate=moreAccurate)
                n, returned_bins, patches = ax.hist(last_regrets, normed=normed, color=colors[evaId], bins=nbbins, log=log)
                addTextForWorstCases(ax, n, returned_bins, patches, normed=normed)
                ax.vlines(np.mean(last_regrets), 0, min(np.max(n), self.repetitions))  # display mean regret on a vertical line
                ax.set_title(eva.strPlayers(short=True), fontdict={'fontsize': 'small'})  # XXX one of x-large, medium, small, None, xx-large, x-small, xx-small, smaller, larger, large
                ax.tick_params(axis='both', labelsize=10)  # XXX https://stackoverflow.com/a/11386056/
        else:
            fig = plt.figure()
            plt.title("Multi-players $M = {}$ : Histogram of regrets for different bandit algorithms\n${}$ arms{}: {}".format(self.nbPlayers, self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(self.nbPlayers, latex=True)))
            plt.xlabel("Regret value $R_T$ at the end of simulation, for $T = {}${}".format(self.horizon, self.signature))
            plt.ylabel("{} of observations, ${}$ repetitions".format("Frequency" if normed else "Number", self.repetitions))
            all_last_regrets = []
            labels = []
            for evaId, eva in enumerate(evaluators):
                all_last_regrets.append(eva.getLastRegrets(envId=envId, moreAccurate=moreAccurate))
                labels.append(eva.strPlayers(short=True))
            ns, returned_bins, patchess = plt.hist(all_last_regrets, label=labels, normed=normed, color=colors, bins=nbbins)
            for n, patches in zip(ns, patchess):
                addTextForWorstCases(plt, n, returned_bins, patches, normed=normed)
            legend()
        # Common part
        show_and_save(self.showplot, savefig, fig=fig, pickleit=PICKLE_IT)
        return fig

    def strPlayers(self, short=False, latex=True):
        """Get a string of the players for this environment."""
        listStrPlayers = [_extract(player.__cachedstr__) for player in self.players]
        if len(set(listStrPlayers)) == 1:  # Unique user
            if latex:
                text = r'${} \times$ {}'.format(self.nbPlayers, listStrPlayers[0])
            else:
                text = r'{} x {}'.format(self.nbPlayers, listStrPlayers[0])
        else:
            text = ', '.join(listStrPlayers)
        text = wraptext(text)
        if not short:
            text = '{} players: {}'.format(self.nbPlayers, text)
        return text


def delayed_play(env, players, horizon, collisionModel,
                 seed=None, repeatId=0, count_ranks_markov_chain=False):
    """Helper function for the parallelization."""
    # Give a unique seed to random & numpy.random for each call of this function
    try:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    except (ValueError, SystemError):
        print("Warning: setting random.seed and np.random.seed seems to not be available. Are you using Windows?")  # XXX
    means = env.means
    if env.isDynamic:
        means = env.newRandomArms()
    players = deepcopy(players)
    nbArms = env.nbArms
    nbPlayers = len(players)
    # random_arm_orders = [np.random.permutation(nbArms) for i in range(nbPlayers)]
    # Start game
    for player in players:
        player.startGame()
    # Store results
    result = ResultMultiPlayers(env.nbArms, horizon, nbPlayers, means=means)
    rewards = np.zeros(nbPlayers)
    choices = np.zeros(nbPlayers, dtype=int)
    pulls = np.zeros((nbPlayers, nbArms), dtype=int)
    collisions = np.zeros(nbArms, dtype=int)

    # print the ranks if possible  # DEBUG
    all_players_have_ranks = count_ranks_markov_chain and (repeatId == 0) and all([hasattr(p, 'rank') for p in players])  # DEBUG
    # this will count all the transitions in the Markov chain, to count their empirical probability at the end  # DEBUG
    if all_players_have_ranks:
        markov_chain_transitions = dict()  # DEBUG
        ranks = [p.rank for p in players]
        binranks = tuple(np.bincount(ranks, minlength=nbPlayers + 1)[1:])
        state = binranks

    prettyRange = tqdm(range(horizon), desc="Time t") if repeatId == 0 else range(horizon)
    for t in prettyRange:
        # Reset the array, faster than reallocating them!
        rewards.fill(0)
        pulls.fill(0)
        collisions.fill(0)
        # Every player decides which arm to pull
        for playerId, player in enumerate(players):
            # XXX here, the environment should apply ONCE a random permutation to each player, in order for the non-modified UCB-like algorithms to work fine in case of collisions (their initial exploration phase is non-random hence leading to only collisions in the first steps, and ruining the performance)
            # choices[i] = random_arm_orders[i][player.choice()]
            choices[playerId] = player.choice()
            # # print(" Round t = \t{}, player \t#{:>2}/{} ({}) \tchose : {} ...".format(t, playerId + 1, len(players), player, choices[playerId]))  # DEBUG

        # Then we decide if there is collisions and what to do why them
        # XXX It is here that the player may receive a reward, if there is no collisions
        collisionModel(t, env.arms, players, choices, rewards, pulls, collisions)

        # Finally we store the results
        result.store(t, choices, rewards, pulls, collisions)

        # XXX During the simulation, if using rhoRand or other ranks policy
        if all_players_have_ranks and t > 1:
            ranks = [p.rank for p in players]
            binranks = tuple(np.bincount(ranks, minlength=nbPlayers + 1)[1:])
            # print(" Round t = \t{}, the list of ranks is \t{}\n   and the point of view of ranks it is \t{} ...".format(t, ranks, binranks))  # DEBUG
            previous_state, state = state, binranks
            markov_chain_transitions[(previous_state, state)] = markov_chain_transitions.get((previous_state, state), 0) + 1
            # print("  One more transition from {} to {} ... Currently it was seen {} times ...".format(previous_state, state, markov_chain_transitions[(previous_state, state)]))

    # Print the quality of estimation of arm ranking for this policy, just for 1st repetition
    if repeatId == 0:
        if all_players_have_ranks:
            # At the end, print the information about the markov chain states and transitions
            print("==> Information about the markov chain states:")  # DEBUG
            states = {s1 for (s1, _) in markov_chain_transitions} or {s2 for (_, s2) in markov_chain_transitions}
            states = sorted(list(states))  # sort it, once and for all
            print("    The Markov chain has {:>4} = (2M-1 choose M) differents states ...".format(len(states)))  # DEBUG
            for s in states:
                print("        ", s)
            print("==> Information about the markov chain transitions:")  # DEBUG
            count_states = {}
            for (sum_count_out, s1) in sorted(zip([
                    sum(
                        markov_chain_transitions.get((s11, s3), 0)
                        for s3 in states
                    ) for s11 in states],
                    states)):
                print("\nState s1 = {} was seen {:>6} times ...".format(s1, sum_count_out))  # DEBUG
                count_states[tuple(sorted(s1))] = \
                    count_states.get(tuple(sorted(s1)), 0) + sum_count_out
                for (count, s2) in sorted(zip([
                        markov_chain_transitions.get((s1, s3), 0)
                        for s3 in states],
                        states)):
                    if count > 0:
                        print("    The transition {} --> {} was seen {:>7} times ({:.2%}) ...".format(s1, s2, count, count / float(horizon)))  # DEBUG
                        if sum_count_out > 0:
                            print("        So the estimated proba is {:.3g} ...".format(count / sum_count_out))
            # now from the set point of view
            print("\n\nNow with states just counting the strong partitions of M = {} ...".format(nbPlayers))  # DEBUG
            suniques = list({tuple(sorted(s1)) for s1 in states})
            for (seen, sunique) in sorted(zip(
                    [count_states[s] for s in suniques],
                    suniques)):
                print("    The state {} was seen {:>7} times ({:.2%}) ...".format(sunique, seen, seen / float(horizon)))  # DEBUG
        # DONE for this visualization

        for playerId, player in enumerate(players):
            try:
                order = player.estimatedOrder()
                print("\nEstimated order by the policy {} after {} steps: {} ...".format(player, horizon, order))
                print("  ==> Optimal arm identification: {:.2%} (relative success)...".format(weightedDistance(order, env.means, n=nbPlayers)))
                print("  ==> Manhattan   distance from optimal ordering: {:.2%} (relative success)...".format(manhattan(order)))
                # print("  ==> Kendell Tau distance from optimal ordering: {:.2%} (relative success)...".format(kendalltau(order)))
                # print("  ==> Spearman    distance from optimal ordering: {:.2%} (relative success)...".format(spearmanr(order)))
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
