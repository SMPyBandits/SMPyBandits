# -*- coding: utf-8 -*-
""" Evaluator class to wrap and run the simulations.
Lots of plotting methods, to have various visualizations.
"""
from __future__ import print_function, division

__author__ = "Lilian Besson"
__version__ = "0.7"

# Generic imports
from copy import deepcopy
import random
# Scientific imports
import numpy as np
import matplotlib.pyplot as plt
# import h5py
# Local imports, libraries
from .usejoblib import USE_JOBLIB, Parallel, delayed
from .usetqdm import USE_TQDM, tqdm
# Local imports, tools and config
from .plotsettings import BBOX_INCHES, signature, maximizeWindow, palette, makemarkers, add_percent_formatter, legend, show_and_save
from .sortedDistance import weightedDistance, manhattan, kendalltau, spearmanr, gestalt, meanDistance, sortedDistance
# Local imports, objects and functions
from .MAB import MAB, MarkovianMAB, DynamicMAB
from .Result import Result


REPETITIONS = 1    #: Default nb of repetitions
DELTA_T_SAVE = 1   #: Default sampling rate for saving
DELTA_T_PLOT = 50  #: Default sampling rate for plotting

# Parameters for the random events
random_shuffle = False
random_invert = False
nb_random_events = 5  #: Default nb of random events

# Flag for experimental aspects
STORE_ALL_REWARDS = True       #: Store all rewards?
STORE_ALL_REWARDS = False      #: Store all rewards?
STORE_REWARDS_SQUARED = True   #: Store rewards squared?
STORE_REWARDS_SQUARED = False  #: Store rewards squared?


class Evaluator(object):
    """ Evaluator class to run the simulations."""

    def __init__(self, configuration,
                 finalRanksOnAverage=True, averageOn=5e-3,
                 useJoblibForPolicies=False):
        self.cfg = configuration  #: Configuration dictionnary
        # Attributes
        self.nbPolicies = len(self.cfg['policies'])  #: Number of policies
        print("Number of policies in this comparison:", self.nbPolicies)
        self.horizon = self.cfg['horizon']  #: Horizon (number of time steps)
        print("Time horizon:", self.horizon)
        self.repetitions = self.cfg.get('repetitions', REPETITIONS)  #: Number of repetitions
        print("Number of repetitions:", self.repetitions)
        self.delta_t_save = self.cfg.get('delta_t_save', DELTA_T_SAVE)  #: Sampling rate for saving
        print("Sampling rate for saving, delta_t_save:", self.delta_t_save)
        self.delta_t_plot = 1 if self.horizon <= 10000 else self.cfg.get('delta_t_plot', DELTA_T_PLOT)  #: Sampling rate for plotting
        print("Sampling rate for plotting, delta_t_plot:", self.delta_t_plot)
        self.duration = int(self.horizon / self.delta_t_save)
        print("Number of jobs for parallelization:", self.cfg['n_jobs'])
        # Parameters for the random events
        self.random_shuffle = self.cfg.get('random_shuffle', random_shuffle)  #: Random shuffling of arms?
        self.random_invert = self.cfg.get('random_invert', random_invert)  #: Random inversion of arms?
        self.nb_random_events = self.cfg.get('nb_random_events', nb_random_events)  #: How many random events?
        self.signature = signature
        if self.nb_random_events > 0:
            if self.random_shuffle:
                self.signature = (r", $\Upsilon={}$ random arms shuffling".format(self.nb_random_events - 1)) + self.signature
            elif self.random_invert:
                self.signature = (r", $\Upsilon={}$ arms inversion".format(self.nb_random_events - 1)) + self.signature
        # Flags
        self.finalRanksOnAverage = finalRanksOnAverage  #: Final display of ranks are done on average rewards?
        self.averageOn = averageOn  #: How many last steps for final rank average rewards
        self.useJoblibForPolicies = useJoblibForPolicies  #: Use joblib to parallelize for loop on policies (useless)
        self.useJoblib = USE_JOBLIB and self.cfg['n_jobs'] != 1  #: Use joblib to parallelize for loop on repetitions (useful)
        self.cache_rewards = self.cfg.get('cache_rewards', False)  #: Should we cache and precompute rewards
        self.showplot = self.cfg.get('showplot', True)  #: Show the plot (interactive display or not)

        # Internal object memory
        self.envs = []  #: List of environments
        self.policies = []  #: List of policies
        self.__initEnvironments__()

        # Internal vectorial memory
        self.rewards = np.zeros((self.nbPolicies, len(self.envs), self.duration))  #: For each env, history of rewards, ie accumulated rewards
        self.last_cum_rewards = np.zeros((self.nbPolicies, len(self.envs), self.repetitions))  #: For each env, last accumulated rewards, to compute variance and histogram of whole regret R_T
        self.minCumRewards = np.inf + np.zeros((self.nbPolicies, len(self.envs), self.duration))  #: For each env, history of minimum of rewards, to compute amplitude (+- STD)
        self.maxCumRewards = -np.inf + np.zeros((self.nbPolicies, len(self.envs), self.duration))  #: For each env, history of maximum of rewards, to compute amplitude (+- STD)

        if STORE_REWARDS_SQUARED:
            self.rewardsSquared = np.zeros((self.nbPolicies, len(self.envs), self.duration))  #: For each env, history of rewards squared
        if STORE_ALL_REWARDS:
            self.allRewards = np.zeros((self.nbPolicies, len(self.envs), self.duration, self.repetitions))  #: For each env, full history of rewards

        self.BestArmPulls = dict()  #: For each env, keep the history of best arm pulls
        self.pulls = dict()  #: For each env, keep the history of best arm pulls
        for env in range(len(self.envs)):
            self.BestArmPulls[env] = np.zeros((self.nbPolicies, self.duration))
            self.pulls[env] = np.zeros((self.nbPolicies, self.envs[env].nbArms))
        print("Number of environments to try:", len(self.envs))
        # To speed up plotting
        self.times = np.arange(1, 1 + self.horizon, self.delta_t_save)

    # --- Init methods

    def __initEnvironments__(self):
        """ Create environments."""
        for configuration_arms in self.cfg['environment']:
            if isinstance(configuration_arms, dict) \
               and "arm_type" in configuration_arms and "params" in configuration_arms \
               and "function" in configuration_arms["params"] and "args" in configuration_arms["params"]:
                self.envs.append(DynamicMAB(configuration_arms))
            elif isinstance(configuration_arms, dict) \
               and "arm_type" in configuration_arms and configuration_arms["arm_type"] == "Markovian" \
               and "params" in configuration_arms \
               and "transitions" in configuration_arms["params"]:
                self.envs.append(MarkovianMAB(configuration_arms))
            else:
                self.envs.append(MAB(configuration_arms))

    def __initPolicies__(self, env):
        """ Create or initialize policies."""
        for policyId, policy in enumerate(self.cfg['policies']):
            print("- Adding policy #{} = {} ...".format(policyId + 1, policy))  # DEBUG
            if isinstance(policy, dict):
                print("  Creating this policy from a dictionnary 'self.cfg['policies'][{}]' = {} ...".format(policyId, policy))  # DEBUG
                self.policies.append(policy['archtype'](env.nbArms, **policy['params']))
            else:
                print("  Using this already created policy 'self.cfg['policies'][{}]' = {} ...".format(policyId, policy))  # DEBUG
                self.policies.append(policy)

    # --- Start computation

    def compute_cache_rewards(self, arms):
        """ Compute only once the rewards, then launch the experiments with the same matrix (r_{k,t})."""
        rewards = np.zeros((len(arms), self.repetitions, self.horizon))
        print("\n===> Pre-computing the rewards ... Of shape {} ...\n    In order for all simulated algorithms to face the same random rewards (robust comparison of A1,..,An vs Aggr(A1,..,An)) ...\n".format(np.shape(rewards)))  # DEBUG
        for armId, arm in tqdm(enumerate(arms), desc="Arms"):
            if hasattr(arm, 'draw_nparray'):  # XXX Use this method to speed up computation
                rewards[armId] = arm.draw_nparray((self.repetitions, self.horizon))
            else:  # Slower
                for repeatId in tqdm(range(self.repetitions), desc="Repetitions"):
                    for t in tqdm(range(self.horizon), desc="Time steps"):
                        rewards[armId, repeatId, t] = arm.draw(t)
        return rewards

    def startAllEnv(self):
        """Simulate all envs."""
        for envId, env in enumerate(self.envs):
            self.startOneEnv(envId, env)

    def startOneEnv(self, envId, env):
        """Simulate that env."""
        print("\n\nEvaluating environment:", repr(env))
        self.policies = []
        self.__initPolicies__(env)
        # Precompute rewards
        if self.cache_rewards:
            allrewards = self.compute_cache_rewards(env.arms)
        else:
            allrewards = None

        def store(r, policyId, repeatId):
            """ Store the result of the #repeatId experiment, for the #policyId policy."""
            self.rewards[policyId, envId, :] += r.rewards
            self.last_cum_rewards[policyId, envId, repeatId] = np.sum(r.rewards)
            if hasattr(self, 'rewardsSquared'):
                self.rewardsSquared[policyId, envId, :] += (r.rewards ** 2)
            if hasattr(self, 'allRewards'):
                self.allRewards[policyId, envId, :, repeatId] = r.rewards
            if hasattr(self, 'minCumRewards'):
                self.minCumRewards[policyId, envId, :] = np.minimum(self.minCumRewards[policyId, envId, :], np.cumsum(r.rewards)) if repeatId > 1 else np.cumsum(r.rewards)
            if hasattr(self, 'maxCumRewards'):
                self.maxCumRewards[policyId, envId, :] = np.maximum(self.maxCumRewards[policyId, envId, :], np.cumsum(r.rewards)) if repeatId > 1 else np.cumsum(r.rewards)
            self.BestArmPulls[envId][policyId, :] += np.cumsum(np.in1d(r.choices, r.indexes_bestarm))
            self.pulls[envId][policyId, :] += r.pulls

        # Start for all policies
        for policyId, policy in enumerate(self.policies):
            print("\n\n\n- Evaluating policy #{}/{}: {} ...".format(policyId + 1, self.nbPolicies, policy))
            if self.useJoblib:
                seeds = np.random.randint(low=0, high=100 * self.repetitions, size=self.repetitions)
                repeatIdout = 0
                for r in Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
                    delayed(delayed_play)(env, policy, self.horizon, random_shuffle=self.random_shuffle, random_invert=self.random_invert, nb_random_events=self.nb_random_events, delta_t_save=self.delta_t_save, allrewards=allrewards, seed=seeds[repeatId], repeatId=repeatId)
                    for repeatId in tqdm(range(self.repetitions), desc="Repeat||")
                ):
                    store(r, policyId, repeatIdout)
                    repeatIdout += 1
            else:
                for repeatId in tqdm(range(self.repetitions), desc="Repeat"):
                    r = delayed_play(env, policy, self.horizon, random_shuffle=self.random_shuffle, random_invert=self.random_invert, nb_random_events=self.nb_random_events, delta_t_save=self.delta_t_save, allrewards=allrewards, repeatId=repeatId)
                    store(r, policyId, repeatId)

    # --- Save to disk methods

    def saveondisk(self, filepath='/tmp/saveondiskEvaluator.hdf5'):
        """ Save the content of the internal date to into a HDF5 file on the disk."""
        # 1) create the shape of what will be store
        # FIXME write it !
        # 2) store it
        with open(filepath, 'r') as hdf:
            hdf.configuration = self.hdf_configuration
            hdf.rewards = self.rewards
            try:
                hdf.minCumRewards = self.minCumRewards
            except (TypeError, AttributeError, KeyError):
                pass
            try:
                hdf.maxCumRewards = self.maxCumRewards
            except (TypeError, AttributeError, KeyError):
                pass
            try:
                hdf.rewardsSquared = self.rewardsSquared
            except (TypeError, AttributeError, KeyError):
                pass
            try:
                hdf.allRewards = self.allRewards
            except (TypeError, AttributeError, KeyError):
                pass
            hdf.BestArmPulls = self.BestArmPulls
            hdf.pulls = self.pulls
        raise ValueError("FIXME finish to write this function saveondisk() for Evaluator!")

    def loadfromdisk(self, hdf, delta_t_save=None, useConfig=False):
        """ Update internal memory of the Evaluator object by loading data the opened HDF5 file."""
        # FIXME I just have to fill all the internal matrices from the HDF5 file ?
        # 1) load configuration
        if useConfig:
            self.__init__(hdf.configuration)
        # 2) load internal matrix memory
        self.rewards = hdf.rewards
        try:
            self.minCumRewards = hdf.minCumRewards
        except (TypeError, AttributeError, KeyError):
            pass
        try:
            self.maxCumRewards = hdf.maxCumRewards
        except (TypeError, AttributeError, KeyError):
            pass
        try:
            self.rewardsSquared = hdf.rewardsSquared
        except (TypeError, AttributeError, KeyError):
            pass
        try:
            self.allRewards = hdf.allRewards
        except (TypeError, AttributeError, KeyError):
            pass
        self.BestArmPulls = hdf.BestArmPulls
        self.pulls = hdf.pulls
        raise ValueError("FIXME finish to write this function loadfromdisk() for Evaluator!")

    def getPulls(self, policyId, envId=0):
        """Extract mean pulls."""
        return self.pulls[envId][policyId, :] / float(self.repetitions)

    def getBestArmPulls(self, policyId, envId=0):
        """Extract mean best arm pulls."""
        # We have to divide by a arange() = cumsum(ones) to get a frequency
        return self.BestArmPulls[envId][policyId, :] / (float(self.repetitions) * self.times)

    def getRewards(self, policyId, envId=0):
        """Extract mean rewards."""
        return self.rewards[policyId, envId, :] / float(self.repetitions)

    def getMaxRewards(self, envId=0):
        """Extract max mean rewards."""
        return np.max(self.rewards[:, envId, :] / float(self.repetitions))

    def getCumulatedRegret(self, policyId, envId=0):
        """Compute cumulative regret."""
        # return self.times * self.envs[envId].maxArm - np.cumsum(self.getRewards(policyId, envId))
        return np.cumsum(self.envs[envId].maxArm - self.getRewards(policyId, envId))

    def getLastRegrets(self, policyId, envId=0):
        """Extract last regrets."""
        return self.horizon * self.envs[envId].maxArm - self.last_cum_rewards[policyId, envId, :]

    def getAverageRewards(self, policyId, envId=0):
        """Extract mean rewards (not `rewards` but `cumsum(rewards)/cumsum(1)`."""
        return np.cumsum(self.getRewards(policyId, envId)) / self.times

    def getRewardsSquared(self, policyId, envId=0):
        """Extract rewards squared."""
        return self.rewardsSquared[policyId, envId, :] / float(self.repetitions)

    def getSTDRegret(self, policyId, envId=0, meanRegret=False):
        """Extract standard deviation of rewards. FIXME experimental! """
        # X = self.times
        # YMAX = self.getMaxRewards(envId=envId)
        # Y = self.getRewards(policyId, envId)
        # Y2 = self.getRewardsSquared(policyId, envId)
        # if meanRegret:  # Cumulated expectation on time
        #     Ycum2 = (np.cumsum(Y) / X)**2
        #     Y2cum = np.cumsum(Y2) / X
        #     assert np.all(Y2cum >= Ycum2), "Error: getSTDRegret found a nan value in the standard deviation (ie a point where Y2cum < Ycum2)."  # DEBUG
        #     stdY = np.sqrt(Y2cum - Ycum2)
        #     YMAX *= 20  # XXX make it look smaller, for the plots
        # else:  # Expectation on nb of repetitions
        #     # https://en.wikipedia.org/wiki/Algebraic_formula_for_the_variance#In_terms_of_raw_moments
        #     # std(Y) = sqrt( E[Y**2] - E[Y]**2 )
        #     # stdY = np.cumsum(np.sqrt(Y2 - Y**2))
        #     stdY = np.sqrt(Y2 - Y**2)
        #     YMAX *= np.log(2 + self.duration)  # Normalize the std variation
        #     YMAX *= 50  # XXX make it look larger, for the plots
        # # Renormalize this standard deviation
        # # stdY /= YMAX
        allRewards = self.allRewards[policyId, envId, :, :]
        return np.std(np.cumsum(allRewards, axis=0), axis=1)

    def getMaxMinReward(self, policyId, envId=0):
        """Extract amplitude of rewards as maxCumRewards - minCumRewards."""
        return (self.maxCumRewards[policyId, envId, :] - self.minCumRewards[policyId, envId, :]) / (float(self.repetitions) ** 0.5)
        # return self.maxCumRewards[policyId, envId, :] - self.minCumRewards[policyId, envId, :]

    # --- Plotting methods

    def plotRegrets(self, envId,
                    savefig=None, meanRegret=False,
                    plotSTD=False, plotMaxMin=False,
                    semilogx=False, semilogy=False, loglog=False,
                    normalizedRegret=False, drawUpperBound=False,
                    ):
        """Plot the centralized cumulated regret, support more than one environments (use evaluators to give a list of other environments). """
        fig = plt.figure()
        ymin = 0
        colors = palette(self.nbPolicies)
        markers = makemarkers(self.nbPolicies)
        X = self.times - 1
        plot_method = plt.loglog if loglog else plt.plot
        plot_method = plt.semilogy if semilogy else plot_method
        plot_method = plt.semilogx if semilogx else plot_method
        for i, policy in enumerate(self.policies):
            if meanRegret:
                Y = self.getAverageRewards(i, envId)
            else:
                Y = self.getCumulatedRegret(i, envId)
                if normalizedRegret:
                    Y /= np.log(2 + X)   # XXX prevent /0
            ymin = min(ymin, np.min(Y))
            lw = 4 if ('$N=' in str(policy) or 'Aggr' in str(policy) or 'CORRAL' in str(policy) or 'LearnExp' in str(policy) or 'Exp4' in str(policy)) else 2
            if semilogx:
                # FIXED for semilogx plots, truncate to only show t >= 100
                plt.semilogx(X[X >= 100][::self.delta_t_plot], Y[X >= 100][::self.delta_t_plot], label=str(policy), color=colors[i], marker=markers[i], markevery=(i / 50., 0.1), lw=lw)
            else:
                plot_method(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=str(policy), color=colors[i], marker=markers[i], markevery=(i / 50., 0.1), lw=lw)
            # Print standard deviation of regret
            if plotSTD and self.repetitions > 1:
                stdY = self.getSTDRegret(i, envId, meanRegret=meanRegret)
                if normalizedRegret:
                    stdY /= np.log(2 + X)
                plt.fill_between(X[::self.delta_t_plot], Y[::self.delta_t_plot] - stdY[::self.delta_t_plot], Y[::self.delta_t_plot] + stdY[::self.delta_t_plot], facecolor=colors[i], alpha=0.2)
            # Print amplitude of regret
            if plotMaxMin and self.repetitions > 1:
                MaxMinY = self.getMaxMinReward(i, envId) / 2.
                if normalizedRegret:
                    MaxMinY /= np.log(2 + X)
                plt.fill_between(X[::self.delta_t_plot], Y[::self.delta_t_plot] - MaxMinY[::self.delta_t_plot], Y[::self.delta_t_plot] + MaxMinY[::self.delta_t_plot], facecolor=colors[i], alpha=0.2)
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}${}".format(self.horizon, self.signature))
        lowerbound = self.envs[envId].lowerbound()
        print("\nThis MAB problem has: \n - a [Lai & Robbins] complexity constant C(mu) = {:.3g} for 1-player problem... \n - a Optimal Arm Identification factor H_OI(mu) = {:.2%} ...".format(self.envs[envId].lowerbound(), self.envs[envId].hoifactor()))  # DEBUG
        if not meanRegret:
            plt.ylim(ymin, plt.ylim()[1])
        # Get a small string to add to ylabel
        ylabel2 = r"%s%s" % (r", $\pm 1$ standard deviation" if (plotSTD and not plotMaxMin) else "", r", $\pm 1$ amplitude" if (plotMaxMin and not plotSTD) else "")
        if meanRegret:
            # We plot a horizontal line ----- at the best arm mean
            plt.plot(X[::self.delta_t_plot], self.envs[envId].maxArm * np.ones_like(X)[::self.delta_t_plot], 'k--', label="Mean of the best arm = ${:.3g}$".format(self.envs[envId].maxArm))
            legend()
            plt.ylabel(r"Mean reward, average on time $\tilde{r}_t = \frac{1}{t} \sum_{s = 1}^{t} \mathbb{E}_{%d}[r_s]$%s" % (self.repetitions, ylabel2))
            if not self.envs[envId].isDynamic:
                plt.ylim(1.06 * self.envs[envId].minArm, 1.06 * self.envs[envId].maxArm)
            plt.title("Mean rewards for different bandit algorithms, averaged ${}$ times\n${}$ arms{}: {}".format(self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(1, latex=True)))
        elif normalizedRegret:
            # We also plot the Lai & Robbins lower bound
            plt.plot(X[::self.delta_t_plot], lowerbound * np.ones_like(X)[::self.delta_t_plot], 'k-', label="Lai & Robbins lower bound = ${:.3g}$".format(lowerbound), lw=3)
            legend()
            plt.ylabel(r"Normalized cumulated regret $\frac{R_t}{\log t} = \frac{t}{\log t} \mu^* - \frac{1}{\log t}\sum_{s = 1}^{t} \mathbb{E}_{%d}[r_s]$%s" % (self.repetitions, ylabel2))
            plt.title("Normalized cumulated regrets for different bandit algorithms, averaged ${}$ times\n${}$ arms{}: {}".format(self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(1, latex=True)))
        else:
            if drawUpperBound and not semilogx:
                # Experiment to print also an upper bound: it is CRAZILY huge!!
                lower_amplitudes = np.asarray([arm.lower_amplitude for arm in self.envs[envId].arms])
                amplitude = np.max(lower_amplitudes[:, 1])
                maxVariance = max([p * (1 - p) for p in self.envs[envId].means])
                K = self.envs[envId].nbArms
                upperbound = 76 * np.sqrt(maxVariance * K * X) + amplitude * K
                plt.plot(X[::self.delta_t_plot], upperbound[::self.delta_t_plot], 'r-', label=r"Minimax upper-bound for kl-UCB++", lw=3)
            # FIXED for semilogx plots, truncate to only show t >= 100
            if semilogx:
                X = X[X >= 100]
            # We also plot the Lai & Robbins lower bound
            plt.plot(X[::self.delta_t_plot], lowerbound * np.log(1 + X)[::self.delta_t_plot], 'k-', label=r"Lai & Robbins lower bound = ${:.3g}\; \log(T)$".format(lowerbound), lw=3)
            legend()
            plt.ylabel(r"Cumulated regret $R_t = t \mu^* - \sum_{s = 1}^{t} \mathbb{E}_{%d}[r_s]$%s" % (self.repetitions, ylabel2))
            plt.title("Cumulated regrets for different bandit algorithms, averaged ${}$ times\n${}$ arms{}: {}".format(self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(1, latex=True)))
        show_and_save(self.showplot, savefig)
        return fig

    def plotBestArmPulls(self, envId, savefig=None):
        """Plot the frequency of pulls of the best channel.

        - Warning: does not adapt to dynamic settings!
        """
        fig = plt.figure()
        colors = palette(self.nbPolicies)
        markers = makemarkers(self.nbPolicies)
        X = self.times[2:]
        for i, policy in enumerate(self.policies):
            Y = self.getBestArmPulls(i, envId)[2:]
            lw = 4 if ('$N=' in str(policy) or 'Aggr' in str(policy) or 'CORRAL' in str(policy) or 'LearnExp' in str(policy) or 'Exp4' in str(policy)) else 2
            plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=str(policy), color=colors[i], marker=markers[i], markevery=(i / 50., 0.1), lw=lw)
        legend()
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}${}".format(self.horizon, self.signature))
        # plt.ylim(-0.03, 1.03)  # Don't force to view on [0%, 100%]
        add_percent_formatter("yaxis", 1.0)
        plt.ylabel(r"Frequency of pulls of the optimal arm")
        plt.title("Best arm pulls frequency for different bandit algorithms, averaged ${}$ times\n${}$ arms{}: {}".format(self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(1, latex=True)))
        show_and_save(self.showplot, savefig)
        return fig

    def printFinalRanking(self, envId=0):
        """Print the final ranking of the different policies."""
        assert 0 < self.averageOn < 1, "Error, the parameter averageOn of a EvaluatorMultiPlayers classs has to be in (0, 1) strictly, but is = {} here ...".format(self.averageOn)  # DEBUG
        print("\nFinal ranking for this environment #{} :".format(envId))
        nbPolicies = self.nbPolicies
        lastY = np.zeros(nbPolicies)
        for i, policy in enumerate(self.policies):
            Y = self.getCumulatedRegret(i, envId)
            if self.finalRanksOnAverage:
                lastY[i] = np.mean(Y[-int(self.averageOn * self.duration)])   # get average value during the last 0.5% of the iterations
            else:
                lastY[i] = Y[-1]  # get the last value
        # Sort lastY and give ranking
        index_of_sorting = np.argsort(lastY)
        for i, k in enumerate(index_of_sorting):
            policy = self.policies[k]
            print("- Policy '{}'\twas ranked\t{} / {} for this simulation (last regret = {:.5g}).".format(str(policy), i + 1, nbPolicies, lastY[k]))
        return lastY, index_of_sorting

    def printLastRegrets(self, envId=0):
        """Print the last regrets of the different policies."""
        for policyId, policy in enumerate(self.policies):
            print("\n  For policy #{} called '{}' ...".format(policyId, policy))
            last_regrets = self.getLastRegrets(policyId, envId=envId)
            print("  Last regrets vector (for all repetitions) is:")
            # print(last_regrets)  # XXX takes too much printing
            print("Mean of   last regret R_T =", np.mean(last_regrets))
            print("Median of last regret R_T =", np.median(last_regrets))
            print("VAR of    last regret R_T =", np.var(last_regrets))

    def plotLastRegrets(self, envId=0, normed=False, subplots=True, bins=None, savefig=None):
        """Plot histogram of the regrets R_T for all policies."""
        colors = palette(self.nbPolicies)
        if subplots:
            # Use a subplots of the good size
            N = self.nbPolicies
            nrows = 1 + int(np.sqrt(N))
            ncols = N // nrows
            if N > nrows * ncols:
                ncols += 1
            nrows, ncols = max(nrows, ncols), min(nrows, ncols)
            fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)
            fig.suptitle("Histogram of regrets for different bandit algorithms\n${}$ arms{}: {}".format(self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(1, latex=True)))
            for policyId, policy in enumerate(self.policies):
                i, j = policyId % nrows, policyId // nrows
                last_regrets = self.getLastRegrets(policyId, envId=envId)
                n, _, _ = axes[i, j].hist(last_regrets, normed=normed, color=colors[policyId], bins=bins)
                axes[i, j].vlines(np.mean(last_regrets), 0, min(np.max(n), self.repetitions))  # display mean regret on a vertical line
                axes[i, j].set_title(str(policy))
                # Add only once the ylabel, xlabel, in the middle
                if i == (nrows // 2) and j == 0:
                    axes[i, j].set_ylabel("Number of observations, ${}$ repetitions".format(self.repetitions))
                if i == nrows - 1 and j == (ncols // 2):
                    axes[i, j].set_xlabel("Regret value $R_T$ at the end of simulation, for $T = {}${}".format(self.horizon, self.signature))
        else:
            fig = plt.figure()
            plt.title("Histogram of regrets for different bandit algorithms\n${}$ arms{}: {}".format(self.envs[envId].nbArms, self.envs[envId].str_sparsity(), self.envs[envId].reprarms(1, latex=True)))
            plt.xlabel("Regret value $R_T$ at the end of simulation, for $T = {}${}".format(self.horizon, self.signature))
            plt.ylabel("Number of observations, ${}$ repetitions".format(self.repetitions))
            all_last_regrets = []
            labels = []
            for policyId, policy in enumerate(self.policies):
                all_last_regrets.append(self.getLastRegrets(policyId, envId=envId))
                labels.append(str(policy))
            plt.hist(all_last_regrets, label=labels, normed=normed, color=colors, bins=bins)
            legend()
        # Common part
        show_and_save(self.showplot, savefig)
        return fig


# Helper function for the parallelization

def delayed_play(env, policy, horizon, delta_t_save=1,
                 random_shuffle=random_shuffle, random_invert=random_invert, nb_random_events=nb_random_events,
                 seed=None, allrewards=None, repeatId=0):
    """Helper function for the parallelization."""
    # Give a unique seed to random & numpy.random for each call of this function
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    except (ValueError, SystemError):
        print("Warning: setting random.seed and np.random.seed seems to not be available. Are you using Windows?")  # XXX
    # We have to deepcopy because this function is Parallel-ized
    if random_shuffle or random_invert:
        env = deepcopy(env)    # XXX this uses a LOT of RAM memory!!!
    if env.isDynamic:  # FIXME
        env.newRandomArms()
    policy = deepcopy(policy)  # XXX this uses a LOT of RAM memory!!!

    indexes_bestarm = np.nonzero(np.isclose(env.means, env.maxArm))[0]

    # Start game
    policy.startGame()
    result = Result(env.nbArms, horizon, indexes_bestarm=indexes_bestarm)  # One Result object, for every policy
    # , delta_t_save=delta_t_save

    # XXX Experimental support for random events: shuffling or inverting the list of arms, at these time steps
    t_events = [i * int(horizon / float(nb_random_events)) for i in range(nb_random_events)]
    if nb_random_events is None or nb_random_events <= 0:
        random_shuffle = False
        random_invert = False

    prettyRange = tqdm(range(horizon), desc="Time t") if repeatId == 0 else range(horizon)
    for t in prettyRange:
        choice = policy.choice()

        # XXX do this quicker!?
        if allrewards is None:
            reward = env.draw(choice, t)
        else:
            reward = allrewards[choice, repeatId, t]

        policy.getReward(choice, reward)

        # if t % delta_t_save == 0:  # XXX inefficient and does not work yet
        #     # if delta_t_save > 1: print("t =", t, "delta_t_save =", delta_t_save, " : saving ...")  # DEBUG
        #     result.store(t, choice, reward)
        result.store(t, choice, reward)

        # XXX Experimental : shuffle the arms at the middle of the simulation
        if random_shuffle and t in t_events:
                indexes_bestarm = env.new_order_of_arm(shuffled(env.arms))
                result.change_in_arms(t, indexes_bestarm)
                if repeatId == 0:
                    print("\nShuffling the arms at time t = {} ...".format(t))  # DEBUG
        # XXX Experimental : invert the order of the arms at the middle of the simulation
        if random_invert and t in t_events:
                indexes_bestarm = env.new_order_of_arm(env.arms[::-1])
                result.change_in_arms(t, indexes_bestarm)
                if repeatId == 0:
                    print("\nInverting the order of the arms at time t = {} ...".format(t))  # DEBUG

    # Print the quality of estimation of arm ranking for this policy, just for 1st repetition
    if repeatId == 0 and hasattr(policy, 'estimatedOrder'):
        order = policy.estimatedOrder()
        print("\nEstimated order by the policy {} after {} steps: {} ...".format(policy, horizon, order))
        print("  ==> Optimal arm identification: {:.2%} (relative success)...".format(weightedDistance(order, env.means, n=1)))
        print("  ==> Manhattan   distance from optimal ordering: {:.2%} (relative success)...".format(manhattan(order)))
        print("  ==> Kendell Tau distance from optimal ordering: {:.2%} (relative success)...".format(kendalltau(order)))
        print("  ==> Spearman    distance from optimal ordering: {:.2%} (relative success)...".format(spearmanr(order)))
        print("  ==> Gestalt     distance from optimal ordering: {:.2%} (relative success)...".format(gestalt(order)))
        print("  ==> Mean distance from optimal ordering: {:.2%} (relative success)...".format(meanDistance(order)))
    return result


# --- Helper for loading a previous Evaluator object

def EvaluatorFromDisk(filepath='/tmp/saveondiskEvaluator.hdf5'):
    """ Create a new Evaluator object from the HDF5 file given in argument."""
    with open(filepath, 'r') as hdf:
        configuration = hdf.configuration
        evaluator = Evaluator(configuration)
        evaluator.loadfromdisk(hdf)
    return evaluator


# --- Utility function

from random import shuffle
from copy import copy


def shuffled(mylist):
    """Returns a shuffled version of the input 1D list. sorted() exists instead of list.sort(), but shuffled() does not exist instead of random.shuffle()...

    >>> from random import seed; seed(1234)  # reproducible results
    >>> mylist = [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9]
    >>> shuffled(mylist)
    [0.9, 0.4, 0.3, 0.6, 0.5, 0.7, 0.1, 0.2, 0.8]
    >>> shuffled(mylist)
    [0.4, 0.3, 0.7, 0.5, 0.8, 0.1, 0.9, 0.6, 0.2]
    >>> shuffled(mylist)
    [0.4, 0.6, 0.9, 0.5, 0.7, 0.2, 0.1, 0.3, 0.8]
    >>> shuffled(mylist)
    [0.8, 0.7, 0.3, 0.1, 0.9, 0.5, 0.6, 0.2, 0.4]
    """
    copiedlist = copy(mylist)
    shuffle(copiedlist)
    return copiedlist
