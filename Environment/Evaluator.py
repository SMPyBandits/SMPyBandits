# -*- coding: utf-8 -*-
""" Evaluator class to wrap and run the simulations."""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.2"

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
from .Result import Result
from .MAB import MAB


# Parameters for the random events
random_shuffle = False
random_invert = False
nb_random_events = 4

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


class Evaluator:
    """ Evaluator class to run the simulations."""

    def __init__(self, configuration,
                 finalRanksOnAverage=True, averageOn=5e-3,
                 useJoblibForPolicies=False):
        # Configuration
        self.cfg = configuration
        # Flags
        self.finalRanksOnAverage = finalRanksOnAverage
        self.averageOn = averageOn
        self.useJoblibForPolicies = useJoblibForPolicies
        self.useJoblib = USE_JOBLIB and self.cfg['n_jobs'] != 1
        # Internal object memory
        self.envs = []
        self.policies = []
        self.__initEnvironments__()
        # Internal vectorial memory
        self.rewards = np.zeros((len(self.cfg['policies']),
                                 len(self.envs), self.cfg['horizon']))
        self.bestArmPulls = dict()
        self.pulls = dict()
        for env in range(len(self.envs)):
            self.bestArmPulls[env] = np.zeros((len(self.cfg['policies']), self.cfg['horizon']))
            self.pulls[env] = np.zeros((len(self.cfg['policies']), self.envs[env].nbArms))
        print("Number of algorithms to compare:", len(self.cfg['policies']))
        print("Number of environments to try:", len(self.envs))
        print("Time horizon:", self.cfg['horizon'])
        print("Number of repetitions:", self.cfg['repetitions'])

    def __initEnvironments__(self):
        for armType in self.cfg['environment']:
            self.envs.append(MAB(armType))

    def __initPolicies__(self, env):
        for polId, policy in enumerate(self.cfg['policies']):
            print("- Adding policy #{} = {} ...".format(polId + 1, policy))  # DEBUG
            self.policies.append(policy['archtype'](env.nbArms,
                                                    **policy['params']))

    def start_all_env(self):
        for envId, env in enumerate(self.envs):
            self.start_one_env(envId, env)

    # @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof
    def start_one_env(self, envId, env):
        print("\nEvaluating environment:", repr(env))
        self.policies = []
        self.__initPolicies__(env)
        # if self.useJoblibForPolicies:
        #     n_jobs = len(self.policies)
        #     joblib.Parallel(n_jobs=n_jobs, verbose=self.cfg['verbosity'])(
        #         joblib.delayed(delayed_start)(self, env, policy, polId, envId)
        #         for polId, policy in enumerate(self.policies)
        #     )
        # else:
        #     for polId, policy in enumerate(self.policies):
        #         delayed_start(self, env, policy, polId, envId)
        # # FIXED I tried to also parallelize this loop on policies, of course it does give any speedup
        for polId, policy in enumerate(self.policies):
            print("\n- Evaluating policy #{}/{}: {} ...".format(polId + 1, len(self.policies), policy))
            if self.useJoblib:
                results = joblib.Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
                    joblib.delayed(delayed_play)(env, policy, self.cfg['horizon'])
                    for _ in range(self.cfg['repetitions'])
                    # , random_shuffle=self.get(['random_shuffle'], None), random_invert=self.get(['random_invert'], None), nb_random_events=self.get(['nb_random_events'], 0)
                )
            else:
                results = []
                for _ in range(self.cfg['repetitions']):
                    r = delayed_play(env, policy, self.cfg['horizon'])
                    # , random_shuffle=self.get(['random_shuffle'], None), random_invert=self.get(['random_invert'], None), nb_random_events=self.get(['nb_random_events'], 0)
                    results.append(r)
            # Get the position of the best arms
            env = self.envs[envId]
            means = np.array([arm.mean() for arm in env.arms])
            bestarm = np.max(means)
            index_bestarm = np.argwhere(np.isclose(means, bestarm))
            # Store the results
            for r in results:
                self.rewards[polId, envId, :] += np.cumsum(r.rewards)
                self.bestArmPulls[envId][polId, :] += np.cumsum(np.in1d(r.choices, index_bestarm))
                self.pulls[envId][polId, :] += r.pulls

    def getPulls(self, policyId, environmentId):
        return self.pulls[environmentId][policyId, :] / float(self.cfg['repetitions'])

    def getbestArmPulls(self, policyId, environmentId):
        # We have to divide by a arange() = cumsum(ones) to get a frequency
        return self.bestArmPulls[environmentId][policyId, :] / float(self.cfg['repetitions']) / np.arange(start=1, stop=1 + self.cfg['horizon'])

    def getReward(self, policyId, environmentId):
        return self.rewards[policyId, environmentId, :] / float(self.cfg['repetitions'])

    def getRegret(self, policyId, environmentId):
        times = np.arange(self.cfg['horizon'])
        return times * self.envs[environmentId].maxArm - self.getReward(policyId, environmentId)

    def plotRewards(self, environmentId, savefig=None, semilogx=False):
        plt.figure()
        ymin = 0
        for i, policy in enumerate(self.policies):
            Y = self.getRegret(i, environmentId)
            ymin = min(ymin, np.min(Y))  # XXX Should be smarter
            if semilogx:
                plt.semilogx(Y, label=str(policy))
            else:
                plt.plot(Y, label=str(policy))
        plt.legend(loc='upper left')
        plt.grid()
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}$".format(self.cfg['horizon']))
        ymax = plt.ylim()[1]
        plt.ylim(ymin, ymax)
        plt.ylabel(r"Cumulative Regret $R_t$")
        plt.title("Regrets for different bandit algoritms, averaged ${}$ times\nArms: ${}${}".format(self.cfg['repetitions'], repr(self.envs[environmentId].arms), signature))
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig)
        plt.show()

    def plotBestArmPulls(self, environmentId, savefig=None):
        plt.figure()
        for i, policy in enumerate(self.policies):
            Y = self.getbestArmPulls(i, environmentId)
            plt.plot(Y, label=str(policy))
        plt.legend(loc='lower right')
        plt.grid()
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}$".format(self.cfg['horizon']))
        plt.ylim(0, 1)
        plt.ylabel(r"Frequency of pulls of the optimal arm")
        plt.title("Best arm pulls frequency for different bandit algoritms, averaged ${}$ times\nArms: ${}${}".format(self.cfg['repetitions'], repr(self.envs[environmentId].arms), signature))
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig)
        plt.show()

    def giveFinalRanking(self, environmentId):
        print("\nFinal ranking for this environment #{} :".format(environmentId))
        nbPolicies = len(self.policies)
        lastY = np.zeros(nbPolicies)
        for i, policy in enumerate(self.policies):
            Y = self.getRegret(i, environmentId)
            if self.finalRanksOnAverage:
                lastY[i] = np.mean(Y[-int(self.averageOn * self.cfg['horizon'])])   # get average value during the last 0.5% of the iterations
            else:
                lastY[i] = Y[-1]  # get the last value
        # print("lastY =", lastY)  # DEBUG
        # Sort lastY and give ranking
        index_of_sorting = np.argsort(lastY)
        # print("index_of_sorting =", index_of_sorting)  # DEBUG
        for i, k in enumerate(index_of_sorting):
            policy = self.policies[k]
            print("- Policy '{}'\twas ranked\t{} / {} for this simulation (last regret = {:.3f}).".format(str(policy), i + 1, nbPolicies, lastY[k]))
        return lastY, index_of_sorting


# Helper function for the parallelization

def delayed_start(self, env, policy, polId, envId):
    print("\n- Evaluating: {} ...".format(policy))
    if self.useJoblib:
        results = joblib.Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
            joblib.delayed(delayed_play)(env, policy, self.cfg['horizon'])
            for _ in range(self.cfg['repetitions'])
        )
    else:
        results = []
        for _ in range(self.cfg['repetitions']):
            results.append(delayed_play(env, policy, self.cfg['horizon']))
    for result in results:
        self.rewards[polId, envId, :] += np.cumsum(result.rewards)
        self.pulls[envId][polId, :] += result.pulls


# @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof
def delayed_play(env, policy, horizon,
                 random_shuffle=random_shuffle, random_invert=random_invert, nb_random_events=nb_random_events):
    # We have to deepcopy because this function is Parallel-ized
    env = deepcopy(env)
    policy = deepcopy(policy)
    horizon = deepcopy(horizon)

    policy.startGame()
    result = Result(env.nbArms, horizon)
    # XXX Experimental support for random events: shuffling or inverting the list of arms, at these time steps
    t_events = [i * int(horizon / float(nb_random_events)) for i in range(nb_random_events)]
    if nb_random_events is None or nb_random_events <= 0:
        random_shuffle = False
        random_invert = False

    for t in range(horizon):
        choice = policy.choice()
        reward = env.arms[choice].draw(t)
        policy.getReward(choice, reward)
        result.store(t, choice, reward)
        # XXX Experimental : shuffle the arms at the middle of the simulation
        if random_shuffle:
            if t in t_events:  # XXX improve this: it is slow to test 'in <a list>', faster to compute a 't % ...'
                shuffle(env.arms)
                # print("Shuffling the arms ...")  # DEBUG
        # XXX Experimental : invert the order of the arms at the middle of the simulation
        if random_invert:
            if t in t_events:  # XXX improve this: it is slow to test 'in <a list>', faster to compute a 't % ...'
                env.arms = env.arms[::-1]
                # print("Inverting the order of the arms ...")  # DEBUG
    return result
