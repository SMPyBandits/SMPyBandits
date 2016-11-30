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
try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    print("joblib not found. Install it from pypi ('pip install joblib') or conda.")
    USE_JOBLIB = False
# Local imports
from .plotsettings import DPI, signature, maximizeWindow, palette
from .Result import Result
from .MAB import MAB


# Parameters for the random events
random_shuffle = False
random_invert = False
nb_random_events = 4


class Evaluator(object):
    """ Evaluator class to run the simulations."""

    def __init__(self, configuration,
                 finalRanksOnAverage=True, averageOn=5e-3,
                 useJoblibForPolicies=False):
        # Configuration
        self.cfg = configuration
        # Attributes
        self.nbPolicies = len(self.cfg['policies'])
        print("Number of policies in this comparaison:", self.nbPolicies)
        self.horizon = self.cfg['horizon']
        print("Time horizon:", self.horizon)
        self.repetitions = self.cfg['repetitions']
        print("Number of repetitions:", self.repetitions)
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
        self.rewards = np.zeros((self.nbPolicies,
                                 len(self.envs), self.horizon))
        self.BestArmPulls = dict()
        self.pulls = dict()
        for env in range(len(self.envs)):
            self.BestArmPulls[env] = np.zeros((self.nbPolicies, self.horizon))
            self.pulls[env] = np.zeros((self.nbPolicies, self.envs[env].nbArms))
        print("Number of environments to try:", len(self.envs))

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
        for polId, policy in enumerate(self.policies):
            print("\n- Evaluating policy #{}/{}: {} ...".format(polId + 1, self.nbPolicies, policy))
            if self.useJoblib:
                results = joblib.Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
                    joblib.delayed(delayed_play)(env, policy, self.horizon)
                    for _ in range(self.repetitions)
                )
            else:
                results = []
                for _ in range(self.repetitions):
                    r = delayed_play(env, policy, self.horizon)
                    results.append(r)
            # Get the position of the best arms
            env = self.envs[envId]
            means = np.array([arm.mean() for arm in env.arms])
            bestarm = np.max(means)
            index_bestarm = np.argwhere(np.isclose(means, bestarm))
            # Store the results
            for r in results:
                self.rewards[polId, envId, :] += np.cumsum(r.rewards)
                self.BestArmPulls[envId][polId, :] += np.cumsum(np.in1d(r.choices, index_bestarm))
                self.pulls[envId][polId, :] += r.pulls

    def getPulls(self, policyId, environmentId=0):
        return self.pulls[environmentId][policyId, :] / float(self.repetitions)

    def getBestArmPulls(self, policyId, environmentId=0):
        # We have to divide by a arange() = cumsum(ones) to get a frequency
        return self.BestArmPulls[environmentId][policyId, :] / (float(self.repetitions) * np.arange(start=1, stop=1 + self.horizon))

    def getReward(self, policyId, environmentId=0):
        return self.rewards[policyId, environmentId, :] / float(self.repetitions)

    def getRegret(self, policyId, environmentId=0):
        return np.arange(1, 1 + self.horizon) * self.envs[environmentId].maxArm - self.getReward(policyId, environmentId)

    def plotRegrets(self, environmentId, savefig=None, semilogx=False):
        plt.figure()
        ymin = 0
        colors = palette(self.nbPolicies)
        for i, policy in enumerate(self.policies):
            Y = self.getRegret(i, environmentId)
            ymin = min(ymin, np.min(Y))  # XXX Should be smarter
            if semilogx:
                plt.semilogx(Y, label=str(policy), color=colors[i])
            else:
                plt.plot(Y, label=str(policy), color=colors[i])
        plt.legend(loc='upper left', numpoints=1)
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}$".format(self.horizon))
        ymax = plt.ylim()[1]
        plt.ylim(ymin, ymax)
        plt.ylabel(r"Cumulative Regret $R_t$")
        plt.title("Regrets for different bandit algoritms, averaged ${}$ times\nArms: ${}${}".format(self.repetitions, repr(self.envs[environmentId].arms), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI, bbox_inches='tight')
        plt.show()

    def plotBestArmPulls(self, environmentId, savefig=None):
        plt.figure()
        colors = palette(self.nbPolicies)
        for i, policy in enumerate(self.policies):
            Y = self.getBestArmPulls(i, environmentId)
            plt.plot(Y, label=str(policy), color=colors[i])
        plt.legend(loc='lower right', numpoints=1)
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}$".format(self.horizon))
        plt.ylim(-0.03, 1.03)
        plt.ylabel(r"Frequency of pulls of the optimal arm")
        plt.title("Best arm pulls frequency for different bandit algoritms, averaged ${}$ times\nArms: ${}${}".format(self.repetitions, repr(self.envs[environmentId].arms), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI, bbox_inches='tight')
        plt.show()

    def printFinalRanking(self, environmentId=0):
        assert 0 < self.averageOn < 1, "Error, the parameter averageOn of a EvaluatorMultiPlayers classs has to be in (0, 1) strictly, but is = {} here ...".format(self.averageOn)
        print("\nFinal ranking for this environment #{} :".format(environmentId))
        nbPolicies = self.nbPolicies
        lastY = np.zeros(nbPolicies)
        for i, policy in enumerate(self.policies):
            Y = self.getRegret(i, environmentId)
            if self.finalRanksOnAverage:
                lastY[i] = np.mean(Y[-int(self.averageOn * self.horizon)])   # get average value during the last 0.5% of the iterations
            else:
                lastY[i] = Y[-1]  # get the last value
        # Sort lastY and give ranking
        index_of_sorting = np.argsort(lastY)
        for i, k in enumerate(index_of_sorting):
            policy = self.policies[k]
            print("- Policy '{}'\twas ranked\t{} / {} for this simulation (last regret = {:.3f}).".format(str(policy), i + 1, nbPolicies, lastY[k]))
        return lastY, index_of_sorting


# Helper function for the parallelization

# @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof
def delayed_play(env, policy, horizon,
                 random_shuffle=random_shuffle, random_invert=random_invert, nb_random_events=nb_random_events):
    # We have to deepcopy because this function is Parallel-ized
    env = deepcopy(env)
    policy = deepcopy(policy)
    horizon = deepcopy(horizon)
    # Start game
    policy.startGame()
    result = Result(env.nbArms, horizon)  # One Result object, for every policy
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
