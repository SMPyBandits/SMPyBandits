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
from .Result import Result
from .MAB import MAB

# Customize here if you want a signature on the titles of each plot
signature = "\n(By Lilian Besson, Nov.2016 - Code on https://github.com/Naereen/AlgoBandits)"


class Evaluator:
    """ Evaluator class to run the simulations."""

    def __init__(self, configuration,
                 finalRanksOnAverage=True, averageOn=5e-3,
                 useJoblibForPolicies=False):
        self.cfg = configuration
        self.finalRanksOnAverage = finalRanksOnAverage
        self.averageOn = averageOn
        self.useJoblibForPolicies = useJoblibForPolicies
        self.useJoblib = USE_JOBLIB and self.cfg['n_jobs'] != 1
        self.envs = []
        self.policies = []
        self.__initEnvironments__()
        self.rewards = np.zeros((len(self.cfg['policies']),
                                 len(self.envs), self.cfg['horizon']))
        self.pulls = dict()
        for env in range(len(self.envs)):
            self.pulls[env] = np.zeros((len(self.cfg['policies']), self.envs[env].nbArms))
        print("Number of algorithms to compare:", len(self.cfg['policies']))
        print("Number of environments to try:", len(self.envs))
        print("Time horizon:", self.cfg['horizon'])
        print("Number of repetitions:", self.cfg['repetitions'])

    def __initEnvironments__(self):
        for armType in self.cfg['environment']:
            self.envs.append(MAB(armType))

    def __initPolicies__(self, env):
        for policy in self.cfg['policies']:
            print("policy =", policy)  # DEBUG
            self.policies.append(policy['archtype'](env.nbArms,
                                                    **policy['params']))

    def start_all_env(self):
        for envId, env in enumerate(self.envs):
            self.start_one_env(envId, env)

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
            # # FIXME try to also parallelize this loop on policies ?
            for polId, policy in enumerate(self.policies):
                print("\n- Evaluating policy #{}/{}: {} ...".format(polId + 1, len(self.policies), policy))
                if self.useJoblib:
                    results = joblib.Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
                        joblib.delayed(delayed_play)(env, policy, self.cfg['horizon'])
                        for _ in range(self.cfg['repetitions'])
                    )
                else:
                    results = []
                    for _ in range(self.cfg['repetitions']):
                        r = delayed_play(env, policy, self.cfg['horizon'])
                        results.append(r)
                for r in results:
                    self.rewards[polId, envId, :] += np.cumsum(r.rewards)
                    self.pulls[envId][polId, :] += r.pulls

    def getReward(self, policyId, environmentId):
        return self.rewards[policyId, environmentId, :] / float(self.cfg['repetitions'])

    def getRegret(self, policyId, environmentId):
        horizon = np.arange(self.cfg['horizon'])
        return horizon * self.envs[environmentId].maxArm - self.getReward(policyId, environmentId)

    def plotResults(self, environmentId, savefig=None, semilogx=False):
        plt.figure()
        nbPolicies = len(self.policies)
        ymin = 0
        for i, policy in enumerate(self.policies):
            if i < 13:
                color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'darkgreen', 'darkblue', 'darkred', 'darkcyan'][i % nbPolicies]  # Default choice
            else:
                # XXX Experimental... Get a random #RGB color from the hash of the policy
                s = list(str(hex(abs(hash(policy))))[2:][::-1])
                shuffle(s)
                color = '#' + str(''.join(s))[:6]
            Y = self.getRegret(i, environmentId)
            ymin = min(ymin, np.min(Y))  # XXX Should be smarter
            print("Using color {} for policy number #{}/{} and called {}...".format(color, i + 1, nbPolicies, str(policy)))
            if semilogx:
                plt.semilogx(Y, label=str(policy), c=color)
            else:
                plt.plot(Y, label=str(policy), c=color)
        plt.legend(loc='upper left')
        plt.grid()
        plt.xlabel(r"Time steps $t = 1, \dots, T$, horizon $T = {}$".format(self.cfg['horizon']))
        ymax = plt.ylim()[1]
        # ymin = max(0, ymin)    # prevent a negative ymin
        plt.ylim(ymin, ymax)
        plt.ylabel(r"Cumulative Regret $R_t$")
        plt.title("Regrets for different bandit algoritms, averaged ${}$ times\nArms: ${}${}".format(self.cfg['repetitions'], repr(self.envs[environmentId].arms), signature))
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


def delayed_play(env, policy, horizon):
    # We have to deepcopy because this function is Parallel-ized
    env = deepcopy(env)
    policy = deepcopy(policy)
    horizon = deepcopy(horizon)

    policy.startGame()
    result = Result(env.nbArms, horizon)
    for t in range(horizon):
        choice = policy.choice()
        reward = env.arms[choice].draw(t)
        policy.getReward(choice, reward)
        result.store(t, choice, reward)
    return result
