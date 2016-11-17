# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import joblib
from copy import deepcopy
import matplotlib.pyplot as plt

from .Result import Result
from .MAB import MAB


class Evaluator:
    def __init__(self, configuration):
        self.cfg = configuration
        self.envs = []
        self.policies = []
        self.__initEnvironments__()
        self.rewards = np.zeros((len(self.cfg['policies']),
                                 len(self.envs), self.cfg['horizon']))
        self.pulls = {}
        for env in xrange(len(self.envs)):
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

    def start(self):
        for envId, env in enumerate(self.envs):
            print("Evaluating environment: " + repr(env))
            self.policies = []
            self.__initPolicies__(env)
            for polId, policy in enumerate(self.policies):
                print("+Evaluating: " + policy.__class__.__name__ + ' (' + policy.params + ") ...")
                results = joblib.Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
                    joblib.delayed(play)(env, policy, self.cfg['horizon'])
                    for _ in xrange(self.cfg['repetitions']))
                for result in results:
                    self.rewards[polId, envId, :] += np.cumsum(result.rewards)
                    self.pulls[envId][polId, :] += result.pulls

    def getReward(self, policyId, environmentId):
        return self.rewards[policyId, environmentId, :] / self.cfg['repetitions']

    def getRegret(self, policyId, environmentId):
        horizon = np.arange(self.cfg['horizon'])
        return horizon * self.envs[environmentId].maxArm - self.getReward(policyId, environmentId)

    def plotResults(self, environment, savefig=None):
        figure = plt.figure()
        for i, policy in enumerate(self.policies):
            plt.plot(self.getRegret(i, environment), label=str(policy))
        plt.legend(loc='upper left')
        plt.grid()
        plt.xlabel("Time steps")
        # ymin, ymax = plt.ylim()
        # ymin = max(0, ymin)    # prevent a negative ymin
        # plt.ylim(ymin, ymax)
        plt.ylabel("Cumulative Regret")
        plt.title("Regrets for different bandit algoritms, averaged {} times\nArms: {}".format(self.cfg['repetitions'], repr(self.cfg['environment'][environment])))
        plt.show()
        if savefig:
            plt.savefig(savefig)


def play(env, policy, horizon):
    env = deepcopy(env)
    policy = deepcopy(policy)
    horizon = deepcopy(horizon)

    policy.startGame()
    result = Result(env.nbArms, horizon)
    for t in xrange(horizon):
        choice = policy.choice()
        reward = env.arms[choice].draw(t)
        policy.getReward(choice, reward)
        result.store(t, choice, reward)
    return result
