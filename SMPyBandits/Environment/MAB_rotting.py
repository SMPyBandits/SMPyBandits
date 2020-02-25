"""
author : Julien SEZNEC
Code to launch (rotting) bandit games.
It is code in a functional programming way : each execution return arrays related to each run.
"""

import time
import numpy as np
import logging
from joblib import Parallel, delayed

REPETITIONS = 1000
HORIZON = 10000

def repetedRuns(policy, arms, rep = REPETITIONS, T = HORIZON, parallel = True, oracle = False):
    rew = np.empty(shape = (rep, T))
    noisy_rew = np.empty(shape = (rep, T))
    time = np.empty(shape = (rep, T))
    pulls = np.empty(shape=(rep, T))
    cumul_pulls = np.empty(shape=(rep, len(arms)))
    if parallel:
        res = Parallel(n_jobs=parallel)(delayed(singleRun)(policy,arms, T, r, oracle) for r in range(rep))
    else:
        res = [singleRun(policy,arms, T=T) for _ in range(rep)]
    rew[:, :] =  np.array([r['cumul'] for r in res ])
    noisy_rew[:, :] = np.array([r['noisy_cumul'] for r in res])
    time[:, :] = np.array([r['time'] for r in res ])
    pulls[:,:] = np.array([r['pulls'] for r in res ])
    cumul_pulls[:,:] = np.array([r['cumul_pulls'] for r in res ])
    return rew, noisy_rew, time, pulls, cumul_pulls

def singleRun(policy, arms, T = HORIZON,rep_index = 0, oracle=False):
    myArms = [arm[0](**arm[1]) for arm in arms]
    if oracle:
        policy[1]['arms'] = myArms
    myPolicy = policy[0](len(myArms), **policy[1])
    myPolicy.startGame()
    logging.debug(str(rep_index) + ' ' + myPolicy.__str__())
    res = play(myArms, myPolicy, T, Oracle=oracle)
    return {
      'cumul': np.array(res['rewards']).cumsum(),
      'noisy_cumul': np.array(res['noisy_rewards']),
      'time' : np.array(res['time']),
      'pulls' : np.array(res['pulls']),
      'cumul_pulls' : np.array(res['cumul_pulls'])
    }


def play(arms, policy, T, Oracle= False):
    noisy_rewards = []
    rewards = []
    times = []
    pulls = []
    cumul_pulls = [0 for _ in range(len(arms))]
    for t in range(T):
        start = time.time()
        choice = policy.choice()
        reward = arms[choice].mean
        noisy_reward = arms[choice].draw(t) if not Oracle else arms[choice].oracle_draw(t)
        policy.getReward(choice, noisy_reward)
        times.append(time.time() - start)
        noisy_rewards.append(noisy_reward)
        rewards.append(reward)
        pulls.append(choice)
        cumul_pulls[choice] += 1
    return {'rewards': rewards, 'noisy_rewards': noisy_rewards, 'time': times, 'pulls': pulls, 'cumul_pulls' : cumul_pulls}