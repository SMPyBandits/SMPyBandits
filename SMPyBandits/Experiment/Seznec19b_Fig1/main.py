"""
author : Julien SEZNEC
Produce the experiment and record the relevant data to reproduce Figure 2 of [Seznec et al.,  2019b]
Reference: [Seznec et al.,  2019b]
Rotting bandits are not harder than stochastic ones; #TODO modify
Julien Seznec, Pierre Ménard, Alessandro Lazaric, Michal Valko ;
Proceedings of Machine Learning Research, PMLR 89:2564-2572, 2019.
http://proceedings.mlr.press/v89/seznec19a.html
https://arxiv.org/abs/1811.11043 (updated version)
"""

from SMPyBandits.Arms import RestlessBinomial
from SMPyBandits.Policies import EFF_FEWA, EFF_RAWUCB, Exp3S, GaussianGLR_IndexPolicy, klUCBloglog_forGLR, GreedyOracle
from SMPyBandits.Environment.MAB_rotting import repetedRuns
from math import sqrt, log
import numpy as np
import pandas as pd
import datetime
import os
import logging
import sys

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs('./data/logging/', exist_ok=True)
logging.basicConfig(filename=os.path.join('./data/logging', date + '.log'), level=logging.INFO,
                    format='%(asctime)s %(message)s')

# Arms & DATA
DAY = 2 if len(sys.argv) == 1 else int(sys.argv[1])
data_file = 'data/Reward/reward_data_day_%s.csv' % (DAY)
DRAWS = 10
logging.info("CONSTANT CONFIG : DATA %s" % data_file)
df = pd.read_csv(data_file, index_col=0).transpose().reset_index(drop=True)

def rew_function(df, col):
    def res(t):
        return df[col][t]
    return res
arms = [[RestlessBinomial, {'rewardFunction': rew_function(df, col), 'draws': DRAWS}] for col in df.columns]  # TODO arms
K = len(arms)
for i, arm in enumerate(arms):
    logging.info("DYNAMIC CONFIG :  ARM " + str(i) + " " + str(arm[0](**arm[1])))

# Config
PARALLEL = -1  # Set positive int to indicate the number of core, -1 to use all the cores, and False to not parallelize
REPETITIONS = 3  if len(sys.argv) == 1 else int(sys.argv[3])# Set the number of repetitions
HORIZON = T = len(df)  # Horizon T
SIGMA = (0.03 * 0.97 * 10) ** .5
V = 0.04
logging.info("CONSTANT CONFIG : CPU %s" % os.cpu_count())
logging.info("CONSTANT CONFIG : REPETITIONS %s" % REPETITIONS)
logging.info("CONSTANT CONFIG : HORIZON %s" % HORIZON)
logging.info("CONSTANT CONFIG : SIGMA %s" % SIGMA)

### SET Policies
policies = [
    [EFF_RAWUCB, {'alpha': 1.4, 'subgaussian': SIGMA,  'm': 1.1}],
    [EFF_RAWUCB, {'alpha': 1.4, 'subgaussian': SIGMA, 'm': 2}],
    [EFF_RAWUCB, {'alpha': 4, 'subgaussian': SIGMA, 'm': 1.1}],
    [EFF_FEWA, {'alpha': .06, 'subgaussian': SIGMA, 'm': 1.1}],
    [EFF_FEWA, {'alpha': 4, 'subgaussian': SIGMA, 'm': 1.1}],
    [GaussianGLR_IndexPolicy,
     {'policy': klUCBloglog_forGLR, 'delta': sqrt(1 / T), 'alpha0': 0, 'per_arm_restart': True, 'sig2': SIGMA ** 2}],
    [Exp3S, {'alpha': 1 / T, 'gamma': (K * V / T)**(1/3)}],
]

policy_ind = 2 if len(sys.argv) == 1 else int(sys.argv[2])
policy = policies[policy_ind]
policy_name = str(policy[0](nbArms=2, **policy[1]))
policy_name_nospace = policy_name.replace(' ', '_')
logging.info("CONSTANT CONFIG :  POLICY " + str(i) + " " + policy_name)
rew, noisy_rew, time, pulls, cumul_pulls = repetedRuns(policy, arms, rep=REPETITIONS, T=HORIZON, parallel=PARALLEL)
oracle_rew = df.max(axis=1).cumsum().to_numpy()
regret = DRAWS*oracle_rew - rew
logging.info("EVENT : SAVING ... ")
path_regret = os.path.join('./data/DAY_%s_REGRET_%s_%s' % (DAY, policy_name_nospace, date))
path_time = os.path.join('./data/DAY_%s_TIME_%s_%s' % (DAY, policy_name_nospace, date))
np.save(path_regret, regret)
np.save(path_time, time)
logging.info("EVENT : END ... ")
