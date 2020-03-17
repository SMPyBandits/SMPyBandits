"""
author: Julien SEZNEC
Produce the experiment and record the relevant data to reproduce Figure 1 of [Seznec et al.,  2019a]
Reference: [Seznec et al.,  2019a]
Rotting bandits are not harder than stochastic ones;
Julien Seznec, Andrea Locatelli, Alexandra Carpentier, Alessandro Lazaric, Michal Valko ;
Proceedings of Machine Learning Research, PMLR 89:2564-2572, 2019.
http://proceedings.mlr.press/v89/seznec19a.html
https://arxiv.org/abs/1811.11043 (updated version)
"""

from SMPyBandits.Arms import RestedRottingGaussian
from SMPyBandits.Policies import FEWA, EFF_FEWA, wSWA, GreedyOracle, RAWUCB, Exp3S, GaussianGLR_IndexPolicy, \
  klUCBloglog_forGLR
from SMPyBandits.Environment.MAB_rotting import repetedRuns
import numpy as np
import datetime
import os
import logging
import sys

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PARALLEL = -1  # Set positive int to indicate the number of core, -1 to use all the cores, and False to not parallelize
REPETITIONS = 4 if len(sys.argv) < 3 else int(sys.argv[2])  # Set the number of repetitions
HORIZON = T = 10000  # Horizon T
sigma = 1  # Gaussian noise std
K = 2

### SET Policies
policies = [
  [FEWA, {'alpha': .03, 'delta': 1}],  # 0
  [FEWA, {'alpha': .06, 'delta': 1}],  # 1
  [FEWA, {'alpha': .1, 'delta': 1}],  # 2
  [EFF_FEWA, {'alpha': 0.06, 'delta': 1}],  # 3
  [wSWA, {'alpha': 0.002}],  # 4
  [wSWA, {'alpha': 0.02}],  # 5
  [wSWA, {'alpha': 0.2}],  # 6
  [RAWUCB, {'alpha': 1.4}],  # 7
  [RAWUCB, {'alpha': 4}],  # 8
  [FEWA, {'alpha': 4}],  # 9
  [GaussianGLR_IndexPolicy, {'policy': klUCBloglog_forGLR, 'delta': np.sqrt(1 / T), 'alpha0': 0,
                             'per_arm_restart': True, 'sig2': sigma ** 2, 'use_localization': False}],  # 10
  [Exp3S, {'alpha': 1 / T, 'gamma': min(1, np.sqrt(K * np.log(K * T) / T))}],  # 11
]
policy_ind = 3 if len(sys.argv) == 1 else int(sys.argv[1])
policy = policies[policy_ind]
policy_name = str(policy[0](nbArms=2, **policy[1]))
policy_name_nospace = policy_name.replace(' ', '_')

regret_path = os.path.join('./data', 'REGRET_' + policy_name_nospace + '_' + date)
time_path = os.path.join('./data', 'TIME_' + policy_name_nospace + '_' + date)
os.makedirs('./data/logging/', exist_ok=True)
logging.basicConfig(filename=os.path.join('./data/logging', date + '.log'), level=logging.INFO,
                    format='%(asctime)s %(message)s')
logging.info("Policy : %s$" % (policy_name))

### SET L/2 in figure 1
mus = [.01 * 1.25 ** i for i in range(30)]
logging.info("CONFIG : CPU %s" % os.cpu_count())
logging.info("CONFIG : REPETITIONS %s" % REPETITIONS)
logging.info("CONFIG : HORIZON %s" % HORIZON)
logging.info("CONFIG : SIGMA %s" % sigma)

noisy_reward_res = []
regret_res = []
time_res = []
overpull_res = []
for m, mu in enumerate(mus):
  logging.info("GAME %s : $\mu = %s$" % (m, mu))
  print(mu)
  ### SET K arms
  arms = [
    [
      RestedRottingGaussian,
      {'decayingFunction': lambda n: mu if n <= HORIZON / 4 else -mu, 'sigma': sigma, }
    ],
    [
      RestedRottingGaussian,
      {'decayingFunction': lambda n: 0, 'sigma': sigma, }
    ],
  ]
  rew, noisy_rew, time, pulls, cumul_pulls = repetedRuns(policy, arms, rep=REPETITIONS, T=HORIZON, parallel=PARALLEL)
  oracle_rew, noisy_oracle_rew, oracle_time, oracle_pull, oracle_cumul_pulls = repetedRuns(
    [GreedyOracle, {}], arms, rep=1, T=HORIZON, oracle=True
  )
  regret = oracle_rew - rew
  regret_res.append(regret)
  time_res.append(time)
logging.info("EVENT : SAVING ... ")
np.save(regret_path, np.array(regret_res))
np.save(time_path, np.array(regret_res))
logging.info("EVENT : END ... ")
