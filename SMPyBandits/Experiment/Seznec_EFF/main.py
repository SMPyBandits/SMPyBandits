"""
author: Julien SEZNEC
Produce the experiment about the efficiency of EFF_RAWUCB
For the thesis manuscript.
"""

from SMPyBandits.Arms import RestedRottingGaussian
from SMPyBandits.Policies import  GreedyOracle, RAWUCB, EFF_RAWUCB
from SMPyBandits.Environment.MAB_rotting import repetedRuns
import numpy as np
import datetime
import os
import logging
import sys

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PARALLEL = -1  # Set positive int to indicate the number of core, -1 to use all the cores, and False to not parallelize
REPETITIONS = 1 if len(sys.argv) < 3 else int(sys.argv[2])  # Set the number of repetitions
HORIZON = T = 10**6  # Horizon T
sigma = 1  # Gaussian noise std
K = 2

### SET Policies
policies = [
  [RAWUCB, {'alpha': 1.4}],  # 0
  [EFF_RAWUCB, {'alpha': 1.4, 'm':2}],  # 1
  [EFF_RAWUCB, {'alpha': 1.4, 'm': 2.1}],  # 2
  [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.9}],  # 3
  [EFF_RAWUCB, {'alpha': 1.4, 'm': 3}],  # 4
  [EFF_RAWUCB, {'alpha': 1.4, 'm': 4}],  # 5
  [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.01}],  # 6
  [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.05}],  # 7
  [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.1}],  # 8
  [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.2}],  # 9
  [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.5}],  # 10
]
policy_ind = 10 if len(sys.argv) == 1 else int(sys.argv[1])
policy = policies[policy_ind]
policy_name = str(policy[0](nbArms=2, **policy[1]))
policy_name_nospace = policy_name.replace(' ', '_')

regret_path = os.path.join('./data', 'REGRET_' + policy_name_nospace + '_' + date)
time_path = os.path.join('./data', 'TIME_' + policy_name_nospace + '_' + date)
os.makedirs('./data/logging/', exist_ok=True)
logging.basicConfig(filename=os.path.join('./data/logging', date + '.log'), level=logging.INFO,
                    format='%(asctime)s %(message)s')
logging.info("Policy : %s$" % (policy_name))

### SET L/2
mus = [0.01, 1]
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
