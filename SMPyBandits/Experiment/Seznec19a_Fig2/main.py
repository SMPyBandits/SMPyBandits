"""
author : Julien SEZNEC
Produce the experiment and record the relevant data to reproduce Figure 2 of [Seznec et al.,  2019a]
Reference: [Seznec et al.,  2019a]
Rotting bandits are not harder than stochastic ones;
Julien Seznec, Andrea Locatelli, Alexandra Carpentier, Alessandro Lazaric, Michal Valko ;
Proceedings of Machine Learning Research, PMLR 89:2564-2572, 2019.
http://proceedings.mlr.press/v89/seznec19a.html
https://arxiv.org/abs/1811.11043 (updated version)
"""

from SMPyBandits.Arms import RestedRottingGaussian
from SMPyBandits.Policies import FEWA, EFF_FEWA, wSWA, GreedyOracle, SWUCB, DiscountedUCB as DUCB, RAWUCB, \
  GaussianGLR_IndexPolicy, Exp3S, klUCBloglog_forGLR, UCB, Exp3WithHorizon as Exp3
from SMPyBandits.Environment.MAB_rotting import repetedRuns
import numpy as np
import datetime
import os
import logging
import sys

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PARALLEL = -1  # Set positive int to indicate the number of core, -1 to use all the cores, and False to not parallelize
REPETITIONS = 1 if len(sys.argv) < 3 else int(sys.argv[2])  # Set the number of repetitions
HORIZON = T = 25000  # Horizon T
sigma = 1  # Gaussian noise std
K = 10

### SET Policies
policies = [
  [FEWA, {'alpha': .06, 'delta': 1}], #0
  [EFF_FEWA, {'alpha': 0.06, 'delta': 1}], #1
  [wSWA, {'alpha': 0.002}], #2
  [wSWA, {'alpha': 0.02}], #3
  [wSWA, {'alpha': 0.2}], #4
  [DUCB, {'gamma': 0.997}], #5
  [SWUCB, {'tau': 200}], #6
  [FEWA, {'alpha': 4}], #7
  [RAWUCB, {'alpha': 1.4}], #8
  [RAWUCB, {'alpha': 4}], #9
  [GaussianGLR_IndexPolicy, {'policy': klUCBloglog_forGLR, 'delta': np.sqrt(1 / T), 'use_increasing_alpha': True,
                             'per_arm_restart': True, 'sig2': sigma ** 2, 'horizon': T, 'use_localization': False}], #10
  [GaussianGLR_IndexPolicy, {'policy': klUCBloglog_forGLR, 'delta': np.sqrt(1 / T), 'alpha0': 0,
                             'per_arm_restart': True, 'sig2': sigma ** 2, 'use_localization': False}],  # 11
  [Exp3S, {'alpha': 1 / T, 'gamma': min(1, np.sqrt(K * np.log(K * T) / T))}], #12
  [UCB, {}], #13
  [Exp3, {'horizon': T}] #14
]
policy_ind = 13 if len(sys.argv) == 1 else int(sys.argv[1])
policy = policies[policy_ind]
policy_name = str(policy[0](nbArms=2, **policy[1]))
policy_name_nospace = policy_name.replace(' ', '_')

os.makedirs('./data/logging/', exist_ok=True)
logging.basicConfig(filename=os.path.join('./data/logging', date + '.log'), level=logging.INFO,
                    format='%(asctime)s %(message)s')
logging.info("Policy : %s$" % (policy_name))

### SET L/2 in figure 1
logging.info("CONFIG : CPU %s" % os.cpu_count())
logging.info("CONFIG : REPETITIONS %s" % REPETITIONS)
logging.info("CONFIG : HORIZON %s" % HORIZON)
logging.info("CONFIG : SIGMA %s" % sigma)

### SET K arms

mus = [0] + [0.001 * np.sqrt(10) ** (i) for i in range(9)]


def abruptDecayFunction(mui, muf, breakpoint):
  return lambda n: mui if n < breakpoint else muf


arms = [
  [RestedRottingGaussian, {'decayingFunction': abruptDecayFunction(mu, -mu, 1000), 'sigma': sigma, }] for mu in mus
]

rew, noisy_rew, time, pulls, cumul_pulls = repetedRuns(policy, arms, rep=REPETITIONS, T=HORIZON, parallel=PARALLEL)
oracle_rew, noisy_oracle_rew, oracle_time, oracle_pull, oracle_cumul_pulls = repetedRuns([GreedyOracle, {}], arms,
                                                                                         rep=1, T=HORIZON, oracle=True)
regret = oracle_rew - rew
diffpulls = np.abs(cumul_pulls - oracle_cumul_pulls)
logging.info("EVENT : SAVING ... ")
path_regret = os.path.join('./data/', 'REGRET_' + policy_name_nospace + '_' + date)
path_diffpull = os.path.join('./data/', 'DIFFPULL_' + policy_name_nospace + '_' + date)
path_time = os.path.join('./data/', 'TIME_' + policy_name_nospace + '_' + date)
np.save(path_regret, regret)
np.save(path_diffpull, diffpulls)
np.save(path_time, np.array(time))
logging.info("EVENT : END ... ")
