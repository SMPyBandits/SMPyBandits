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

from SMPyBandits.Arms import UnboundedGaussian as Gaussian
from SMPyBandits.Policies import FEWA, GreedyOracle, UCB
from SMPyBandits.Environment.MAB_rotting import repetedRuns
import numpy as np
import datetime
import os
import logging
import sys

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
### SET Policies
policies = [
    [FEWA, {'alpha': .01, 'delta': 1, 'subgaussian': 1}],
    [FEWA, {'alpha': .06, 'delta': 1, 'subgaussian': 1}],
    [FEWA, {'alpha': 0.25, 'delta': 1, 'subgaussian': 1}],
    [UCB, {}]
]
policy_ind = 0 if len(sys.argv) == 1 else sys.argv[1]
policy = policies[policy_ind]
policy_name = str(policy[0](nbArms=2, **policy[1]))
policy_name_nospace = policy_name.replace(' ', '_')

os.makedirs('./data/logging/', exist_ok=True)
logging.basicConfig(filename=os.path.join('./data/logging', date + '.log'), level=logging.INFO,
                    format='%(asctime)s %(message)s')
logging.info("Policy : %s$" % (policy_name))

PARALLEL = -1  # Set positive int to indicate the number of core, -1 to use all the cores, and False to not parallelize
REPETITIONS = 1000 if len(sys.argv) < 3 else sys.argv[2]  # Set the number of repetitions
HORIZON = 5000  # Horizon T
sigma = 1  # Gaussian noise std

### SET L/2 in figure 1
logging.info("CONFIG : CPU %s" % os.cpu_count())
logging.info("CONFIG : REPETITIONS %s" % REPETITIONS)
logging.info("CONFIG : HORIZON %s" % HORIZON)
logging.info("CONFIG : SIGMA %s" % sigma)

### GAME 1
logging.info("EVENT : GAME 1 mu = 0.14")
arms = [
    [Gaussian, {'sigma': sigma, "mu": 0}],
    [Gaussian, {'sigma': sigma, "mu": 0.14}],
]

rew, noisy_rew, time, pulls, cumul_pulls = repetedRuns(policy, arms, rep=REPETITIONS, T=HORIZON, parallel=PARALLEL)
oracle_rew, noisy_oracle_rew, oracle_time, oracle_pull, oracle_cumul_pulls = repetedRuns(
    [GreedyOracle, {}], arms, rep=1, T=HORIZON, oracle=True
)
regret = oracle_rew - rew
logging.info("EVENT : SAVING ... ")
path_regret = os.path.join('./data/', 'REGRET1_' + policy_name_nospace + '_' + date)
path_time = os.path.join('./data/', 'TIME1_' + policy_name_nospace + '_' + date)
np.save(path_regret, regret)
np.save(path_time, time)

### GAME 2
logging.info("EVENT : GAME 2 mu = 1")
arms = [
    [Gaussian, {'sigma': sigma, "mu": 0}],
    [Gaussian, {'sigma': sigma, "mu": 1}],
]

rew, noisy_rew, time, pulls, cumul_pulls = repetedRuns(policy, arms, rep=REPETITIONS, T=HORIZON, parallel=PARALLEL)
oracle_rew, noisy_oracle_rew, oracle_time, oracle_pull, oracle_cumul_pulls = repetedRuns(
    [GreedyOracle, {}], arms, rep=1, T=HORIZON, oracle=True
)
regret = oracle_rew - rew
logging.info("EVENT : SAVING ... ")
path_regret = os.path.join('./data/', 'REGRET2_' + policy_name_nospace + '_' + date)
path_time = os.path.join('./data/', 'TIME2_' + policy_name_nospace + '_' + date)
np.save(path_regret, regret)
np.save(path_time, time)
logging.info("EVENT : END ... ")
