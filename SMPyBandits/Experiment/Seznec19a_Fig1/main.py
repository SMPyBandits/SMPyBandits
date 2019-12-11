from ...Arms import RestedRottingArm
from ...Policies import FEWA, EFF_FEWA, wSWA
from ...Environment.MAB_rotting import repetedRuns
import numpy as np
import datetime
import os
import logging

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
path = os.path.join('./data', date + "fig1")
os.mkdir(path)
logging.basicConfig(filename=os.path.join(path, 'logging.log'), level=logging.INFO, format='%(asctime)s %(message)s')


PARALLEL = -1 # Set positive int to indicate the number of core, -1 to use all the cores, and False to not parallelize
REPETITIONS = 291 # Set the number of repetitions
HORIZON = 10000 # Horizon T
sigma = 1 #Gaussian noise std

### SET L/2 in figure 1
mus = [.01 * 1.25 ** i for i in range(30)]
logging.info("CONFIG : CPU %s" % os.cpu_count())
logging.info("CONFIG : REPETITIONS %s" % REPETITIONS)
logging.info("CONFIG : HORIZON %s" % HORIZON)
logging.info("CONFIG : SIGMA %s" % sigma)

### SET Policies
policies = [
  [FEWA, {'alpha': .03, 'delta': 1}],
  [FEWA, {'alpha': .06, 'delta': 1}],
  [FEWA, {'alpha': .1, 'delta': 1}],
  [EFF_FEWA, {'alpha' : 0.06, 'delta':1, 'm':1.1}],
  [wSWA, {'alpha' : 0.002}],
  [wSWA, {'alpha' : 0.02}],
  [wSWA, {'alpha' : 0.2}],
]

for i, pol in enumerate(policies):
  logging.info("POLICY " + str(i) + " : " + str(pol[0](nbArms=2, **pol[1])))


for i, policy in enumerate(policies):
  logging.info("STARTING POLICY " + str(i) + " : " + str(pol[0](nbArms=2, **pol[1])))
  noisy_reward_res =[]
  regret_res = []
  time_res = []
  overpull_res =[]
  for m, mu in enumerate(mus):
    logging.info("GAME %s : $\mu = %s$" % (m, mu))
    print(mu)
    ### SET K arms
    arms = [
      [
        RestedRottingArm,
        {'function': constant, 'sigma': sigma, 'functionName': 'constant', 'functionArgs': {"mu": 0}}
      ],
      [
        RestedRottingArm,
        {'function': abruptSingleDecay, 'sigma': sigma,
         'functionName': 'AbruptSingleDecay', 'functionArgs': {"mu": mu, "switchPoint": HORIZON / 4}}
      ],
    ]
    rew, noisy_rew, time, pulls, cumul_pulls = repetedRuns(arms, policies, rep=REPETITIONS, T=HORIZON, parallel=PARALLEL)
    oracle_rew, noisy_oracle_rew, oracle_time, oracle_pull, oracle_cumul_pulls = repetedRuns(arms, [[GreedyOracle, {}]], rep=1, T=HORIZON, oracle=True)
    regret = oracle_rew - rew
    logging.info("EVENT : SAVING ... ")
    regret_res.append(regret)
    np.save(os.path.join(path, 'regret'), np.array(regret_res))
logging.info("EVENT : END ... ")




