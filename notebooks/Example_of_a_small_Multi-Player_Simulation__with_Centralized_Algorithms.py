
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#An-example-of-a-small-Multi-Player-simulation,-with-Centralized-Algorithms" data-toc-modified-id="An-example-of-a-small-Multi-Player-simulation,-with-Centralized-Algorithms-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>An example of a small Multi-Player simulation, with Centralized Algorithms</a></div><div class="lev2 toc-item"><a href="#Creating-the-problem" data-toc-modified-id="Creating-the-problem-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Creating the problem</a></div><div class="lev3 toc-item"><a href="#Parameters-for-the-simulation" data-toc-modified-id="Parameters-for-the-simulation-111"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Parameters for the simulation</a></div><div class="lev3 toc-item"><a href="#Three-MAB-problems-with-Bernoulli-arms" data-toc-modified-id="Three-MAB-problems-with-Bernoulli-arms-112"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>Three MAB problems with Bernoulli arms</a></div><div class="lev3 toc-item"><a href="#Some-RL-algorithms" data-toc-modified-id="Some-RL-algorithms-113"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>Some RL algorithms</a></div><div class="lev2 toc-item"><a href="#Creating-the-EvaluatorMultiPlayers-objects" data-toc-modified-id="Creating-the-EvaluatorMultiPlayers-objects-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Creating the <code>EvaluatorMultiPlayers</code> objects</a></div><div class="lev2 toc-item"><a href="#Solving-the-problem" data-toc-modified-id="Solving-the-problem-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Solving the problem</a></div><div class="lev2 toc-item"><a href="#Plotting-the-results" data-toc-modified-id="Plotting-the-results-14"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Plotting the results</a></div><div class="lev3 toc-item"><a href="#First-problem" data-toc-modified-id="First-problem-141"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>First problem</a></div><div class="lev3 toc-item"><a href="#Second-problem" data-toc-modified-id="Second-problem-142"><span class="toc-item-num">1.4.2&nbsp;&nbsp;</span>Second problem</a></div><div class="lev3 toc-item"><a href="#Third-problem" data-toc-modified-id="Third-problem-143"><span class="toc-item-num">1.4.3&nbsp;&nbsp;</span>Third problem</a></div><div class="lev3 toc-item"><a href="#Comparing-their-performances" data-toc-modified-id="Comparing-their-performances-144"><span class="toc-item-num">1.4.4&nbsp;&nbsp;</span>Comparing their performances</a></div>

# ---
# # An example of a small Multi-Player simulation, with Centralized Algorithms
# 
# First, be sure to be in the main folder, or to have [SMPyBandits](https://github.com/SMPyBandits/SMPyBandits) installed, and import `EvaluatorMultiPlayers` from `Environment` package:

# In[1]:


get_ipython().system('pip install SMPyBandits watermark')
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -p SMPyBandits -a "Lilian Besson"')


# In[2]:


# Local imports
from SMPyBandits.Environment import EvaluatorMultiPlayers, tqdm


# We also need arms, for instance `Bernoulli`-distributed arm:

# In[3]:


# Import arms
from SMPyBandits.Arms import Bernoulli


# And finally we need some single-player and multi-player Reinforcement Learning algorithms:

# In[4]:


# Import algorithms
from SMPyBandits.Policies import *
from SMPyBandits.PoliciesMultiPlayers import *


# In[5]:


# Just improving the ?? in Jupyter. Thanks to https://nbviewer.jupyter.org/gist/minrk/7715212
from __future__ import print_function
from IPython.core import page
def myprint(s):
    try:
        print(s['text/plain'])
    except (KeyError, TypeError):
        print(s)
page.page = myprint


# For instance, this imported the `UCB` algorithm:

# In[6]:


get_ipython().run_line_magic('pinfo', 'UCBalpha')


# As well as the `CentralizedMultiplePlay` multi-player policy:

# In[7]:


get_ipython().run_line_magic('pinfo', 'CentralizedMultiplePlay')


# We also need a collision model. The usual ones are defined in the `CollisionModels` package, and the only one we need is the classical one, where two or more colliding users don't receive any rewards.

# In[8]:


# Collision Models
from SMPyBandits.Environment.CollisionModels import onlyUniqUserGetsReward

get_ipython().run_line_magic('pinfo', 'onlyUniqUserGetsReward')


# ---
# ## Creating the problem

# ### Parameters for the simulation
# - $T = 10000$ is the time horizon,
# - $N = 100$ is the number of repetitions (should be larger to have consistent results),
# - $M = 2$ is the number of players,
# - `N_JOBS = 4` is the number of cores used to parallelize the code.

# In[9]:


HORIZON = 10000
REPETITIONS = 100
NB_PLAYERS = 2
N_JOBS = 4
collisionModel = onlyUniqUserGetsReward


# ### Three MAB problems with Bernoulli arms
# We consider in this example $3$ problems, with `Bernoulli` arms, of different means.
# 
# 1. The first problem is very easy, with two good arms and three arms, with a fixed gap $\Delta = \max_{\mu_i \neq \mu_j}(\mu_{i} - \mu_{j}) = 0.1$.
# 2. The second problem is as easier, with a larger gap.
# 3. Third problem is harder, with a smaller gap, and a very large difference between the two optimal arms and the suboptimal arms.
# 
# > Note: right now, the multi-environments evaluator does not work well for MP policies, if there is a number different of arms in the scenarios. So I use the same number of arms in all the problems.

# In[10]:


ENVIRONMENTS = [  # 1)  Bernoulli arms
        {   # Scenario 1 from [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779]
            "arm_type": Bernoulli,
            "params": [0.3, 0.4, 0.5, 0.6, 0.7]
        },
        {   # Classical scenario
             "arm_type": Bernoulli,
             "params": [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        {   # Harder scenario
             "arm_type": Bernoulli,
             "params": [0.005, 0.01, 0.015, 0.84, 0.85]
        }
    ]


# ### Some RL algorithms
# We will compare Thompson Sampling against $\mathrm{UCB}_1$, using two different centralized policy:
# 
# 1. `CentralizedMultiplePlay` is the naive use of a Bandit algorithm for Multi-Player decision making: at every step, the internal decision making process is used to determine not $1$ arm but $M$ to sample. For UCB-like algorithm, the decision making is based on a $\arg\max$ on UCB-like indexes, usually of the form $I_j(t) = X_j(t) + B_j(t)$, where $X_j(t) = \hat{\mu_j}(t) = \sum_{\tau \leq t} r_j(\tau) / N_j(t)$ is the empirical mean of arm $j$, and $B_j(t)$ is a bias term, of the form $B_j(t) = \sqrt{\frac{\alpha \log(t)}{2 N_j(t)}}$.
# 
# 2. `CentralizedIMP` is very similar, but instead of following the internal decision making for all the decisions, the system uses just the empirical means $X_j(t)$ to determine $M-1$ arms to sample, and the bias-corrected term (i.e., the internal decision making, can be sampling from a Bayesian posterior for instance) is used just for one decision. It is an heuristic, proposed in [[Komiyama, Honda, Nakagawa, 2016]](https://arxiv.org/abs/1506.00779).

# In[12]:


nbArms = len(ENVIRONMENTS[0]['params'])
assert all(len(env['params']) == nbArms for env in ENVIRONMENTS), "Error: not yet support if different environments have different nb of arms"
nbArms

SUCCESSIVE_PLAYERS = [
    CentralizedMultiplePlay(NB_PLAYERS, nbArms, UCBalpha, alpha=1).children,
    CentralizedIMP(NB_PLAYERS, nbArms, UCBalpha, alpha=1).children,
    CentralizedMultiplePlay(NB_PLAYERS, nbArms, Thompson).children,
    CentralizedIMP(NB_PLAYERS, nbArms, Thompson).children
]
SUCCESSIVE_PLAYERS


# The mother class in this case does all the job here, as we use centralized learning.

# In[13]:


OnePlayer = SUCCESSIVE_PLAYERS[0][0]
OnePlayer.nbArms

OneMother = OnePlayer.mother
OneMother
OneMother.nbArms


# Complete configuration for the problem:

# In[14]:


configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Collision model
    "collisionModel": onlyUniqUserGetsReward,
    # --- Arms
    "environment": ENVIRONMENTS,
    # --- Algorithms
    "successive_players": SUCCESSIVE_PLAYERS,
}


# ---
# ## Creating the `EvaluatorMultiPlayers` objects
# We will need to create several objects, as the simulation first runs one policy against each environment, and then aggregate them to compare them.

# In[15]:


get_ipython().run_cell_magic('time', '', 'N_players = len(configuration["successive_players"])\n\n# List to keep all the EvaluatorMultiPlayers objects\nevs = [None] * N_players\nevaluators = [[None] * N_players] * len(configuration["environment"])\n\nfor playersId, players in tqdm(enumerate(configuration["successive_players"]), desc="Creating"):\n    print("\\n\\nConsidering the list of players :\\n", players)\n    conf = configuration.copy()\n    conf[\'players\'] = players\n    evs[playersId] = EvaluatorMultiPlayers(conf)')


# ##  Solving the problem
# Now we can simulate the $2$ environments, for the successive policies. That part can take some time.

# In[16]:


get_ipython().run_cell_magic('time', '', 'for playersId, evaluation in tqdm(enumerate(evs), desc="Policies"):\n    for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):\n        # Evaluate just that env\n        evaluation.startOneEnv(envId, env)\n        # Storing it after simulation is done\n        evaluators[envId][playersId] = evaluation')


# ## Plotting the results
# And finally, visualize them, with the plotting method of a `EvaluatorMultiPlayers` object:

# In[24]:


def plotAll(evaluation, envId):
    evaluation.printFinalRanking(envId)
    # Rewards
    evaluation.plotRewards(envId)
    # Fairness
    #evaluation.plotFairness(envId, fairness="STD")
    # Centralized regret
    evaluation.plotRegretCentralized(envId, subTerms=True)
    #evaluation.plotRegretCentralized(envId, semilogx=True, subTerms=True)
    # Number of switches
    #evaluation.plotNbSwitchs(envId, cumulated=False)
    evaluation.plotNbSwitchs(envId, cumulated=True)
    # Frequency of selection of the best arms
    evaluation.plotBestArmPulls(envId)
    # Number of collisions - not for Centralized* policies
    #evaluation.plotNbCollisions(envId, cumulated=False)
    #evaluation.plotNbCollisions(envId, cumulated=True)
    # Frequency of collision in each arm
    #evaluation.plotFrequencyCollisions(envId, piechart=True)


# In[25]:


import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12.4, 7)


# ### First problem
# $\mu = [0.3, 0.4, 0.5, 0.6, 0.7]$ was an easy Bernoulli problem.

# In[26]:


for playersId in tqdm(range(len(evs)), desc="Policies"):
    evaluation = evaluators[0][playersId]
    plotAll(evaluation, 0)


# ### Second problem
# $\mu = [0.1, 0.3, 0.5, 0.7, 0.9]$ was an easier Bernoulli problem, with larger gap $\Delta = 0.2$.

# In[27]:


for playersId in tqdm(range(len(evs)), desc="Policies"):
    evaluation = evaluators[1][playersId]
    plotAll(evaluation, 1)


# ### Third problem
# $\mu = [0.005, 0.01, 0.015, 0.84, 0.85]$ is an harder Bernoulli problem, as there is a huge gap between suboptimal and optimal arms.

# In[28]:


for playersId in tqdm(range(len(evs)), desc="Policies"):
    evaluation = evaluators[2][playersId]
    plotAll(evaluation, 2)


# ---
# ### Comparing their performances

# In[29]:


def plotCombined(e0, eothers, envId):
    # Centralized regret
    e0.plotRegretCentralized(envId, evaluators=eothers)
    # Fairness
    e0.plotFairness(envId, fairness="STD", evaluators=eothers)
    # Number of switches
    e0.plotNbSwitchsCentralized(envId, cumulated=True, evaluators=eothers)
    # Number of collisions - not for Centralized* policies
    #e0.plotNbCollisions(envId, cumulated=True, evaluators=eothers)


# In[30]:


N = len(configuration["environment"])
for envId, env in enumerate(configuration["environment"]):
   e0, eothers = evaluators[envId][0], evaluators[envId][1:]
   plotCombined(e0, eothers, envId)


# ---
# > That's it for this demo!
