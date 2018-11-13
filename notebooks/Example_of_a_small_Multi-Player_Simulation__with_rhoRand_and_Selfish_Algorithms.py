
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#An-example-of-a-small-Multi-Player-simulation,-with-rhoRand-and-Selfish,-for-different-algorithms" data-toc-modified-id="An-example-of-a-small-Multi-Player-simulation,-with-rhoRand-and-Selfish,-for-different-algorithms-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>An example of a small Multi-Player simulation, with rhoRand and Selfish, for different algorithms</a></div><div class="lev2 toc-item"><a href="#Creating-the-problem" data-toc-modified-id="Creating-the-problem-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Creating the problem</a></div><div class="lev3 toc-item"><a href="#Parameters-for-the-simulation" data-toc-modified-id="Parameters-for-the-simulation-111"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Parameters for the simulation</a></div><div class="lev3 toc-item"><a href="#Three-MAB-problems-with-Bernoulli-arms" data-toc-modified-id="Three-MAB-problems-with-Bernoulli-arms-112"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>Three MAB problems with Bernoulli arms</a></div><div class="lev3 toc-item"><a href="#Some-RL-algorithms" data-toc-modified-id="Some-RL-algorithms-113"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>Some RL algorithms</a></div><div class="lev2 toc-item"><a href="#Creating-the-EvaluatorMultiPlayers-objects" data-toc-modified-id="Creating-the-EvaluatorMultiPlayers-objects-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Creating the <code>EvaluatorMultiPlayers</code> objects</a></div><div class="lev2 toc-item"><a href="#Solving-the-problem" data-toc-modified-id="Solving-the-problem-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Solving the problem</a></div><div class="lev2 toc-item"><a href="#Plotting-the-results" data-toc-modified-id="Plotting-the-results-14"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Plotting the results</a></div><div class="lev3 toc-item"><a href="#First-problem" data-toc-modified-id="First-problem-141"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>First problem</a></div><div class="lev3 toc-item"><a href="#Second-problem" data-toc-modified-id="Second-problem-142"><span class="toc-item-num">1.4.2&nbsp;&nbsp;</span>Second problem</a></div><div class="lev3 toc-item"><a href="#Third-problem" data-toc-modified-id="Third-problem-143"><span class="toc-item-num">1.4.3&nbsp;&nbsp;</span>Third problem</a></div><div class="lev3 toc-item"><a href="#Comparing-their-performances" data-toc-modified-id="Comparing-their-performances-144"><span class="toc-item-num">1.4.4&nbsp;&nbsp;</span>Comparing their performances</a></div>

# ---
# # An example of a small Multi-Player simulation, with rhoRand and Selfish, for different algorithms
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


# For instance, this imported the `Thompson` algorithm:

# In[6]:


get_ipython().run_line_magic('pinfo', 'Thompson')


# As well as the `rhoRand` and `Selfish` multi-player policy:

# In[7]:


get_ipython().run_line_magic('pinfo', 'rhoRand')


# In[8]:


get_ipython().run_line_magic('pinfo', 'Selfish')


# We also need a collision model. The usual ones are defined in the `CollisionModels` package, and the only one we need is the classical one, where two or more colliding users don't receive any rewards.

# In[9]:


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

# In[21]:


HORIZON = 10000
REPETITIONS = 100
NB_PLAYERS = 2
N_JOBS = 1
collisionModel = onlyUniqUserGetsReward


# ### Three MAB problems with Bernoulli arms
# We consider in this example $3$ problems, with `Bernoulli` arms, of different means.
# 
# 1. The first problem is very easy, with two good arms and three arms, with a fixed gap $\Delta = \max_{\mu_i \neq \mu_j}(\mu_{i} - \mu_{j}) = 0.1$.
# 2. The second problem is as easier, with a larger gap.
# 3. Third problem is harder, with a smaller gap, and a very large difference between the two optimal arms and the suboptimal arms.
# 
# > Note: right now, the multi-environments evaluator does not work well for MP policies, if there is a number different of arms in the scenarios. So I use the same number of arms in all the problems.

# In[22]:


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
# We will use Thompson Sampling, $\mathrm{UCB}_1$ and $\mathrm{kl}$-$\mathrm{UCB}$, using two different decentralized policy:
# 
# 1. `rhoRand`: each player starts with a rank $1$, and on collision it selects a new rank $r \sim U([1,M])$. Instead of aiming at the best arm, each player aims at the $r$-th best arm (as given by its UCB index, or posterior sampling if Thompson sampling).
# 
# 2. `Selfish`: each player runs a classical RL algorithm, on the joint reward $\tilde{r}$:
#    $$ \tilde{r}_j(t) = r_j(t) \times (1 - \eta_j(t)) $$
#    where $r_j(t)$ is the reward from the sensing of the $j$-th arm at time $t \geq 1$ and $\eta_j(t)$ is a boolean indicator saying that there were a collision on that arm at time $t$.
#    In other words, if a player chose arm $j$ and was alone on it, it receives $r_j(t)$, and if it was not alone, all players who chose arm $j$ receive $0$.

# In[23]:


nbArms = len(ENVIRONMENTS[0]['params'])
nbArms

SUCCESSIVE_PLAYERS = [
    # UCB alpha=1
    rhoRand(NB_PLAYERS, nbArms, UCBalpha, alpha=1).children,
    Selfish(NB_PLAYERS, nbArms, UCBalpha, alpha=1).children,
    # Thompson Sampling
    rhoRand(NB_PLAYERS, nbArms, Thompson).children,
    Selfish(NB_PLAYERS, nbArms, Thompson).children,
    # Thompson Sampling
    rhoRand(NB_PLAYERS, nbArms, klUCB).children,
    Selfish(NB_PLAYERS, nbArms, klUCB).children,
]
SUCCESSIVE_PLAYERS


# The mother class in this case does not do any centralized learning, as `Selfish` and `rhoRand` are fully decentralized scenario.

# In[24]:


OneMother = SUCCESSIVE_PLAYERS[0][0].mother
OneMother
OneMother.nbArms

OnePlayer = SUCCESSIVE_PLAYERS[0][0]
OnePlayer.nbArms


# Complete configuration for the problem:

# In[25]:


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

# In[26]:


get_ipython().run_cell_magic('time', '', 'N_players = len(configuration["successive_players"])\n\n# List to keep all the EvaluatorMultiPlayers objects\nevs = [None] * N_players\nevaluators = [[None] * N_players] * len(configuration["environment"])\n\nfor playersId, players in tqdm(enumerate(configuration["successive_players"]), desc="Creating"):\n    print("\\n\\nConsidering the list of players :\\n", players)\n    conf = configuration.copy()\n    conf[\'players\'] = players\n    evs[playersId] = EvaluatorMultiPlayers(conf)')


# ##  Solving the problem
# Now we can simulate the $2$ environments, for the successive policies. That part can take some time.

# In[27]:


get_ipython().run_cell_magic('time', '', 'for playersId, evaluation in tqdm(enumerate(evs), desc="Policies"):\n    for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):\n        # Evaluate just that env\n        evaluation.startOneEnv(envId, env)\n        # Storing it after simulation is done\n        evaluators[envId][playersId] = evaluation')


# ## Plotting the results
# And finally, visualize them, with the plotting method of a `EvaluatorMultiPlayers` object:

# In[ ]:


def plotAll(evaluation, envId):
    evaluation.printFinalRanking(envId)
    # Rewards
    evaluation.plotRewards(envId)
    # Fairness
    evaluation.plotFairness(envId, fairness="STD")
    # Centralized regret
    evaluation.plotRegretCentralized(envId, subTerms=True)
    #evaluation.plotRegretCentralized(envId, semilogx=True, subTerms=True)
    # Number of switches
    #evaluation.plotNbSwitchs(envId, cumulated=False)
    evaluation.plotNbSwitchs(envId, cumulated=True)
    # Frequency of selection of the best arms
    evaluation.plotBestArmPulls(envId)
    # Number of collisions
    evaluation.plotNbCollisions(envId, cumulated=False)
    evaluation.plotNbCollisions(envId, cumulated=True)
    # Frequency of collision in each arm
    evaluation.plotFrequencyCollisions(envId, piechart=True)


# In[29]:


import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12.4, 7)


# ### First problem
# $\mu = [0.3, 0.4, 0.5, 0.6, 0.7]$ was an easy Bernoulli problem.

# In[30]:


for playersId in tqdm(range(len(evs)), desc="Policies"):
    evaluation = evaluators[0][playersId]
    plotAll(evaluation, 0)


# ### Second problem
# $\mu = [0.1, 0.3, 0.5, 0.7, 0.9]$ was an easier Bernoulli problem, with larger gap $\Delta = 0.2$.

# In[31]:


for playersId in tqdm(range(len(evs)), desc="Policies"):
    evaluation = evaluators[1][playersId]
    plotAll(evaluation, 1)


# ### Third problem
# $\mu = [0.005, 0.01, 0.015, 0.84, 0.85]$ is an harder Bernoulli problem, as there is a huge gap between suboptimal and optimal arms.

# In[32]:


for playersId in tqdm(range(len(evs)), desc="Policies"):
    evaluation = evaluators[2][playersId]
    plotAll(evaluation, 2)


# ---
# ### Comparing their performances

# In[33]:


def plotCombined(e0, eothers, envId):
    # Centralized regret
    e0.plotRegretCentralized(envId, evaluators=eothers)
    # Fairness
    e0.plotFairness(envId, fairness="STD", evaluators=eothers)
    # Number of switches
    e0.plotNbSwitchsCentralized(envId, cumulated=True, evaluators=eothers)
    # Number of collisions - not for Centralized* policies
    e0.plotNbCollisions(envId, cumulated=True, evaluators=eothers)


# In[34]:


N = len(configuration["environment"])
for envId, env in enumerate(configuration["environment"]):
   e0, eothers = evaluators[envId][0], evaluators[envId][1:]
   plotCombined(e0, eothers, envId)


# ---
# > That's it for this demo!
