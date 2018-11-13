
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#An-example-of-a-small-Single-Player-simulation" data-toc-modified-id="An-example-of-a-small-Single-Player-simulation-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>An example of a small Single-Player simulation</a></div><div class="lev2 toc-item"><a href="#Creating-the-problem" data-toc-modified-id="Creating-the-problem-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Creating the problem</a></div><div class="lev3 toc-item"><a href="#Parameters-for-the-simulation" data-toc-modified-id="Parameters-for-the-simulation-111"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Parameters for the simulation</a></div><div class="lev3 toc-item"><a href="#Some-MAB-problem-with-Bernoulli-arms" data-toc-modified-id="Some-MAB-problem-with-Bernoulli-arms-112"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>Some MAB problem with Bernoulli arms</a></div><div class="lev3 toc-item"><a href="#Some-RL-algorithms" data-toc-modified-id="Some-RL-algorithms-113"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>Some RL algorithms</a></div><div class="lev2 toc-item"><a href="#Creating-the-Evaluator-object" data-toc-modified-id="Creating-the-Evaluator-object-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Creating the <code>Evaluator</code> object</a></div><div class="lev2 toc-item"><a href="#Solving-the-problem" data-toc-modified-id="Solving-the-problem-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Solving the problem</a></div><div class="lev2 toc-item"><a href="#Plotting-the-results" data-toc-modified-id="Plotting-the-results-14"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Plotting the results</a></div><div class="lev3 toc-item"><a href="#First-problem" data-toc-modified-id="First-problem-141"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>First problem</a></div><div class="lev3 toc-item"><a href="#Second-problem" data-toc-modified-id="Second-problem-142"><span class="toc-item-num">1.4.2&nbsp;&nbsp;</span>Second problem</a></div><div class="lev3 toc-item"><a href="#Third-problem" data-toc-modified-id="Third-problem-143"><span class="toc-item-num">1.4.3&nbsp;&nbsp;</span>Third problem</a></div>

# ---
# # An example of a small Single-Player simulation
# 
# First, be sure to be in the main folder, or to have [SMPyBandits](https://github.com/SMPyBandits/SMPyBandits) installed, and import `Evaluator` from `Environment` package:

# In[2]:


get_ipython().system('pip install SMPyBandits watermark')
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -p SMPyBandits -a "Lilian Besson"')


# In[3]:


# Local imports
from SMPyBandits.Environment import Evaluator, tqdm


# We also need arms, for instance `Bernoulli`-distributed arm:

# In[4]:


# Import arms
from SMPyBandits.Arms import Bernoulli


# And finally we need some single-player Reinforcement Learning algorithms:

# In[5]:


# Import algorithms
from SMPyBandits.Policies import *


# For instance, this imported the [`UCB` algorithm](https://en.wikipedia.org/wiki/Multi-armed_bandit#Bandit_strategies) is the `UCBalpha` class:

# In[6]:


# Just improving the ?? in Jupyter. Thanks to https://nbviewer.jupyter.org/gist/minrk/7715212
from __future__ import print_function
from IPython.core import page
def myprint(s):
    try:
        print(s['text/plain'])
    except (KeyError, TypeError):
        print(s)
page.page = myprint


# In[7]:


get_ipython().run_line_magic('pinfo', 'UCBalpha')


# With more details, here the code:

# In[8]:


get_ipython().run_line_magic('pinfo2', 'UCBalpha')


# ---
# ## Creating the problem

# ### Parameters for the simulation
# - $T = 10000$ is the time horizon,
# - $N = 10$ is the number of repetitions,
# - `N_JOBS = 4` is the number of cores used to parallelize the code.

# In[9]:


HORIZON = 10000
REPETITIONS = 10
N_JOBS = 1


# ### Some MAB problem with Bernoulli arms
# We consider in this example $3$ problems, with `Bernoulli` arms, of different means.

# In[10]:


ENVIRONMENTS = [  # 1)  Bernoulli arms
        {   # A very easy problem, but it is used in a lot of articles
            "arm_type": Bernoulli,
            "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        {   # An other problem, best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3 - 0.6) and very good arms (0.78, 0.8, 0.82)
            "arm_type": Bernoulli,
            "params": [0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.795, 0.8, 0.805]
        },
        {   # A very hard problem, as used in [Cappé et al, 2012]
            "arm_type": Bernoulli,
            "params": [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.1]
        },
    ]


# ### Some RL algorithms
# We compare Thompson Sampling against $\mathrm{UCB}_1$, and $\mathrm{kl}-\mathrm{UCB}$.

# In[11]:


POLICIES = [
        # --- UCB1 algorithm
        {
            "archtype": UCBalpha,
            "params": {
                "alpha": 1
            }
        },
        {
            "archtype": UCBalpha,
            "params": {
                "alpha": 0.5  # Smallest theoretically acceptable value
            }
        },
        # --- Thompson algorithm
        {
            "archtype": Thompson,
            "params": {}
        },
        # --- KL algorithms, here only klUCB
        {
            "archtype": klUCB,
            "params": {}
        },
        # --- BayesUCB algorithm
        {
            "archtype": BayesUCB,
            "params": {}
        },
    ]


# Complete configuration for the problem:

# In[12]:


configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Arms
    "environment": ENVIRONMENTS,
    # --- Algorithms
    "policies": POLICIES,
}
configuration


# ---
# ## Creating the `Evaluator` object

# In[13]:


evaluation = Evaluator(configuration)


# ##  Solving the problem
# Now we can simulate all the $3$ environments. That part can take some time.

# In[14]:


for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
    # Evaluate just that env
    evaluation.startOneEnv(envId, env)


# ## Plotting the results
# And finally, visualize them, with the plotting method of a `Evaluator` object:

# In[15]:


def plotAll(evaluation, envId):
    evaluation.printFinalRanking(envId)
    evaluation.plotRegrets(envId)
    evaluation.plotRegrets(envId, semilogx=True)
    evaluation.plotRegrets(envId, meanReward=True)
    evaluation.plotBestArmPulls(envId)


# In[16]:


evaluation.nb_break_points


# In[20]:


import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12.4, 7)


# ### First problem
# $[B(0.1), B(0.2), B(0.3), B(0.4), B(0.5), B(0.6), B(0.7), B(0.8), B(0.9)]$

# In[21]:


_ = plotAll(evaluation, 0)


# ### Second problem
# $[B(0.01), B(0.02), B(0.3), B(0.4), B(0.5), B(0.6), B(0.795), B(0.8), B(0.805)]$

# In[22]:


plotAll(evaluation, 1)


# ### Third problem
# $[B(0.01), B(0.01), B(0.01), B(0.02), B(0.02), B(0.02), B(0.05), B(0.05), B(0.1)]$

# In[23]:


plotAll(evaluation, 2)


# ---
# > That's it for this demo!
