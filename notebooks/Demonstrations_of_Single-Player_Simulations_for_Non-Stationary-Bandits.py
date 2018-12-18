
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Demonstrations-of-Single-Player-Simulations-for-Non-Stationary-Bandits" data-toc-modified-id="Demonstrations-of-Single-Player-Simulations-for-Non-Stationary-Bandits-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Demonstrations of Single-Player Simulations for Non-Stationary-Bandits</a></div><div class="lev2 toc-item"><a href="#Creating-the-problem" data-toc-modified-id="Creating-the-problem-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Creating the problem</a></div><div class="lev3 toc-item"><a href="#Parameters-for-the-simulation" data-toc-modified-id="Parameters-for-the-simulation-111"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Parameters for the simulation</a></div><div class="lev3 toc-item"><a href="#Two-MAB-problems-with-Bernoulli-arms-and-piecewise-stationary-means" data-toc-modified-id="Two-MAB-problems-with-Bernoulli-arms-and-piecewise-stationary-means-112"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>Two MAB problems with Bernoulli arms and piecewise stationary means</a></div><div class="lev3 toc-item"><a href="#Some-MAB-algorithms" data-toc-modified-id="Some-MAB-algorithms-113"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>Some MAB algorithms</a></div><div class="lev4 toc-item"><a href="#Parameters-of-the-algorithms" data-toc-modified-id="Parameters-of-the-algorithms-1131"><span class="toc-item-num">1.1.3.1&nbsp;&nbsp;</span>Parameters of the algorithms</a></div><div class="lev4 toc-item"><a href="#Algorithms" data-toc-modified-id="Algorithms-1132"><span class="toc-item-num">1.1.3.2&nbsp;&nbsp;</span>Algorithms</a></div><div class="lev2 toc-item"><a href="#Creating-the-Evaluator-object" data-toc-modified-id="Creating-the-Evaluator-object-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Creating the <code>Evaluator</code> object</a></div><div class="lev2 toc-item"><a href="#Solving-the-problem" data-toc-modified-id="Solving-the-problem-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Solving the problem</a></div><div class="lev3 toc-item"><a href="#First-problem" data-toc-modified-id="First-problem-131"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>First problem</a></div><div class="lev3 toc-item"><a href="#Second-problem" data-toc-modified-id="Second-problem-132"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>Second problem</a></div><div class="lev2 toc-item"><a href="#Plotting-the-results" data-toc-modified-id="Plotting-the-results-14"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Plotting the results</a></div><div class="lev3 toc-item"><a href="#First-problem-with-change-on-only-one-arm-(Local-Restart-should-be-better)" data-toc-modified-id="First-problem-with-change-on-only-one-arm-(Local-Restart-should-be-better)-141"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>First problem with change on only one arm (Local Restart should be better)</a></div><div class="lev3 toc-item"><a href="#Second-problem-with-changes-on-all-arms-(Global-restart-should-be-better)" data-toc-modified-id="Second-problem-with-changes-on-all-arms-(Global-restart-should-be-better)-142"><span class="toc-item-num">1.4.2&nbsp;&nbsp;</span>Second problem with changes on all arms (Global restart should be better)</a></div>

# ---
# # Demonstrations of Single-Player Simulations for Non-Stationary-Bandits
# 
# This notebook shows how to 1) **define**, 2) **launch**, and 3) **plot the results** of numerical simulations of piecewise stationary (multi-armed) bandits problems using my framework [SMPyBandits](https://github.com/SMPyBandits/SMPyBandits).
# For more details on the maths behind this problem, see this page in the documentation: [SMPyBandits.GitHub.io/NonStationaryBandits.html](https://smpybandits.github.io/NonStationaryBandits.html).
# 
# First, be sure to be in the main folder, or to have [SMPyBandits](https://github.com/SMPyBandits/SMPyBandits) installed, and import `Evaluator` from `Environment` package.
# 
# <span style="color:red">WARNING</span>
# If you are running this notebook locally, in the [`notebooks`](https://github.com/SMPyBandits/SMPyBandits/tree/master/notebooks) folder in the [`SMPyBandits`](https://github.com/SMPyBandits/SMPyBandits/) source, you need to do:

# In[3]:


import sys
sys.path.insert(0, '..')


# If you are running this notebook elsewhere, `SMPyBandits` can be `pip install`ed easily:
# (this is especially true if you run this notebook from Google Colab or MyBinder).

# In[4]:


try:
    import SMPyBandits
except ImportError:
    get_ipython().system('pip3 install SMPyBandits')


# Let's just check the versions of the installed modules:

# In[49]:


get_ipython().system('pip3 install watermark > /dev/null')


# In[50]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -p SMPyBandits,numpy,matplotlib -a "Lilian Besson"')


# We can now import all the modules we need for this demonstration.

# In[8]:


import numpy as np


# In[9]:


# Local imports
from SMPyBandits.Environment import Evaluator, tqdm


# In[51]:


# Large figures for pretty notebooks
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (19.80, 10.80)


# We also need arms, for instance `Bernoulli`-distributed arm:

# In[52]:


# Import arms
from SMPyBandits.Arms import Bernoulli


# And finally we need some single-player Reinforcement Learning algorithms:

# In[53]:


# Import algorithms
from SMPyBandits.Policies import *


# ---
# ## Creating the problem

# ### Parameters for the simulation
# - $T = 1000$ is the time horizon,
# - $N = 50$ is the number of repetitions,
# - `N_JOBS = 4` is the number of cores used to parallelize the code.

# In[97]:


from multiprocessing import cpu_count
CPU_COUNT = cpu_count()
N_JOBS = CPU_COUNT if CPU_COUNT <= 4 else CPU_COUNT - 4

print(f"Using {N_JOBS} jobs in parallel...")


# In[98]:


HORIZON = 1000
REPETITIONS = 100

print(f"Using T = {HORIZON}, and N = {REPETITIONS} repetitions")


# ### Two MAB problems with Bernoulli arms and piecewise stationary means
# 
# We consider in this example $2$ problems, with `Bernoulli` arms, of different piecewise stationary means.
# 
# 1. The first problem has changes on only one arm at every breakpoint times,
# 2. The second problem has changes on all arms at every breakpoint times.

# In[29]:


ENVIRONMENTS = []

ENVIRONMENTS += [
    {   # A simple piece-wise stationary problem
        "arm_type": Bernoulli,
        "params": {
            "listOfMeans": [
                [0.2, 0.5, 0.9],  # 0    to 399
                [0.2, 0.2, 0.9],  # 400  to 799
                [0.2, 0.2, 0.1],  # 800  to 1199
                [0.7, 0.2, 0.1],  # 1200 to 1599
                [0.7, 0.5, 0.1],  # 1600 to end
            ],
            "changePoints": [
                int(0    * HORIZON / 2000.0),
                int(400  * HORIZON / 2000.0),
                int(800  * HORIZON / 2000.0),
                int(1200 * HORIZON / 2000.0),
                int(1600 * HORIZON / 2000.0),
                # 20000,  # XXX larger than horizon, just to see if it is a problem?
            ],
        }
    },
]

# Pb 2 changes are on all or almost arms at a time
ENVIRONMENTS += [
    {   # A simple piece-wise stationary problem
        "arm_type": Bernoulli,
        "params": {
            "listOfMeans": [
                [0.4, 0.5, 0.9],  # 0    to 399
                [0.5, 0.4, 0.7],  # 400  to 799
                [0.6, 0.3, 0.5],  # 800  to 1199
                [0.7, 0.2, 0.3],  # 1200 to 1599
                [0.8, 0.1, 0.1],  # 1600 to end
            ],
            "changePoints": [
                int(0    * HORIZON / 2000.0),
                int(400  * HORIZON / 2000.0),
                int(800  * HORIZON / 2000.0),
                int(1200 * HORIZON / 2000.0),
                int(1600 * HORIZON / 2000.0),
                # 20000,  # XXX larger than horizon, just to see if it is a problem?
            ],
        }
    },
]

list_nb_arms = [len(env["params"]["listOfMeans"][0]) for env in ENVIRONMENTS]
NB_ARMS = max(list_nb_arms)
assert all(n == NB_ARMS for n in list_nb_arms), "Error: it is NOT supported to have successive problems with a different number of arms!"
print(f"==> Using K = {NB_ARMS} arms")

NB_BREAK_POINTS = max(len(env["params"]["changePoints"]) for env in ENVIRONMENTS)
print(f"==> Using Upsilon_T = {NB_BREAK_POINTS} change points")

CHANGE_POINTS = np.unique(np.array(list(set.union(*(set(env["params"]["changePoints"]) for env in ENVIRONMENTS)))))
print(f"==> Using the following {list(CHANGE_POINTS)} change points")


# ### Some MAB algorithms
# 
# We want compare some classical MAB algorithms ($\mathrm{UCB}_1$, Thompson Sampling and $\mathrm{kl}$-$\mathrm{UCB}$) that are designed to solve stationary problems against other algorithms designed to solve piecewise-stationary problems.

# #### Parameters of the algorithms

# In[30]:


klucb = klucb_mapping.get(str(ENVIRONMENTS[0]['arm_type']), klucbBern)
klucb


# In[31]:


EPSS   = [0.1, 0.05]

ALPHAS = [1]

TAUS   = [
        500, 1000, 2000,
        int(2 * np.sqrt(HORIZON * np.log(HORIZON) / (1 + NB_BREAK_POINTS))),  # "optimal" value according to [Garivier & Moulines, 2008]
    ]

GAMMAS = [
        0.2, 0.4, 0.6, 0.8,
        0.95, 0.99,
        max(min(1, (1 - np.sqrt((1 + NB_BREAK_POINTS) / HORIZON)) / 4.), 0),  # "optimal" value according to [Garivier & Moulines, 2008]
    ]

WINDOW_SIZE = 800 if HORIZON >= 10000 else 80


# #### Algorithms

# In[32]:


POLICIES = [  # XXX Regular adversarial bandits algorithms!
        { "archtype": Exp3PlusPlus, "params": {} },
    ] + [  # XXX Regular stochastic bandits algorithms!
        { "archtype": UCBalpha, "params": { "alpha": 1, } },
        { "archtype": klUCB, "params": { "klucb": klucb, } },
        { "archtype": Thompson, "params": { "posterior": Beta, } },
    ] + [  # XXX This is still highly experimental!
        { "archtype": DiscountedThompson, "params": { "posterior": DiscountedBeta, "gamma": gamma } }
        # for gamma in GAMMAS
        for gamma in [0.99, 0.9, 0.7]
    ] + [  # --- The Exp3R algorithm works reasonably well
        { "archtype": Exp3R, "params": { "horizon": HORIZON, } }
    ] + [  # --- XXX The Exp3RPlusPlus variant of Exp3R algorithm works also reasonably well
        { "archtype": Exp3RPlusPlus, "params": { "horizon": HORIZON, } }
    ] + [  # --- XXX Test a few CD-MAB algorithms that need to know NB_BREAK_POINTS
        { "archtype": archtype, "params": {
            "horizon": HORIZON,
            "max_nb_random_events": NB_BREAK_POINTS,
            "policy": policy,
            "per_arm_restart": per_arm_restart,
        } }
        for archtype in [
            CUSUM_IndexPolicy,
            PHT_IndexPolicy,  # OK PHT_IndexPolicy is very much like CUSUM
        ]
        for policy in [
            # UCB,  # XXX comment to only test klUCB
            klUCB,
        ]
        for per_arm_restart in [
            True,  # Per-arm restart XXX comment to only test global arm
            False, # Global restart XXX seems more efficient? (at least more memory efficient!)
        ]
    ] + [  # --- XXX Test a few CD-MAB algorithms
        { "archtype": archtype, "params": {
            "horizon": HORIZON,
            "policy": policy,
            "per_arm_restart": per_arm_restart,
        } }
        for archtype in [
            BernoulliGLR_IndexPolicy,  # OK BernoulliGLR_IndexPolicy is very much like CUSUM
            GaussianGLR_IndexPolicy,  # OK GaussianGLR_IndexPolicy is very much like Bernoulli GLR
            SubGaussianGLR_IndexPolicy, # OK SubGaussianGLR_IndexPolicy is very much like Gaussian GLR
        ]
        for policy in [
            # UCB,  # XXX comment to only test klUCB
            klUCB,
        ]
        for per_arm_restart in [
            True,  # Per-arm restart XXX comment to only test global arm
            False, # Global restart XXX seems more efficient? (at least more memory efficient!)
        ]
    ] + [  # --- XXX The Monitored_IndexPolicy with specific tuning of the input parameters
        { "archtype": Monitored_IndexPolicy, "params": { "horizon": HORIZON, "w": WINDOW_SIZE, "b": np.sqrt(WINDOW_SIZE/2 * np.log(2 * NB_ARMS * HORIZON**2)), "policy": policy, "per_arm_restart": per_arm_restart, } }
        for per_arm_restart in [
            True,  # Per-arm restart XXX comment to only test global arm
            False, # Global restart XXX seems more efficient? (at least more memory efficient!)
        ]
        for policy in [
            # UCB,
            klUCB,  # XXX comment to only test UCB
        ]
    ] + [  # --- DONE The SW_UCB_Hash algorithm works fine!
        { "archtype": SWHash_IndexPolicy, "params": { "alpha": alpha, "lmbda": lmbda, "policy": UCB } }
        for alpha in ALPHAS
        for lmbda in [1]  # [0.1, 0.5, 1, 5, 10]
    ] + [ # --- # XXX experimental other version of the sliding window algorithm, knowing the horizon
        { "archtype": SWUCBPlus, "params": { "horizon": HORIZON, "alpha": alpha, } }
        for alpha in ALPHAS
    ] + [ # --- # XXX experimental discounted UCB algorithm, knowing the horizon
        { "archtype": DiscountedUCBPlus, "params": { "max_nb_random_events": max_nb_random_events, "alpha": alpha, "horizon": HORIZON, } }
        for alpha in ALPHAS
        for max_nb_random_events in [NB_BREAK_POINTS]
    ] + [  # --- DONE the OracleSequentiallyRestartPolicy with klUCB/UCB policy works quite well, but NOT optimally!
        { "archtype": OracleSequentiallyRestartPolicy, "params": { "changePoints": CHANGE_POINTS, "policy": policy,
            "per_arm_restart": per_arm_restart,
            # "full_restart_when_refresh": full_restart_when_refresh,
        } }
        for policy in [
            UCB,
            klUCB,  # XXX comment to only test UCB
            Exp3PlusPlus,  # XXX comment to only test UCB
        ]
        for per_arm_restart in [True]  #, False]
        # for full_restart_when_refresh in [True, False]
    ]


# The complete configuration for the problems and these algorithms is then a simple dictionary:

# In[33]:


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
    # --- Random events
    "nb_break_points": NB_BREAK_POINTS,
    # --- Should we plot the lower-bounds or not?
    "plot_lowerbound": False,  # XXX Default
    # --- Cache rewards: use the same random rewards for all algorithms (more fair comparison!)
    "cache_rewards": True,
}

configuration


# In[70]:


# (almost) unique hash from the configuration
hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))
print(f"This configuraiton has a hash value = {hashvalue}")


# In[77]:


import os, os.path


# In[79]:


subfolder = "SP__K{}_T{}_N{}__{}_algos".format(env.nbArms, configuration['horizon'], configuration['repetitions'], len(configuration['policies']))
PLOT_DIR = "plots"
plot_dir = os.path.join(PLOT_DIR, subfolder)

# Create the sub folder
if os.path.isdir(plot_dir):
    print("{} is already a directory here...".format(plot_dir))
elif os.path.isfile(plot_dir):
    raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
else:
    os.mkdir(plot_dir)

print(f"Using sub folder = '{subfolder}' and plotting in '{plot_dir}'...")


# In[83]:


mainfig = os.path.join(plot_dir, "main")
print(f"Using main figure name as '{mainfig}_{hashvalue}'...")


# ---
# ## Creating the `Evaluator` object

# In[34]:


evaluation = Evaluator(configuration)


# ##  Solving the problem
# Now we can simulate all the environments.
# 
# <span style="color:red">WARNING</span>
# That part takes some time, most stationary algorithms run with a time complexity linear in the horizon (ie., time takes $\mathcal{O}(T)$) and most piecewise stationary algorithms run with a time complexity **square** in the horizon (ie., time takes $\mathcal{O}(T^2)$).

# ### First problem

# In[35]:


get_ipython().run_cell_magic('time', '', 'envId = 0\nenv = evaluation.envs[envId]\n# Show the problem\n%time evaluation.plotHistoryOfMeans(envId)\n# Evaluate just that env\n%time evaluation.startOneEnv(envId, env)')


# ### Second problem

# In[36]:


get_ipython().run_cell_magic('time', '', 'envId = 1\nenv = evaluation.envs[envId]\n# Show the problem\n%time evaluation.plotHistoryOfMeans(envId)\n# Evaluate just that env\n%time evaluation.startOneEnv(envId, env)')


# ## Plotting the results
# And finally, visualize them, with the plotting method of a `Evaluator` object:

# In[37]:


def printAll(evaluation, envId):
    print("\nGiving the vector of final regrets ...")
    evaluation.printLastRegrets(envId)
    print("\nGiving the final ranks ...")
    evaluation.printFinalRanking(envId)
    print("\nGiving the mean and std running times ...")
    evaluation.printRunningTimes(envId)
    print("\nGiving the mean and std memory consumption ...")
    evaluation.printMemoryConsumption(envId)


# In[89]:


def plotAll(evaluation, envId, mainfig=None):
    savefig = mainfig
    if savefig is not None: savefig = f"{mainfig}__LastRegrets__env{envId+1}-{len(evaluation.envs)}"
    print("\nPlotting a boxplot of the final regrets ...")
    evaluation.plotLastRegrets(envId, boxplot=True, savefig=savefig)

    if savefig is not None: savefig = f"{mainfig}__RunningTimes__env{envId+1}-{len(evaluation.envs)}"
    print("\nPlotting the mean and std running times ...")
    evaluation.plotRunningTimes(envId, savefig=savefig)

    if savefig is not None: savefig = f"{mainfig}__MemoryConsumption__env{envId+1}-{len(evaluation.envs)}"
    print("\nPlotting the mean and std memory consumption ...")
    evaluation.plotMemoryConsumption(envId, savefig=savefig)

    if savefig is not None: savefig = f"{mainfig}__Regrets__env{envId+1}-{len(evaluation.envs)}"
    print("\nPlotting the mean regrets ...")
    evaluation.plotRegrets(envId, savefig=savefig)

    if savefig is not None: savefig = f"{mainfig}__MeanReward__env{envId+1}-{len(evaluation.envs)}"
    print("\nPlotting the mean rewards ...")
    evaluation.plotRegrets(envId, meanReward=True, savefig=savefig)

    if savefig is not None: savefig = f"{mainfig}__BestArmPulls__env{envId+1}-{len(evaluation.envs)}"
    print("\nPlotting the best arm pulls ...")
    evaluation.plotBestArmPulls(envId, savefig=savefig)

    if savefig is not None: savefig = f"{mainfig}__LastRegrets__env{envId+1}-{len(evaluation.envs)}"
    print("\nPlotting an histogram of the final regrets ...")
    evaluation.plotLastRegrets(envId, subplots=True, sharex=True, sharey=False, savefig=savefig)


# In[65]:


evaluation.nb_break_points


# ### First problem with change on only one arm (Local Restart should be better)
# 
# Let's first print the results then plot them:

# In[84]:


envId = 0


# In[85]:


_ = evaluation.plotHistoryOfMeans(envId, savefig=f"{mainfig}__HistoryOfMeans__env{envId+1}-{len(evaluation.envs)}")


# In[42]:


_ = printAll(evaluation, envId)


# In[88]:


_ = plotAll(evaluation, envId, mainfig=mainfig)


# ### Second problem with changes on all arms (Global restart should be better)
# 
# Let's first print the results then plot them:

# In[90]:


envId = 1


# In[91]:


_ = evaluation.plotHistoryOfMeans(envId, savefig=f"{mainfig}__HistoryOfMeans__env{envId+1}-{len(evaluation.envs)}")


# In[92]:


_ = printAll(evaluation, envId)


# In[93]:


_ = plotAll(evaluation, envId, mainfig=mainfig)


# ---
# > That's it for this demo!
