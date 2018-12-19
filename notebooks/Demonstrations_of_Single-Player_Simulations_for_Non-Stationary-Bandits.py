
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Demonstrations-of-Single-Player-Simulations-for-Non-Stationary-Bandits" data-toc-modified-id="Demonstrations-of-Single-Player-Simulations-for-Non-Stationary-Bandits-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Demonstrations of Single-Player Simulations for Non-Stationary-Bandits</a></div><div class="lev2 toc-item"><a href="#Creating-the-problem" data-toc-modified-id="Creating-the-problem-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Creating the problem</a></div><div class="lev3 toc-item"><a href="#Parameters-for-the-simulation" data-toc-modified-id="Parameters-for-the-simulation-111"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Parameters for the simulation</a></div><div class="lev3 toc-item"><a href="#Two-MAB-problems-with-Bernoulli-arms-and-piecewise-stationary-means" data-toc-modified-id="Two-MAB-problems-with-Bernoulli-arms-and-piecewise-stationary-means-112"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>Two MAB problems with Bernoulli arms and piecewise stationary means</a></div><div class="lev3 toc-item"><a href="#Some-MAB-algorithms" data-toc-modified-id="Some-MAB-algorithms-113"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>Some MAB algorithms</a></div><div class="lev4 toc-item"><a href="#Parameters-of-the-algorithms" data-toc-modified-id="Parameters-of-the-algorithms-1131"><span class="toc-item-num">1.1.3.1&nbsp;&nbsp;</span>Parameters of the algorithms</a></div><div class="lev4 toc-item"><a href="#Algorithms" data-toc-modified-id="Algorithms-1132"><span class="toc-item-num">1.1.3.2&nbsp;&nbsp;</span>Algorithms</a></div><div class="lev2 toc-item"><a href="#Checking-if-the-problems-are-too-hard-or-not" data-toc-modified-id="Checking-if-the-problems-are-too-hard-or-not-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Checking if the problems are too hard or not</a></div><div class="lev2 toc-item"><a href="#Creating-the-Evaluator-object" data-toc-modified-id="Creating-the-Evaluator-object-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Creating the <code>Evaluator</code> object</a></div><div class="lev2 toc-item"><a href="#Solving-the-problem" data-toc-modified-id="Solving-the-problem-14"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Solving the problem</a></div><div class="lev3 toc-item"><a href="#First-problem" data-toc-modified-id="First-problem-141"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>First problem</a></div><div class="lev3 toc-item"><a href="#Second-problem" data-toc-modified-id="Second-problem-142"><span class="toc-item-num">1.4.2&nbsp;&nbsp;</span>Second problem</a></div><div class="lev2 toc-item"><a href="#Plotting-the-results" data-toc-modified-id="Plotting-the-results-15"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Plotting the results</a></div><div class="lev3 toc-item"><a href="#First-problem-with-change-on-only-one-arm-(Local-Restart-should-be-better)" data-toc-modified-id="First-problem-with-change-on-only-one-arm-(Local-Restart-should-be-better)-151"><span class="toc-item-num">1.5.1&nbsp;&nbsp;</span>First problem with change on only one arm (Local Restart should be better)</a></div><div class="lev3 toc-item"><a href="#Second-problem-with-changes-on-all-arms-(Global-restart-should-be-better)" data-toc-modified-id="Second-problem-with-changes-on-all-arms-(Global-restart-should-be-better)-152"><span class="toc-item-num">1.5.2&nbsp;&nbsp;</span>Second problem with changes on all arms (Global restart should be better)</a></div>

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

# In[1]:


import sys
sys.path.insert(0, '..')


# If you are running this notebook elsewhere, `SMPyBandits` can be `pip install`ed easily:
# (this is especially true if you run this notebook from Google Colab or MyBinder).

# In[2]:


try:
    import SMPyBandits
except ImportError:
    get_ipython().system('pip3 install SMPyBandits')


# Let's just check the versions of the installed modules:

# In[3]:


get_ipython().system('pip3 install watermark > /dev/null')


# In[4]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -p SMPyBandits,numpy,matplotlib -a "Lilian Besson"')


# We can now import all the modules we need for this demonstration.

# In[5]:


import numpy as np


# In[44]:


FIGSIZE = (19.80, 10.80)
DPI = 160


# In[45]:


# Large figures for pretty notebooks
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = FIGSIZE
mpl.rcParams['figure.dpi'] = DPI


# In[46]:


# Local imports
from SMPyBandits.Environment import Evaluator, tqdm


# In[47]:


# Large figures for pretty notebooks
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = FIGSIZE
mpl.rcParams['figure.dpi'] = DPI


# In[48]:


# Large figures for pretty notebooks
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = FIGSIZE
mpl.rcParams['figure.dpi'] = DPI


# We also need arms, for instance `Bernoulli`-distributed arm:

# In[10]:


# Import arms
from SMPyBandits.Arms import Bernoulli


# And finally we need some single-player Reinforcement Learning algorithms:

# In[11]:


# Import algorithms
from SMPyBandits.Policies import *


# ---
# ## Creating the problem

# ### Parameters for the simulation
# - $T = 2000$ is the time horizon,
# - $N = 100$ is the number of repetitions, or 1 to debug the simulations,
# - `N_JOBS = 4` is the number of cores used to parallelize the code,
# - $5$ piecewise stationary sequences will have length 400

# In[49]:


from multiprocessing import cpu_count
CPU_COUNT = cpu_count()
N_JOBS = CPU_COUNT if CPU_COUNT <= 4 else CPU_COUNT - 4

print("Using {} jobs in parallel...".format(N_JOBS))


# In[51]:


HORIZON = 2000
REPETITIONS = 50

print("Using T = {}, and N = {} repetitions".format(HORIZON, REPETITIONS))


# ### Two MAB problems with Bernoulli arms and piecewise stationary means
# 
# We consider in this example $2$ problems, with `Bernoulli` arms, of different piecewise stationary means.
# 
# 1. The first problem has changes on only one arm at every breakpoint times,
# 2. The second problem has changes on all arms at every breakpoint times.

# In[53]:


ENVIRONMENTS = []


# In[54]:


ENVIRONMENT_0 = {   # A simple piece-wise stationary problem
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
        ],
    }
}


# In[55]:


# Pb 2 changes are on all or almost arms at a time
ENVIRONMENT_1 =  {   # A simple piece-wise stationary problem
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
        ],
    }
}


# In[56]:


ENVIRONMENTS = [
    ENVIRONMENT_0,
    ENVIRONMENT_1,
]

list_nb_arms = [len(env["params"]["listOfMeans"][0]) for env in ENVIRONMENTS]
NB_ARMS = max(list_nb_arms)
assert all(n == NB_ARMS for n in list_nb_arms), "Error: it is NOT supported to have successive problems with a different number of arms!"
print("==> Using K = {} arms".format(NB_ARMS))

NB_BREAK_POINTS = max(len(env["params"]["changePoints"]) for env in ENVIRONMENTS)
print("==> Using Upsilon_T = {} change points".format(NB_BREAK_POINTS))

CHANGE_POINTS = np.unique(np.array(list(set.union(*(set(env["params"]["changePoints"]) for env in ENVIRONMENTS)))))
print("==> Using the following {} change points".format(list(CHANGE_POINTS)))


# ### Some MAB algorithms
# 
# We want compare some classical MAB algorithms ($\mathrm{UCB}_1$, Thompson Sampling and $\mathrm{kl}$-$\mathrm{UCB}$) that are designed to solve stationary problems against other algorithms designed to solve piecewise-stationary problems.

# #### Parameters of the algorithms

# In[15]:


klucb = klucb_mapping.get(str(ENVIRONMENTS[0]['arm_type']), klucbBern)
klucb


# In[61]:


WINDOW_SIZE = int(80 * np.ceil(HORIZON / 10000))
print("M-UCB will use a window of size {}".format(WINDOW_SIZE))


# #### Algorithms

# In[62]:


POLICIES = [  # XXX Regular adversarial bandits algorithms!
        { "archtype": Exp3PlusPlus, "params": {} },
    ] + [  # XXX Regular stochastic bandits algorithms!
        { "archtype": UCBalpha, "params": { "alpha": 1, } },
        { "archtype": klUCB, "params": { "klucb": klucb, } },
        { "archtype": Thompson, "params": { "posterior": Beta, } },
    ] + [  # XXX This is still highly experimental!
        { "archtype": DiscountedThompson, "params": {
            "posterior": DiscountedBeta, "gamma": gamma
        } }
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
        { "archtype": Monitored_IndexPolicy, "params": {
            "horizon": HORIZON,
            "w": WINDOW_SIZE,
            "b": np.sqrt(WINDOW_SIZE/2 * np.log(2 * NB_ARMS * HORIZON**2)),
            "policy": policy,
            "per_arm_restart": per_arm_restart,
        } }
        for policy in [
            # UCB,
            klUCB,  # XXX comment to only test UCB
        ]
        for per_arm_restart in [
            True,  # Per-arm restart XXX comment to only test global arm
            False, # Global restart XXX seems more efficient? (at least more memory efficient!)
        ]
    ] + [  # --- DONE The SW_UCB_Hash algorithm works fine!
        { "archtype": SWHash_IndexPolicy, "params": {
            "alpha": alpha, "lmbda": lmbda, "policy": UCB,
        } }
        for alpha in [1.0]
        for lmbda in [1]
    ] + [ # --- # XXX experimental other version of the sliding window algorithm, knowing the horizon
        { "archtype": SWUCBPlus, "params": {
            "horizon": HORIZON, "alpha": alpha,
        } }
        for alpha in [1.0]
    ] + [ # --- # XXX experimental discounted UCB algorithm, knowing the horizon
        { "archtype": DiscountedUCBPlus, "params": {
            "max_nb_random_events": max_nb_random_events, "alpha": alpha, "horizon": HORIZON,
        } }
        for alpha in [1.0]
        for max_nb_random_events in [NB_BREAK_POINTS]
    ] + [  # --- DONE the OracleSequentiallyRestartPolicy with klUCB/UCB policy works quite well, but NOT optimally!
        { "archtype": OracleSequentiallyRestartPolicy, "params": {
            "changePoints": CHANGE_POINTS, "policy": policy,
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

# In[64]:


configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 0,      # Max joblib verbosity
    # --- Arms
    "environment": ENVIRONMENTS,
    # --- Algorithms
    "policies": POLICIES,
    # --- Random events
    "nb_break_points": NB_BREAK_POINTS,
    # --- Should we plot the lower-bounds or not?
    "plot_lowerbound": False,  # XXX Default
}

configuration


# In[65]:


# (almost) unique hash from the configuration
hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))
print("This configuration has a hash value = {}".format(hashvalue))


# In[66]:


import os, os.path


# In[67]:


subfolder = "SP__K{}_T{}_N{}__{}_algos".format(NB_ARMS, HORIZON, REPETITIONS, len(POLICIES))
PLOT_DIR = "plots"
plot_dir = os.path.join(PLOT_DIR, subfolder)

# Create the sub folder
if os.path.isdir(plot_dir):
    print("{} is already a directory here...".format(plot_dir))
elif os.path.isfile(plot_dir):
    raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
else:
    os.mkdir(plot_dir)

print("Using sub folder = '{}' and plotting in '{}'...".format(subfolder, plot_dir))


# In[68]:


mainfig = os.path.join(plot_dir, "main")
print("Using main figure name as '{}_{}'...".format(mainfig, hashvalue))


# ---
# ## Checking if the problems are too hard or not

# If we assume we have a result that bounds the delay of the Change-Detection algorithm by a certain quantity $D$, we can check that the sequence lengths (ie, $\tau_{m+1}-\tau_m$) are *large* enough for the CD-klUCB algorithm (our proposal) to be efficient.

# In[69]:


def lowerbound_on_sequence_length(horizon, gap):
    r""" A function that computes the lower-bound (we will find) on the sequence length to have a reasonable bound on the delay of our change-detection algorithm.

    - It returns the smallest possible sequence length :math:`L = \tau_{m+1} - \tau_m` satisfying:

    .. math:: L \geq \frac{8}{\Delta^2} \log(T).
    """
    if np.isclose(gap, 0): return 0
    condition = lambda length: length >= (8/gap**2) * np.log(horizon)
    length = 1
    while not condition(length):
        length += 1
    return length


# In[70]:


def check_condition_on_piecewise_stationary_problems(horizon, listOfMeans, changePoints):
    """ Check some conditions on the piecewise stationary problem."""
    M = len(listOfMeans)
    print("For a piecewise stationary problem with M = {} sequences...".format(M))  # DEBUG
    for m in range(M - 1):
        mus_m = listOfMeans[m]
        tau_m = changePoints[m]
        mus_mp1 = listOfMeans[m+1]
        tau_mp1 = changePoints[m+1]
        print("\nChecking m-th (m = {}) sequence, µ_m = {}, µ_m+1 = {} and tau_m = {} and tau_m+1 = {}".format(m, mus_m, mus_mp1, tau_m, tau_mp1))  # DEBUG
        for i, (mu_i_m, mu_i_mp1) in enumerate(zip(mus_m, mus_mp1)):
            gap = abs(mu_i_m - mu_i_mp1)
            length = tau_mp1 - tau_m
            lowerbound = lowerbound_on_sequence_length(horizon, gap)
            print("   - For arm i = {}, gap = {:.3g} and length = {} with lowerbound on length = {}...".format(i, gap, length, lowerbound))  # DEBUG
            if length < lowerbound:
                print("WARNING For arm i = {}, gap = {:.3g} and length = {} < lowerbound on length = {} !!".format(i, gap, length, lowerbound))  # DEBUG


# In[71]:


for envId, env in enumerate(configuration["environment"]):
    print("\n\n\nChecking environment number {}".format(envId))  # DEBUG
    listOfMeans = env["params"]["listOfMeans"]
    changePoints = env["params"]["changePoints"]
    check_condition_on_piecewise_stationary_problems(HORIZON, listOfMeans, changePoints)


# We checked that the two problems are not "easy enough" for our approach to be provably efficient.

# ---
# ## Creating the `Evaluator` object

# In[72]:


evaluation = Evaluator(configuration)


# In[73]:


def printAll(evaluation, envId):
    print("\nGiving the vector of final regrets ...")
    evaluation.printLastRegrets(envId)
    print("\nGiving the final ranks ...")
    evaluation.printFinalRanking(envId)
    print("\nGiving the mean and std running times ...")
    evaluation.printRunningTimes(envId)
    print("\nGiving the mean and std memory consumption ...")
    evaluation.printMemoryConsumption(envId)


# ##  Solving the problem
# Now we can simulate all the environments.
# 
# <span style="color:red">WARNING</span>
# That part takes some time, most stationary algorithms run with a time complexity linear in the horizon (ie., time takes $\mathcal{O}(T)$) and most piecewise stationary algorithms run with a time complexity **square** in the horizon (ie., time takes $\mathcal{O}(T^2)$).

# ### First problem

# In[74]:


get_ipython().run_cell_magic('time', '', 'envId = 0\nenv = evaluation.envs[envId]\n# Show the problem\nevaluation.plotHistoryOfMeans(envId)')


# In[75]:


get_ipython().run_cell_magic('time', '', '# Evaluate just that env\nevaluation.startOneEnv(envId, env)')


# In[76]:


_ = printAll(evaluation, envId)


# ### Second problem

# In[33]:


get_ipython().run_cell_magic('time', '', 'envId = 1\nenv = evaluation.envs[envId]\n# Show the problem\nevaluation.plotHistoryOfMeans(envId)')


# In[34]:


get_ipython().run_cell_magic('time', '', '# Evaluate just that env\nevaluation.startOneEnv(envId, env)')


# ## Plotting the results
# And finally, visualize them, with the plotting method of a `Evaluator` object:

# In[82]:


def plotAll(evaluation, envId, mainfig=None):
    savefig = mainfig
    if savefig is not None: savefig = "{}__LastRegrets__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting a boxplot of the final regrets ...")
    evaluation.plotLastRegrets(envId, boxplot=True, savefig=savefig)

    if savefig is not None: savefig = "{}__RunningTimes__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting the mean and std running times ...")
    evaluation.plotRunningTimes(envId, savefig=savefig)

    if savefig is not None: savefig = "{}__MemoryConsumption__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting the mean and std memory consumption ...")
    evaluation.plotMemoryConsumption(envId, savefig=savefig)

    if savefig is not None: savefig = "{}__Regrets__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting the mean regrets ...")
    evaluation.plotRegrets(envId, savefig=savefig)

    if savefig is not None: savefig = "{}__MeanReward__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting the mean rewards ...")
    evaluation.plotRegrets(envId, meanReward=True, savefig=savefig)

    if savefig is not None: savefig = "{}__LastRegrets__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting an histogram of the final regrets ...")
    evaluation.plotLastRegrets(envId, subplots=True, sharex=True, sharey=False, savefig=savefig)


# In[83]:


evaluation.nb_break_points


# ### First problem with change on only one arm (Local Restart should be better)
# 
# Let's first print the results then plot them:

# In[79]:


envId = 0


# In[80]:


_ = evaluation.plotHistoryOfMeans(envId, savefig="{}__HistoryOfMeans__env{}-{}".format(mainfig, envId+1, len(evaluation.envs)))


# In[81]:


_ = plotAll(evaluation, envId, mainfig=mainfig)


# ### Second problem with changes on all arms (Global restart should be better)
# 
# Let's first print the results then plot them:

# In[40]:


envId = 1


# In[41]:


_ = evaluation.plotHistoryOfMeans(envId, savefig="{}__HistoryOfMeans__env{}-{}".format(mainfig, envId+1, len(evaluation.envs)))


# In[42]:


_ = printAll(evaluation, envId)


# In[43]:


_ = plotAll(evaluation, envId, mainfig=mainfig)


# ---
# > That's it for this demo!
