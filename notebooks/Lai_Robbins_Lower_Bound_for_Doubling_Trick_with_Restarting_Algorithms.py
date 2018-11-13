
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Lai-&amp;-Robbins-lower-bound-for-stochastic-bandit-with-full-restart-points" data-toc-modified-id="Lai-&amp;-Robbins-lower-bound-for-stochastic-bandit-with-full-restart-points-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Lai &amp; Robbins lower-bound for stochastic bandit with full restart points</a></div><div class="lev2 toc-item"><a href="#Creating-the-problem" data-toc-modified-id="Creating-the-problem-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Creating the problem</a></div><div class="lev3 toc-item"><a href="#Parameters-for-the-simulation" data-toc-modified-id="Parameters-for-the-simulation-111"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Parameters for the simulation</a></div><div class="lev3 toc-item"><a href="#Some-MAB-problem-with-Bernoulli-arms" data-toc-modified-id="Some-MAB-problem-with-Bernoulli-arms-112"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>Some MAB problem with Bernoulli arms</a></div><div class="lev3 toc-item"><a href="#Some-RL-algorithms" data-toc-modified-id="Some-RL-algorithms-113"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>Some RL algorithms</a></div><div class="lev2 toc-item"><a href="#Creating-the-Evaluator-object" data-toc-modified-id="Creating-the-Evaluator-object-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Creating the <code>Evaluator</code> object</a></div><div class="lev2 toc-item"><a href="#Solving-the-problem" data-toc-modified-id="Solving-the-problem-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Solving the problem</a></div><div class="lev2 toc-item"><a href="#Plotting-the-results" data-toc-modified-id="Plotting-the-results-14"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Plotting the results</a></div><div class="lev2 toc-item"><a href="#Visualisation-the-lower-bound-for-algorithms-that-restart-at-breaking-points" data-toc-modified-id="Visualisation-the-lower-bound-for-algorithms-that-restart-at-breaking-points-15"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Visualisation the lower-bound for algorithms that restart at breaking points</a></div><div class="lev2 toc-item"><a href="#Seeing-the-lower-bound-on-the-regret-plot" data-toc-modified-id="Seeing-the-lower-bound-on-the-regret-plot-16"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Seeing the lower-bound on the regret plot</a></div><div class="lev2 toc-item"><a href="#Conclusion" data-toc-modified-id="Conclusion-17"><span class="toc-item-num">1.7&nbsp;&nbsp;</span>Conclusion</a></div>

# ---
# # Lai & Robbins lower-bound for stochastic bandit with full restart points
# 
# First, be sure to be in the main folder, or to have installed [`SMPyBandits`](https://github.com/SMPyBandits/SMPyBandits), and import `Evaluator` from `Environment` package:

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


get_ipython().system('pip install SMPyBandits watermark')
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -p SMPyBandits -a "Lilian Besson"')


# In[3]:


# Local imports
from SMPyBandits.Environment import Evaluator, tqdm
from SMPyBandits.Environment.plotsettings import legend, makemarkers


# We also need arms, for instance `Bernoulli`-distributed arm:

# In[4]:


# Import arms
from SMPyBandits.Arms import Bernoulli


# And finally we need some single-player Reinforcement Learning algorithms:

# In[7]:


# Import algorithms
from SMPyBandits.Policies import *


# In[8]:


import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12.4, 7)


# ---
# ## Creating the problem

# ### Parameters for the simulation
# - $T = 20000$ is the time horizon,
# - $N = 40$ is the number of repetitions,
# - `N_JOBS = 4` is the number of cores used to parallelize the code.

# In[9]:


HORIZON = 20000
REPETITIONS = 40
N_JOBS = 4


# ### Some MAB problem with Bernoulli arms
# We consider in this example $3$ problems, with `Bernoulli` arms, of different means.

# In[10]:


ENVIRONMENTS = [  # 1)  Bernoulli arms
        {   # A very easy problem, but it is used in a lot of articles
            "arm_type": Bernoulli,
            "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
    ]


# ### Some RL algorithms
# We compare some policies that use the [`DoublingTrickWrapper`](https://smpybandits.github.io/docs/Policies.DoublingTrickWrapper.html#module-Policies.DoublingTrickWrapper) policy, with a common growing scheme.

# In[11]:


NEXT_HORIZONS = [
    # next_horizon__arithmetic,
    next_horizon__geometric,
    # next_horizon__exponential,
    # next_horizon__exponential_slow,
    next_horizon__exponential_generic
]


# In[12]:


POLICIES = [
    # --- Doubling trick algorithm
    {
        "archtype": DoublingTrickWrapper,
        "params": {
            "next_horizon": next_horizon,
            "full_restart": full_restart,
            "policy": policy,
        }
    }
    for policy in [
        UCBH,
        MOSSH,
        klUCBPlusPlus,
        ApproximatedFHGittins,
    ]
    for full_restart in [
        True,
        # False,
    ]
    for next_horizon in NEXT_HORIZONS
]


# Complete configuration for the problem:

# In[13]:


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

# In[14]:


evaluation = Evaluator(configuration)


# ##  Solving the problem
# Now we can simulate all the $3$ environments. That part can take some time.

# In[15]:


for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
    # Evaluate just that env
    evaluation.startOneEnv(envId, env)


# ## Plotting the results
# And finally, visualize them, with the plotting method of a `Evaluator` object:

# In[16]:


def plotAll(evaluation, envId):
    evaluation.printFinalRanking(envId)
    fig = evaluation.plotRegrets(envId)
    # evaluation.plotRegrets(envId, semilogx=True)
    # evaluation.plotRegrets(envId, meanRegret=True)
    # evaluation.plotBestArmPulls(envId)
    return fig


# In[17]:


fig = plotAll(evaluation, 0)


# ## Visualisation the lower-bound for algorithms that restart at breaking points

# In[18]:


DEFAULT_FIRST_HORIZON = 100

def lower_bound_with_breakpoints(next_horizon, horizon, env,
                                 first_horizon=DEFAULT_FIRST_HORIZON,
                                 fig=None, marker=None):
    points, gap = breakpoints(next_horizon, first_horizon, horizon)
    X = np.arange(1, horizon)
    Y = np.log(X)
    # Durty estimate
    for estimate_horizon in points:
        if estimate_horizon <= horizon:
            before_breakpoint = np.max(np.where(X == estimate_horizon - 1)[0])
            lower_bound_before_breakpoint = Y[before_breakpoint]
            print("At time {}, lowerbound was {}".format(estimate_horizon, lower_bound_before_breakpoint))
            after = np.where(X >= estimate_horizon)
            Y[after] = np.log(X[after] - X[before_breakpoint]) + lower_bound_before_breakpoint
    if fig is None:  # new figure if needed
        fig, ax = plt.subplots()
        ax.set_xlabel("Time steps t=1..T, $T = {}$".format(horizon))
        ax.set_ylabel("Regret lower-bound")
        ax.set_title("Lai & Robbins lower-bound for problem with $K={}$ arms and $C_K={:.3g}$\nAnd doubling trick with restart points ({})".format(env.nbArms, env.lowerbound(), next_horizon.__latex_name__))
    else:
        ax = fig.axes[0]
        # https://stackoverflow.com/a/26845924/
        ax_legend = ax.legend()
        ax_legend.remove()
    complexity = env.lowerbound()
    ax.plot(X, complexity * Y,
            'k--' if marker is None else '{}k--'.format(marker),
            markevery=(0.0, 0.1),
            label="LB, DT restart ({})".format(next_horizon.__latex_name__))
    legend(fig=fig)
    fig.show()
    return fig


# In[19]:


_ = lower_bound_with_breakpoints(next_horizon__exponential_generic, HORIZON, evaluation.envs[0])


# ## Seeing the lower-bound on the regret plot

# In[20]:


fig = plotAll(evaluation, 0)


# In[21]:


markers = makemarkers(len(NEXT_HORIZONS))


# In[22]:


for i, next_horizon in enumerate(NEXT_HORIZONS):
    fig = lower_bound_with_breakpoints(next_horizon, HORIZON, evaluation.envs[0], fig=fig, marker=markers[i])


# In[23]:


fig


# ## Conclusion
# 
# That's it for today, folks!
