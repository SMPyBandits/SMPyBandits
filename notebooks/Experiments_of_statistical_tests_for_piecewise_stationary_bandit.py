
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Requirements-and-helper-functions" data-toc-modified-id="Requirements-and-helper-functions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Requirements and helper functions</a></div><div class="lev2 toc-item"><a href="#Requirements" data-toc-modified-id="Requirements-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Requirements</a></div><div class="lev2 toc-item"><a href="#Mathematical-notations-for-stationary-problems" data-toc-modified-id="Mathematical-notations-for-stationary-problems-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Mathematical notations for stationary problems</a></div><div class="lev2 toc-item"><a href="#Generating-fake-stationary-data" data-toc-modified-id="Generating-fake-stationary-data-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Generating fake stationary data</a></div><div class="lev2 toc-item"><a href="#Mathematical-notations-for-piecewise-stationary-problems" data-toc-modified-id="Mathematical-notations-for-piecewise-stationary-problems-14"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Mathematical notations for piecewise stationary problems</a></div><div class="lev2 toc-item"><a href="#Generating-fake-piecewise-stationary-data" data-toc-modified-id="Generating-fake-piecewise-stationary-data-15"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Generating fake piecewise stationary data</a></div><div class="lev1 toc-item"><a href="#Python-implementations-of-some-statistical-tests" data-toc-modified-id="Python-implementations-of-some-statistical-tests-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Python implementations of some statistical tests</a></div><div class="lev2 toc-item"><a href="#A-stupid-detection-test-(pure-random!)" data-toc-modified-id="A-stupid-detection-test-(pure-random!)-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>A stupid detection test (pure random!)</a></div><div class="lev2 toc-item"><a href="#Monitored" data-toc-modified-id="Monitored-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Monitored</a></div><div class="lev2 toc-item"><a href="#CUSUM" data-toc-modified-id="CUSUM-23"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>CUSUM</a></div><div class="lev2 toc-item"><a href="#PHT" data-toc-modified-id="PHT-24"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>PHT</a></div><div class="lev2 toc-item"><a href="#Gaussian-GLR" data-toc-modified-id="Gaussian-GLR-25"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Gaussian GLR</a></div><div class="lev2 toc-item"><a href="#Bernoulli-GLR" data-toc-modified-id="Bernoulli-GLR-26"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Bernoulli GLR</a></div><div class="lev2 toc-item"><a href="#List-of-all-Python-algorithms" data-toc-modified-id="List-of-all-Python-algorithms-27"><span class="toc-item-num">2.7&nbsp;&nbsp;</span>List of all Python algorithms</a></div><div class="lev1 toc-item"><a href="#Numba-implementations-of-some-statistical-tests" data-toc-modified-id="Numba-implementations-of-some-statistical-tests-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Numba implementations of some statistical tests</a></div><div class="lev1 toc-item"><a href="#Cython-implementations-of-some-statistical-tests" data-toc-modified-id="Cython-implementations-of-some-statistical-tests-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Cython implementations of some statistical tests</a></div><div class="lev1 toc-item"><a href="#Comparing-the-different-implementations" data-toc-modified-id="Comparing-the-different-implementations-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Comparing the different implementations</a></div><div class="lev2 toc-item"><a href="#Toy-data" data-toc-modified-id="Toy-data-51"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Toy data</a></div><div class="lev2 toc-item"><a href="#Checking-time-and-memory-efficiency?" data-toc-modified-id="Checking-time-and-memory-efficiency?-52"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Checking time and memory efficiency?</a></div><div class="lev2 toc-item"><a href="#Checking-detection-delay" data-toc-modified-id="Checking-detection-delay-53"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Checking detection delay</a></div><div class="lev2 toc-item"><a href="#Checking-false-alarm-probabilities" data-toc-modified-id="Checking-false-alarm-probabilities-54"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Checking false alarm probabilities</a></div><div class="lev2 toc-item"><a href="#Checking-missed-detection-probabilities" data-toc-modified-id="Checking-missed-detection-probabilities-55"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>Checking missed detection probabilities</a></div><div class="lev1 toc-item"><a href="#Conclusions" data-toc-modified-id="Conclusions-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Conclusions</a></div>

# # Requirements and helper functions

# ## Requirements
# 
# This notebook requires to have numpy and matplotlib installed.
# I'm also exploring usage of numba and cython later, so they are also needed.

# In[1]:


get_ipython().system('pip install watermark numpy scipy matplotlib numba cython tqdm')
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -p numpy,scipy,matplotlib,numba,cython,tqdm -a "Lilian Besson"')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import numba


# In[3]:


def in_notebook():
    """Check if the code is running inside a Jupyter notebook or not. Cf. http://stackoverflow.com/a/39662359/.

    >>> in_notebook()
    False
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole?
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal running IPython?
            return False
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


# In[4]:


if in_notebook():
    from tqdm import tqdm_notebook as tqdm
    print("Info: Using the Jupyter notebook version of the tqdm() decorator, tqdm_notebook() ...")  # DEBUG
else:
    from tqdm import tqdm


# ## Mathematical notations for stationary problems
# 
# We consider $K \geq 1$ arms, which are distributions $\nu_k$.
# We focus on Bernoulli distributions, which are characterized by their means, $\nu_k = \mathcal{B}(\mu_k)$ for $\mu_k\in[0,1]$.
# A stationary bandit problem is defined here by the vector $[\mu_1,\dots,\mu_K]$.
# 
# For a fixed problem and a *horizon* $T\in\mathbb{N}$, $T\geq1$, we draw samples from the $K$ distributions to get *data*: $\forall t, r_k(t) \sim \nu_k$, ie, $\mathbb{P}(r_k(t) = 1) = \mu_k$ and $r_k(t) \in \{0,1\}$.

# ## Generating fake stationary data
# 
# Here we give some examples of stationary problems and examples of data we can draw from them.

# In[5]:


def bernoulli_samples(means, horizon=1000):
    if np.size(means) == 1:
        return np.random.binomial(1, means, size=horizon)
    else:
        results = np.zeros((np.size(means), horizon))
        for i, mean in enumerate(means):
            results[i] = np.random.binomial(1, mean, size=horizon)
        return results


# In[6]:


problem1 = [0.5]

bernoulli_samples(problem1, horizon=20)


# For bandit problem with $K \geq 2$ arms, the *goal* is to design an online learning algorithm that roughly do the following:
# 
# - For time $t=1$ to $t=T$ (unknown horizon)
#     1. Algorithm $A$ decide to draw arm $A(t) \in\{1,\dots,K\}$,
#     2. Get the reward $r(t) = r_{A(t)}(t) \sim \nu_{A(t)}$ from the (Bernoulli) distribution of that arm,
#     3. Give this observation of reward $r(t)$ coming from arm $A(t)$ to the algorithm,
#     4. Update internal state of the algorithm
# 
# An algorithm is efficient if it obtains a high (expected) sum reward, ie, $\sum_{t=1}^T r(t)$.

# In[7]:


problem2 = [0.1, 0.5, 0.9]

bernoulli_samples(problem2, horizon=20)


# For instance on these data, the best arm is clearly the third one, with expected reward of $\mu^* = \max_k \mu_k = 0.9$.

# ## Mathematical notations for piecewise stationary problems
# 
# Now we fix the horizon $T\in\mathbb{N}$, $T\geq1$ and we also consider a set of $\Upsilon_T$ *break points*, $\tau_1,\dots,\tau_{\Upsilon_T} \in\{1,\dots,T\}$. We denote $\tau_0 = 0$ and $\tau_{\Upsilon_T+1} = T$ for convenience of notations.
# We can assume that breakpoints are far "enough" from each other, for instance that there exists an integer $N\in\mathbb{N},N\geq1$ such that $\min_{i=0}^{\Upsilon_T} \tau_{i+1} - \tau_i \geq N K$. That is, on each *stationary interval*, a uniform sampling of the $K$ arms gives at least $N$ samples by arm.
# 
# Now, in any stationary interval $[\tau_i + 1, \tau_{i+1}]$, the $K \geq 1$ arms are distributions $\nu_k^{(i)}$.
# We focus on Bernoulli distributions, which are characterized by their means, $\nu_k^{(i)} := \mathcal{B}(\mu_k^{(i)})$ for $\mu_k^{(i)}\in[0,1]$.
# A piecewise stationary bandit problem is defined here by the vector $[\mu_k^{(i)}]_{1\leq k \leq K, 1 \leq i \leq \Upsilon_T}$.
# 
# For a fixed problem and a *horizon* $T\in\mathbb{N}$, $T\geq1$, we draw samples from the $K$ distributions to get *data*: $\forall t, r_k(t) \sim \nu_k^{(i)}$ for $i$ the unique index of stationary interval such that $t\in[\tau_i + 1, \tau_{i+1}]$.

# ## Generating fake piecewise stationary data
# 
# The format to define piecewise stationary problem will be the following. It is compact but generic!
# 
# The first example considers a unique arm, with 2 breakpoints uniformly spaced.
# - On the first interval, for instance from $t=1$ to $t=500$, that is $\tau_1 = 500$, $\mu_1^{(1)} = 0.1$,
# - On the second interval, for instance from $t=501$ to $t=1000$, that is $\tau_2 = 100$, $\mu_1^{(2)} = 0.5$,
# - On the third interval, for instance from $t=1001$ to $t=1500$, that $\mu_1^{(3)} = 0.9$.

# In[8]:


# With 1 arm only!
problem_piecewise_0 = lambda horizon: {
    "listOfMeans": [
        [0.1],  # 0    to 499
        [0.5],  # 500  to 999
        [0.8],  # 1000  to 1499
    ],
    "changePoints": [
        int(0    * horizon / 1500.0),
        int(500  * horizon / 1500.0),
        int(1000  * horizon / 1500.0),
    ],
}


# In[9]:


# With 2 arms
problem_piecewise_1 = lambda horizon: {
    "listOfMeans": [
        [0.1, 0.2],  # 0    to 399
        [0.1, 0.3],  # 400  to 799
        [0.5, 0.3],  # 800  to 1199
        [0.4, 0.3],  # 1200 to 1599
        [0.3, 0.9],  # 1600 to end
    ],
    "changePoints": [
        int(0    * horizon / 2000.0),
        int(400  * horizon / 2000.0),
        int(800  * horizon / 2000.0),
        int(1200 * horizon / 2000.0),
        int(1600 * horizon / 2000.0),
    ],
}


# In[10]:


# With 3 arms
problem_piecewise_2 = lambda horizon: {
    "listOfMeans": [
        [0.2, 0.5, 0.9],  # 0    to 399
        [0.2, 0.2, 0.9],  # 400  to 799
        [0.2, 0.2, 0.1],  # 800  to 1199
        [0.7, 0.2, 0.1],  # 1200 to 1599
        [0.7, 0.5, 0.1],  # 1600 to end
    ],
    "changePoints": [
        int(0    * horizon / 2000.0),
        int(400  * horizon / 2000.0),
        int(800  * horizon / 2000.0),
        int(1200 * horizon / 2000.0),
        int(1600 * horizon / 2000.0),
    ],
}


# In[11]:


# With 3 arms
problem_piecewise_3 = lambda horizon: {
    "listOfMeans": [
        [0.4, 0.5, 0.9],  # 0    to 399
        [0.5, 0.4, 0.7],  # 400  to 799
        [0.6, 0.3, 0.5],  # 800  to 1199
        [0.7, 0.2, 0.3],  # 1200 to 1599
        [0.8, 0.1, 0.1],  # 1600 to end
    ],
    "changePoints": [
        int(0    * horizon / 2000.0),
        int(400  * horizon / 2000.0),
        int(800  * horizon / 2000.0),
        int(1200 * horizon / 2000.0),
        int(1600 * horizon / 2000.0),
    ],
}


# Now we can write a utility function that transform this compact representation into a full list of means.

# In[12]:


def getFullHistoryOfMeans(problem, horizon=2000):
    """Return the vector of mean of the arms, for a piece-wise stationary MAB.

    - It is a numpy array of shape (nbArms, horizon).
    """
    pb = problem(horizon)
    listOfMeans, changePoints = pb['listOfMeans'], pb['changePoints']
    nbArms = len(listOfMeans[0])
    if horizon is None:
        horizon = np.max(changePoints)
    meansOfArms = np.ones((nbArms, horizon))
    for armId in range(nbArms):
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(changePoints) - 1 and t >= changePoints[nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfArms[armId][t] = listOfMeans[nbChangePoint][armId]
    return meansOfArms


# For examples :

# In[13]:


getFullHistoryOfMeans(problem_piecewise_0, horizon=50)


# In[14]:


getFullHistoryOfMeans(problem_piecewise_1, horizon=50)


# In[15]:


getFullHistoryOfMeans(problem_piecewise_2, horizon=50)


# In[16]:


getFullHistoryOfMeans(problem_piecewise_3, horizon=50)


# And now we need to be able to generate samples from such distributions.

# In[17]:


def piecewise_bernoulli_samples(problem, horizon=1000):
    fullMeans = getFullHistoryOfMeans(problem, horizon=horizon)
    nbArms, horizon = np.shape(fullMeans)
    results = np.zeros((nbArms, horizon))
    for i in range(nbArms):
        mean_i = fullMeans[i, :]
        for t in range(horizon):
            mean_i_t = mean_i[t]
            results[i, t] = np.random.binomial(1, mean_i_t)
    return results


# Examples:

# In[18]:


getFullHistoryOfMeans(problem_piecewise_0, horizon=100)
piecewise_bernoulli_samples(problem_piecewise_0, horizon=100)


# We easily spot the (approximate) location of the breakpoint!
# 
# Another example:

# In[19]:


piecewise_bernoulli_samples(problem_piecewise_1, horizon=100)


# ----
# # Python implementations of some statistical tests
# 
# I will implement here the following statistical tests (and I give a link to the implementation of the correspond bandit policy in my framework [`SMPyBandits`](https://smpybandits.github.io/)
# 
# - Monitored (based on a McDiarmid inequality), for Monitored-UCB or [`M-UCB`](),
# - CUSUM, for [`CUSUM-UCB`](https://smpybandits.github.io/docs/Policies.CD_UCB.html?highlight=cusum#Policies.CD_UCB.CUSUM_IndexPolicy),
# - PHT, for [`PHT-UCB`](https://smpybandits.github.io/docs/Policies.CD_UCB.html?highlight=cusum#Policies.CD_UCB.PHT_IndexPolicy),
# - Gaussian GLR, for [`GaussianGLR-UCB`](https://smpybandits.github.io/docs/Policies.CD_UCB.html?highlight=glr#Policies.CD_UCB.GaussianGLR_IndexPolicy),
# - Bernoulli GLR, for [`BernoulliGLR-UCB`](https://smpybandits.github.io/docs/Policies.CD_UCB.html?highlight=glr#Policies.CD_UCB.BernoulliGLR_IndexPolicy).

# ## A stupid detection test (pure random!)
# Just to be sure that the test functions work as wanted, I start by writing a stupid change detection test, which is purely random!

# In[20]:


def PurelyRandom(all_data, t, proba=0.5):
    return np.random.random() < proba


# ## Monitored

# In[21]:


NB_ARMS = 1
WINDOW_SIZE = 80


# In[22]:


def Monitored(all_data, t,
              window_size=WINDOW_SIZE, threshold_b=None,
    ):
    r""" A change is detected for the current arm if the following test is true:

    .. math:: |\sum_{i=w/2+1}^{w} Y_i - \sum_{i=1}^{w/2} Y_i | > b ?

    - where :math:`Y_i` is the i-th data in the latest w data from this arm (ie, :math:`X_k(t)` for :math:`t = n_k - w + 1` to :math:`t = n_k` current number of samples from arm k).
    - where :attr:`threshold_b` is the threshold b of the test, and :attr:`window_size` is the window-size w.
    """
    data = all_data[:t]
    # don't try to detect change if there is not enough data!
    if len(data) < window_size:
        return False
    
    # compute parameters
    horizon = len(all_data)
    if threshold_b is None:
        threshold_b = np.sqrt(window_size/2 * np.log(2 * NB_ARMS * horizon**2))

    last_w_data = data[-window_size:]
    sum_first_half = np.sum(last_w_data[:window_size//2])
    sum_second_half = np.sum(last_w_data[window_size//2:])
    return abs(sum_first_half - sum_second_half) > threshold_b


# ## CUSUM

# In[23]:


#: Precision of the test.
EPSILON = 0.5

#: Default value of :math:`\lambda`.
LAMBDA = 1

#: Hypothesis on the speed of changes: between two change points, there is at least :math:`M * K` time steps, where K is the number of arms, and M is this constant.
MIN_NUMBER_OF_OBSERVATION_BETWEEN_CHANGE_POINT = 100

MAX_NB_RANDOM_EVENTS = 1


# In[24]:


from scipy.special import comb


# In[25]:


def compute_h_alpha__CUSUM(horizon, 
        verbose=False,
        M=MIN_NUMBER_OF_OBSERVATION_BETWEEN_CHANGE_POINT,
        max_nb_random_events=MAX_NB_RANDOM_EVENTS,
        nbArms=1,
        epsilon=EPSILON,
        lmbda=LAMBDA,
    ):
    r""" Compute the values :math:`C_1^+, C_1^-, C_1, C_2, h` from the formulas in Theorem 2 and Corollary 2 in the paper."""
    T = int(max(1, horizon))
    UpsilonT = int(max(1, max_nb_random_events))
    K = int(max(1, nbArms))
    # print("compute_h_alpha__CUSUM() with:\nT = {}, UpsilonT = {}, K = {}, epsilon = {}, lmbda = {}, M = {}".format(T, UpsilonT, K, epsilon, lmbda, M))  # DEBUG
    C2 = np.log(3) + 2 * np.exp(- 2 * epsilon**2 * M) / lmbda
    C1_minus = np.log(((4 * epsilon) / (1-epsilon)**2) * comb(M, int(np.floor(2 * epsilon * M))) * (2 * epsilon)**M + 1)
    C1_plus = np.log(((4 * epsilon) / (1+epsilon)**2) * comb(M, int(np.ceil(2 * epsilon * M))) * (2 * epsilon)**M + 1)
    C1 = min(C1_minus, C1_plus)
    if C1 == 0: C1 = 1  # FIXME
    h = 1/C1 * np.log(T / UpsilonT)
    alpha = K * np.sqrt((C2 * UpsilonT)/(C1 * T) * np.log(T / UpsilonT))
    alpha *= 0.01  # FIXME Just divide alpha to not have too large
    alpha = max(0, min(1, alpha))  # crop to [0, 1]
    # print("Gave C2 = {}, C1- = {} and C1+ = {} so C1 = {}, and h = {} and alpha = {}".format(C2, C1_minus, C1_plus, C1, h, alpha))  # DEBUG
    return h, alpha


# In[26]:


def CUSUM(all_data, t,
          epsilon=EPSILON,
          M=MIN_NUMBER_OF_OBSERVATION_BETWEEN_CHANGE_POINT,
          threshold_h=None,
    ):
    r""" Detect a change in the current arm, using the two-sided CUSUM algorithm [Page, 1954].

    - For each *data* k, compute:

    .. math::

        s_k^- &= (y_k - \hat{u}_0 - \varepsilon) 1(k > M),\\
        s_k^+ &= (\hat{u}_0 - y_k - \varepsilon) 1(k > M),\\
        g_k^+ &= max(0, g_{k-1}^+ + s_k^+),\\
        g_k^- &= max(0, g_{k-1}^- + s_k^-),\\

    - The change is detected if :math:`\max(g_k^+, g_k^-) > h`, where :attr:`threshold_h` is the threshold of the test,
    - And :math:`\hat{u}_0 = \frac{1}{M} \sum_{k=1}^{M} y_k` is the mean of the first M samples, where M is :attr:`M` the min number of observation between change points.
    """
    data = all_data[:t]
    
    # compute parameters
    horizon = len(all_data)
    if threshold_h is None:
        threshold_h, _ = compute_h_alpha__CUSUM(horizon, M, 1, epsilon=epsilon)

    gp, gm = 0, 0
    # First we use the first M samples to calculate the average :math:`\hat{u_0}`.
    u0hat = np.mean(data[:M])
    for k, y_k in enumerate(data):
        if k <= M:
            continue
        sp = u0hat - y_k - epsilon  # no need to multiply by (k > self.M)
        sm = y_k - u0hat - epsilon  # no need to multiply by (k > self.M)
        gp, gm = max(0, gp + sp), max(0, gm + sm)
        if max(gp, gm) >= threshold_h:
            return True
    return False


# ## PHT

# In[27]:


def PHT(all_data, t,
          epsilon=EPSILON,
          M=MIN_NUMBER_OF_OBSERVATION_BETWEEN_CHANGE_POINT,
          threshold_h=None,
    ):
    r""" Detect a change in the current arm, using the two-sided PHT algorithm [Hinkley, 1971].

    - For each *data* k, compute:

    .. math::

        s_k^- &= y_k - \hat{y}_k - \varepsilon,\\
        s_k^+ &= \hat{y}_k - y_k - \varepsilon,\\
        g_k^+ &= max(0, g_{k-1}^+ + s_k^+),\\
        g_k^- &= max(0, g_{k-1}^- + s_k^-),\\

    - The change is detected if :math:`\max(g_k^+, g_k^-) > h`, where :attr:`threshold_h` is the threshold of the test,
    - And :math:`\hat{y}_k = \frac{1}{k} \sum_{s=1}^{k} y_s` is the mean of the first k samples.
    """
    data = all_data[:t]
    
    # compute parameters
    horizon = len(all_data)
    if threshold_h is None:
        threshold_h, _ = compute_h_alpha__CUSUM(horizon, M, 1, epsilon=epsilon)

    gp, gm = 0, 0
    # First we use the first M samples to calculate the average :math:`\hat{u_0}`.
    for k, y_k in enumerate(data):
        y_k_hat = np.mean(data[:k])
        sp = y_k_hat - y_k - epsilon
        sm = y_k - y_k_hat - epsilon
        gp, gm = max(0, gp + sp), max(0, gm + sm)
        if max(gp, gm) >= threshold_h:
            return True
    return False


# ## Gaussian GLR

# In[28]:


def compute_c_alpha__GLR(t0, t, horizon, verbose=False, exponentBeta=1.05, alpha_t1=0.1):
    r""" Compute the values :math:`c, \alpha` from the corollary of of Theorem 2 from ["Sequential change-point detection: Laplace concentration of scan statistics and non-asymptotic delay bounds", O.-A. Maillard, 2018].

    .. note:: I am currently exploring the following variant (November 2018):

        - The probability of uniform exploration, :math:`\alpha`, is computed as a function of the current time:

        .. math:: \forall t>0, \alpha = \alpha_t := \alpha_{t=1} \frac{1}{\max(1, t^{\beta})}.

        - with :math:`\beta > 1, \beta` = ``exponentBeta`` (=1.05) and :math:`\alpha_{t=1} < 1, \alpha_{t=1}` = ``alpha_t1`` (=0.01).
    """
    T = int(max(1, horizon))
    delta = 1.0 / T
    if verbose: print("compute_c_alpha__GLR() with t = {}, t0 = {}, T = {}, delta = 1/T = {}".format(t, t0, T, delta))  # DEBUG
    t_m_t0 = abs(t - t0)
    c = (1 + (1 / (t_m_t0 + 1.0))) * 2 * np.log((2 * t_m_t0 * np.sqrt(t_m_t0 + 2)) / delta)
    if c < 0 and np.isinf(c): c = float('+inf')
    assert exponentBeta > 1.0, "Error: compute_c_alpha__GLR should have a exponentBeta > 1 but it was given = {}...".format(exponentBeta)  # DEBUG
    alpha = alpha_t1 / max(1, t)**exponentBeta
    if verbose: print("Gave c = {} and alpha = {}".format(c, alpha))  # DEBUG
    return c, alpha


# In[29]:


def klGauss(x, y, sig2x=0.25):
    r""" Kullback-Leibler divergence for Gaussian distributions of means ``x`` and ``y`` and variances ``sig2x`` and ``sig2y``, :math:`\nu_1 = \mathcal{N}(x, \sigma_x^2)` and :math:`\nu_2 = \mathcal{N}(y, \sigma_x^2)`:

    .. math:: \mathrm{KL}(\nu_1, \nu_2) = \frac{(x - y)^2}{2 \sigma_y^2} + \frac{1}{2}\left( \frac{\sigma_x^2}{\sigma_y^2} - 1 \log\left(\frac{\sigma_x^2}{\sigma_y^2}\right) \right).

    See https://en.wikipedia.org/wiki/Normal_distribution#Other_properties

    - sig2y = sig2x (same variance).
    """
    return (x - y) ** 2 / (2. * sig2x)


# In[30]:


def GaussianGLR(all_data, t,
          threshold_h=None,
    ):
    r""" Detect a change in the current arm, using the Generalized Likelihood Ratio test (GLR) and the :attr:`kl` function.

    - For each *time step* :math:`s` between :math:`t_0=0` and :math:`t`, compute:

    .. math::

        G^{\mathcal{N}_1}_{t_0:s:t} = (s-t_0+1)(t-s) \mathrm{kl}(\mu_{s+1,t}, \mu_{t_0,s}) / (t-t_0+1).

    - The change is detected if there is a time :math:`s` such that :math:`G^{\mathcal{N}_1}_{t_0:s:t} > h`, where :attr:`threshold_h` is the threshold of the test,
    - And :math:`\mu_{a,b} = \frac{1}{b-a+1} \sum_{s=a}^{b} y_s` is the mean of the samples between :math:`a` and :math:`b`.
    """
    data = all_data[:t]
    t0 = 0
    horizon = len(all_data)
    
    # compute parameters
    if threshold_h is None:
        threshold_h, _ = compute_c_alpha__GLR(0, t, horizon)

    mu = lambda a, b: np.mean(data[a : b+1])
    for s in range(t0, t - 1):
        this_kl = klGauss(mu(s+1, t), mu(t0, s))
        glr = ((s - t0 + 1) * (t - s) / (t - t0 + 1)) * this_kl
        if glr >= threshold_h:
            return True
    return False


# ## Bernoulli GLR

# In[31]:


eps = 1e-6  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]

def klBern(x, y):
    r""" Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution#Kullback.E2.80.93Leibler_divergence

    .. math:: \mathrm{KL}(\mathcal{B}(x), \mathcal{B}(y)) = x \log(\frac{x}{y}) + (1-x) \log(\frac{1-x}{1-y})."""
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


# In[32]:


def BernoulliGLR(all_data, t,
          threshold_h=None,
    ):
    r""" Detect a change in the current arm, using the Generalized Likelihood Ratio test (GLR) and the :attr:`kl` function.

    - For each *time step* :math:`s` between :math:`t_0=0` and :math:`t`, compute:

    .. math::

        G^{\mathcal{N}_1}_{t_0:s:t} = (s-t_0+1)(t-s) \mathrm{kl}(\mu_{s+1,t}, \mu_{t_0,s}) / (t-t_0+1).

    - The change is detected if there is a time :math:`s` such that :math:`G^{\mathcal{N}_1}_{t_0:s:t} > h`, where :attr:`threshold_h` is the threshold of the test,
    - And :math:`\mu_{a,b} = \frac{1}{b-a+1} \sum_{s=a}^{b} y_s` is the mean of the samples between :math:`a` and :math:`b`.
    """
    data = all_data[:t]
    t0 = 0
    horizon = len(all_data)
    
    # compute parameters
    if threshold_h is None:
        threshold_h, _ = compute_c_alpha__GLR(0, t, horizon)

    mu = lambda a, b: np.mean(data[a : b+1])
    for s in range(t0, t - 1):
        this_kl = klBern(mu(s+1, t), mu(t0, s))
        glr = ((s - t0 + 1) * (t - s) / (t - t0 + 1)) * this_kl
        if glr >= threshold_h:
            return True
    return False


# ## List of all Python algorithms

# In[33]:


all_CD_algorithms = [
    PurelyRandom,
    Monitored, CUSUM, PHT, GaussianGLR, BernoulliGLR
]


# ----
# # Numba implementations of some statistical tests
# 
# I should try to use the [`numba.jit`](https://numba.pydata.org/numba-doc/latest/reference/jit-compilation.html#numba.jit) decorator for all the functions defined above.

# In[34]:


import numba


# In[35]:


@numba.jit(nopython=True)
def klBern_numba(x, y):
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


# In[36]:


@numba.jit(nopython=True)
def klGauss_numba(x, y, sig2x=0.25):
    return (x - y) ** 2 / (2. * sig2x)


# <center><span style="font-size:xx-large; color:red;">TODO TODO TODO TODO TODO TODO</span></center>

# ----
# # Cython implementations of some statistical tests
# 
# I should try to use the [`%%cython`](https://cython.readthedocs.io/en/latest/src/quickstart/build.html#jupyter-notebook) magic for all the functions defined above.

# In[37]:


get_ipython().run_line_magic('load_ext', 'cython')


# In[38]:


get_ipython().run_cell_magic('cython', '', 'from libc.math cimport log\neps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]\n\ndef klBern_cython(float x, float y) -> float:\n    x = min(max(x, eps), 1 - eps)\n    y = min(max(y, eps), 1 - eps)\n    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))')


# In[39]:


get_ipython().run_cell_magic('cython', '', 'from libc.math cimport log\neps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]\n\ndef klGauss_cython(float x, float y, float sig2x=0.25) -> float:\n    return (x - y) ** 2 / (2. * sig2x)')


# <center><span style="font-size:xx-large; color:red;">TODO TODO TODO TODO TODO TODO</span></center>

# ----
# # Comparing the different implementations
# 
# I now want to compare, on a simple non stationary problem, the efficiency of the different change detection algorithms, in terms of:
# 
# - speed of computations (we should see that naive Python is much slower than Numba, which is also slower than the Cython version),
# - memory of algorithms? I guess we will draw the same observations,
# 
# But most importantly, in terms of:
# 
# - detection delay, as a function of the amplitude of the breakpoint, or number of prior data (from $t=1$ to $t=\tau$), or as a function of the parameter(s) of the algorithm,
# - probability of false detection, or missed detection.

# ## Toy data

# In[40]:


# With 1 arm only! With 1 change only!
toy_problem_piecewise = lambda firstMean, secondMean, tau: lambda horizon: {
    "listOfMeans": [
        [firstMean],  # 0    to 499
        [secondMean],  # 500  to 999
    ],
    "changePoints": [
        0,
        tau
    ],
}


# In[41]:


def get_toy_data(firstMean=0.5, secondMean=0.9, tau=None, horizon=100):
    if tau is None:
        tau = horizon // 2
    elif isinstance(tau, float):
        tau = int(tau * horizon)
    problem = toy_problem_piecewise(firstMean, secondMean, tau)
    data = piecewise_bernoulli_samples(problem, horizon=horizon)
    data = data.reshape(horizon)
    return data


# It is now very easy to get data and "see" manually on the data the location of the breakpoint:

# In[42]:


get_toy_data(firstMean=0.1, secondMean=0.9, tau=0.5, horizon=100)


# In[43]:


get_toy_data(firstMean=0.1, secondMean=0.9, tau=0.2, horizon=100)


# In[44]:


get_toy_data(firstMean=0.1, secondMean=0.4, tau=0.5, horizon=100)


# Of course, we want to check that detecting the change becomes harder when:
# 
# - the gap $\Delta = |\mu^{(2)} - \mu^{(1)}|$ decreases,
# - the number of samples before the change decreases ($\tau$ decreases),
# - the number of samples after the change decreases ($T - \tau$ decreases).

# ## Checking time and memory efficiency?

# In[45]:


import time


# In[46]:


def test_for_all_times(data, CDAlgorithm):
    horizon = len(data)
    # print(f"For test_for_all_times, horizon = {horizon} and algorithm {CDAlgorithm}")
    for t in range(0, horizon + 1):
        _ = CDAlgorithm(data, t)


# In[47]:


def check_timeEfficiency(firstMean=0.1, secondMean=0.4, tau=0.5, horizon=100, repetitions=50):
    if isinstance(tau, float):
        tau = int(tau * horizon)
    print(f"\nGenerating toy data for mu^1 = {firstMean}, mu^2 = {secondMean}, tau = {tau} and horizon = {horizon}...")
    times = np.zeros((repetitions, len(all_CD_algorithms)))
    for rep in tqdm(range(repetitions), desc="Repetitions"):
        data = get_toy_data(firstMean=firstMean, secondMean=secondMean, tau=tau, horizon=horizon)
        for i, CDAlgorithm in enumerate(all_CD_algorithms):
            startTime = time.time()
            _ = test_for_all_times(data, CDAlgorithm)
            endTime = time.time()
            times[rep, i] = endTime - startTime
    for i, CDAlgorithm in enumerate(all_CD_algorithms):
        mean_time = np.mean(times[:, i])
        print(f"- For algorithm {CDAlgorithm}, CPU time was {mean_time:.3g} seconds in average...")
    return times


# For examples:

# In[48]:


_ = check_timeEfficiency(firstMean=0.1, secondMean=0.4, tau=0.5, horizon=100)


# In[49]:


get_ipython().run_cell_magic('time', '', '_ = check_timeEfficiency(firstMean=0.1, secondMean=0.9, tau=0.5, horizon=1000)')


# ## Checking detection delay

# In[50]:


def detection_delay(data, tau, CDAlgorithm):
    horizon = len(data)
    # print(f"For detection_delay, horizon = {horizon}, tau = {tau} and algorithm {CDAlgorithm}")
    for t in range(tau, horizon + 1):
        if CDAlgorithm(data, t):
            # print(f"Algorithm {CDAlgorithm} detected the change point at time t = {t} after the change point, with delay = {t - tau}!")
            return t - tau
    return horizon - tau


# Now we can check the detection delay for our different algorithms.

# In[51]:


def check_detection_delay(firstMean=0.1, secondMean=0.4, tau=0.5, horizon=100, repetitions=50):
    if isinstance(tau, float):
        tau = int(tau * horizon)
    print(f"\nGenerating toy data for mu^1 = {firstMean}, mu^2 = {secondMean}, tau = {tau} and horizon = {horizon}...")
    delays = np.zeros((repetitions, len(all_CD_algorithms)))
    for rep in tqdm(range(repetitions), desc="Repetitions"):
        data = get_toy_data(firstMean=firstMean, secondMean=secondMean, tau=tau, horizon=horizon)
        for i, CDAlgorithm in enumerate(all_CD_algorithms):
            delay = detection_delay(data, tau, CDAlgorithm)
            delays[rep, i] = delay
    for i, CDAlgorithm in enumerate(all_CD_algorithms):
        mean_delay = np.mean(delays[:, i])
        print(f"- For algorithm {CDAlgorithm}, detection delay was {mean_delay:.3g} steps in average...")
    return delays


# For examples:

# In[52]:


_ = check_detection_delay(firstMean=0.1, secondMean=0.4, tau=0.5, horizon=100)


# In[53]:


get_ipython().run_cell_magic('time', '', '_ = check_detection_delay(firstMean=0.1, secondMean=0.9, tau=0.5, horizon=1000)')


# ## Checking false alarm probabilities

# In[54]:


def false_alarm(data, tau, CDAlgorithm):
    horizon = len(data)
    # print(f"For false_alarm, horizon = {horizon}, tau = {tau} and algorithm {CDAlgorithm}")
    for t in range(0, tau):
        if CDAlgorithm(data, t):
            # print(f"Algorithm {CDAlgorithm} detected the change point at time t = {t} BEFORE the change point!")
            return True
    return False


# Now we can check the false alarm probabilities for our different algorithms.

# In[55]:


def check_false_alarm(firstMean=0.1, secondMean=0.4, tau=0.5, horizon=100, repetitions=50):
    if isinstance(tau, float):
        tau = int(tau * horizon)
    print(f"\nGenerating toy data for mu^1 = {firstMean}, mu^2 = {secondMean}, tau = {tau} and horizon = {horizon}...")
    alarms = np.zeros((repetitions, len(all_CD_algorithms)))
    for rep in tqdm(range(repetitions), desc="Repetitions"):
        data = get_toy_data(firstMean=firstMean, secondMean=secondMean, tau=tau, horizon=horizon)
        for i, CDAlgorithm in enumerate(all_CD_algorithms):
            alarm = false_alarm(data, tau, CDAlgorithm)
            alarms[rep, i] = alarm
    for i, CDAlgorithm in enumerate(all_CD_algorithms):
        mean_alarm = np.sum(alarms[:, i]) / float(repetitions)
        print(f"- For algorithm {CDAlgorithm}, a false alarm happened {mean_alarm:.3g} times in average...")
    return alarms


# For examples:

# In[56]:


_ = check_false_alarm(firstMean=0.1, secondMean=0.4, tau=0.5, horizon=100)


# In[57]:


get_ipython().run_cell_magic('time', '', '_ = check_false_alarm(firstMean=0.1, secondMean=0.9, tau=0.5, horizon=1000)')


# ## Checking missed detection probabilities

# In[58]:


def missed_detection(data, tau, CDAlgorithm):
    horizon = len(data)
    # print(f"For missed_detection, horizon = {horizon}, tau = {tau} and algorithm {CDAlgorithm}")
    for t in range(tau, horizon + 1):
        if CDAlgorithm(data, t):
            return False
    return True


# Now we can check the false alarm probabilities for our different algorithms.

# In[59]:


def check_missed_detection(firstMean=0.1, secondMean=0.4, tau=0.5, horizon=100, repetitions=50):
    if isinstance(tau, float):
        tau = int(tau * horizon)
    print(f"\nGenerating toy data for mu^1 = {firstMean}, mu^2 = {secondMean}, tau = {tau} and horizon = {horizon}...")
    misses = np.zeros((repetitions, len(all_CD_algorithms)))
    for rep in tqdm(range(repetitions), desc="Repetitions"):
        data = get_toy_data(firstMean=firstMean, secondMean=secondMean, tau=tau, horizon=horizon)
        for i, CDAlgorithm in enumerate(all_CD_algorithms):
            miss = missed_detection(data, tau, CDAlgorithm)
            misses[rep, i] = miss
    for i, CDAlgorithm in enumerate(all_CD_algorithms):
        mean_miss = np.sum(misses[:, i]) / float(repetitions)
        print(f"- For algorithm {CDAlgorithm}, a missed detection happened {mean_miss:.3g} times in average...")
    return misses


# For examples:

# In[60]:


_ = check_missed_detection(firstMean=0.1, secondMean=0.4, tau=0.5, horizon=100)


# In[61]:


get_ipython().run_cell_magic('time', '', '_ = check_missed_detection(firstMean=0.1, secondMean=0.9, tau=0.5, horizon=1000)')


# ----
# # Conclusions

# <center><span style="font-size:xx-large; color:red;">TODO TODO TODO TODO TODO TODO</span></center>
