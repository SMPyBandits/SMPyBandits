
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Trying-to-use-Unsupervised-Learning-algorithms-for-a-Gaussian-bandit-problem" data-toc-modified-id="Trying-to-use-Unsupervised-Learning-algorithms-for-a-Gaussian-bandit-problem-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Trying to use Unsupervised Learning algorithms for a Gaussian bandit problem</a></div><div class="lev2 toc-item"><a href="#Creating-the-Gaussian-bandit-problem" data-toc-modified-id="Creating-the-Gaussian-bandit-problem-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Creating the Gaussian bandit problem</a></div><div class="lev2 toc-item"><a href="#Getting-data-from-a-first-phase-of-uniform-exploration" data-toc-modified-id="Getting-data-from-a-first-phase-of-uniform-exploration-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Getting data from a first phase of uniform exploration</a></div><div class="lev2 toc-item"><a href="#Fitting-an-Unsupervised-Learning-algorithm" data-toc-modified-id="Fitting-an-Unsupervised-Learning-algorithm-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Fitting an Unsupervised Learning algorithm</a></div><div class="lev2 toc-item"><a href="#Using-the-prediction-to-decide-the-next-arm-to-sample" data-toc-modified-id="Using-the-prediction-to-decide-the-next-arm-to-sample-14"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Using the prediction to decide the next arm to sample</a></div><div class="lev2 toc-item"><a href="#Manual-implementation-of-basic-Gaussian-kernel-fitting" data-toc-modified-id="Manual-implementation-of-basic-Gaussian-kernel-fitting-15"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Manual implementation of basic Gaussian kernel fitting</a></div><div class="lev2 toc-item"><a href="#Implementing-a-Policy-from-that-idea" data-toc-modified-id="Implementing-a-Policy-from-that-idea-16"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Implementing a Policy from that idea</a></div><div class="lev3 toc-item"><a href="#Basic-algorithm" data-toc-modified-id="Basic-algorithm-161"><span class="toc-item-num">1.6.1&nbsp;&nbsp;</span>Basic algorithm</a></div><div class="lev3 toc-item"><a href="#A-variant,-by-aggregating-samples" data-toc-modified-id="A-variant,-by-aggregating-samples-162"><span class="toc-item-num">1.6.2&nbsp;&nbsp;</span>A variant, by aggregating samples</a></div><div class="lev3 toc-item"><a href="#Implementation" data-toc-modified-id="Implementation-163"><span class="toc-item-num">1.6.3&nbsp;&nbsp;</span>Implementation</a></div><div class="lev2 toc-item"><a href="#Comparing-its-performance-on-this-Gaussian-problem" data-toc-modified-id="Comparing-its-performance-on-this-Gaussian-problem-17"><span class="toc-item-num">1.7&nbsp;&nbsp;</span>Comparing its performance on this Gaussian problem</a></div><div class="lev3 toc-item"><a href="#Configuring-an-experiment" data-toc-modified-id="Configuring-an-experiment-171"><span class="toc-item-num">1.7.1&nbsp;&nbsp;</span>Configuring an experiment</a></div><div class="lev3 toc-item"><a href="#Running-an-experiment" data-toc-modified-id="Running-an-experiment-172"><span class="toc-item-num">1.7.2&nbsp;&nbsp;</span>Running an experiment</a></div><div class="lev3 toc-item"><a href="#Visualizing-the-results" data-toc-modified-id="Visualizing-the-results-173"><span class="toc-item-num">1.7.3&nbsp;&nbsp;</span>Visualizing the results</a></div><div class="lev2 toc-item"><a href="#Another-experiment,-with-just-more-Gaussian-arms" data-toc-modified-id="Another-experiment,-with-just-more-Gaussian-arms-18"><span class="toc-item-num">1.8&nbsp;&nbsp;</span>Another experiment, with just more Gaussian arms</a></div><div class="lev3 toc-item"><a href="#Running-the-experiment" data-toc-modified-id="Running-the-experiment-181"><span class="toc-item-num">1.8.1&nbsp;&nbsp;</span>Running the experiment</a></div><div class="lev3 toc-item"><a href="#Visualizing-the-results" data-toc-modified-id="Visualizing-the-results-182"><span class="toc-item-num">1.8.2&nbsp;&nbsp;</span>Visualizing the results</a></div><div class="lev3 toc-item"><a href="#Very-good-performance!" data-toc-modified-id="Very-good-performance!-183"><span class="toc-item-num">1.8.3&nbsp;&nbsp;</span>Very good performance!</a></div><div class="lev2 toc-item"><a href="#Another-experiment,-with-Bernoulli-arms" data-toc-modified-id="Another-experiment,-with-Bernoulli-arms-19"><span class="toc-item-num">1.9&nbsp;&nbsp;</span>Another experiment, with Bernoulli arms</a></div><div class="lev3 toc-item"><a href="#Running-the-experiment" data-toc-modified-id="Running-the-experiment-191"><span class="toc-item-num">1.9.1&nbsp;&nbsp;</span>Running the experiment</a></div><div class="lev3 toc-item"><a href="#Visualizing-the-results" data-toc-modified-id="Visualizing-the-results-192"><span class="toc-item-num">1.9.2&nbsp;&nbsp;</span>Visualizing the results</a></div><div class="lev2 toc-item"><a href="#Conclusion" data-toc-modified-id="Conclusion-110"><span class="toc-item-num">1.10&nbsp;&nbsp;</span>Conclusion</a></div><div class="lev3 toc-item"><a href="#Non-logarithmic-regret" data-toc-modified-id="Non-logarithmic-regret-1101"><span class="toc-item-num">1.10.1&nbsp;&nbsp;</span>Non-logarithmic regret</a></div><div class="lev3 toc-item"><a href="#Comparing-time-complexity" data-toc-modified-id="Comparing-time-complexity-1102"><span class="toc-item-num">1.10.2&nbsp;&nbsp;</span>Comparing <em>time complexity</em></a></div><div class="lev3 toc-item"><a href="#Not-so-efficient-for-Bernoulli-arms" data-toc-modified-id="Not-so-efficient-for-Bernoulli-arms-1103"><span class="toc-item-num">1.10.3&nbsp;&nbsp;</span>Not so efficient for Bernoulli arms</a></div>

# ----
# # Trying to use Unsupervised Learning algorithms for a Gaussian bandit problem
# 
# This small [Jupyter notebook](https://www.jupyter.org/) presents an experiment, in the context of [Multi-Armed Bandit problems](https://en.wikipedia.org/wiki/Multi-armed_bandit) (MAB).
# 
# [I am](http://perso.crans.org/besson/) trying to answer a simple question:
# 
# > "Can we use generic unsupervised learning algorithm, like [Kernel Density estimation](http://scikit-learn.org/stable/modules/density.html) or [Ridge Regression](http://scikit-learn.org/stable/modules/linear_model.html), instead of MAB algorithms like [UCB](http://sbubeck.com/SurveyBCB12.pdf) or [Thompson Sampling](https://en.wikipedia.org/wiki/Thompson_sampling) ?
# 
# I will use my [SMPyBandits](https://smpybandits.github.io/) library, for which a complete documentation is available, [here at https://smpybandits.github.io/](https://smpybandits.github.io/), and the [scikit-learn package](http://scikit-learn.org/).

# ## Creating the Gaussian bandit problem
# First, be sure to be in the main folder, or to have installed [`SMPyBandits`](https://github.com/SMPyBandits/SMPyBandits), and import the [`MAB` class](https://smpybandits.github.io/docs/Environment.MAB.html#Environment.MAB.MAB) from [the `Environment` package](https://smpybandits.github.io/docs/Environment.html#module-Environment):

# In[2]:


import numpy as np


# In[3]:


get_ipython().system('pip install SMPyBandits watermark')
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -p SMPyBandits -a "Lilian Besson"')


# In[4]:


from SMPyBandits.Environment import MAB


# And also, import the [`Gaussian` class](https://smpybandits.github.io/docs/Arms.Gaussian.html#Arms.Gaussian.Gaussian) to create Gaussian-distributed arms.

# In[5]:


from SMPyBandits.Arms import Gaussian


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


get_ipython().run_line_magic('pinfo', 'Gaussian')


# Let create a simple bandit problem, with 3 arms, and visualize an histogram showing the repartition of rewards.

# In[8]:


means = [0.45, 0.5, 0.55]
M = MAB(Gaussian(mu, sigma=0.2) for mu in means)


# In[10]:


_ = M.plotHistogram(horizon=1000000)


# > As we can see, the rewards of the different arms are close. It won't be easy to distinguish them.

# Then we can generate some draws, from all arms, from time $t=1$ to $t=T_0$, for let say $T_0 = 1000$ :

# In[11]:


T_0 = 1000
shape = (T_0,)
draws = np.array([ b.draw_nparray(shape) for b in M.arms ])


# In[12]:


draws


# The *empirical means* of each arm can be estimated, quite easily, and could be used to make all the decisions from $t \geq T_0 + 1$.

# In[13]:


empirical_means = np.mean(draws, axis=1)


# In[14]:


empirical_means


# Clearly, the last arm is the best. And the empirical means $\hat{\mu}_k(t)$ for $k=1,\dots,K$, are really close to the true one, as $T_0 = 1000$ is quite large.

# In[15]:


def relative_error(x, y):
    return np.abs(x - y) / x


# In[16]:


relative_error(means, empirical_means)


# That's less than $3\%$ error, it's already quite good!
# 
# *Conclusion* : If we have "enough" samples, and the distribution are not too close, there is no need to do any learning: just pick the arm with highest mean, from now on, and you will be good!

# In[17]:


best_arm_estimated = np.argmax(empirical_means)
best_arm = np.argmax(means)
assert best_arm_estimated == best_arm, "Error: the best arm is wrongly estimated, even after {} samples.".format(T_0)


# ----
# ## Getting data from a first phase of uniform exploration
# 
# But maybe $T_0 = 1000$ was really too large...
# 
# Let assume that the initial data was obtained from an algorithm which starts playing by exploring every arm, uniformly at random, until it gets "enough" data.
# 
# - On the one hand, if we ask him to sample each arm $1000$ times, of course the empirical mean $\hat{\mu_k(t)}$ will correctly estimate the true mean $\mu_k$ (if the gap $\Delta = \min_{i \neq j} |\mu_i - \mu_j|$ is not too small).
# - But on the other hand, starting with a long phase of uniform exploration will increase dramatically the regret.
# 
# What if we want to use the same technique on very few data?
# Let see with $T_0 = 10$, if the empirical means are still as close to the true ones.

# In[18]:


np.random.seed(10000)  # for reproducibility of the error best_arm_estimated = 1
T_0 = 10
draws = np.array([ b.draw_nparray((T_0, )) for b in M.arms ])
empirical_means = np.mean(draws, axis=1)
empirical_means


# In[20]:


relative_error(means, empirical_means)
best_arm_estimated = np.argmax(empirical_means)
best_arm_estimated
assert best_arm_estimated == best_arm, "Error: the best arm is wrongly estimated, even after {} samples.".format(T_0)


# Clearly, if there is not enough sample, the [*empirical mean* estimator](https://smpybandits.github.io/docs/Policies.EmpiricalMeans.html#Policies.EmpiricalMeans.EmpiricalMeans) *can* be wrong.
# It will not always be wrong with so few samples, but it can.

# ----
# ## Fitting an Unsupervised Learning algorithm
# 
# We should use the initial data for more than just getting empirical means.
# 
# Let use a simple *Unsupervised Learning* algorithm, implemented in the [scikit-learn (`sklearn`)](http://scikit-learn.org/) package: [1D Kernel Density estimation](http://scikit-learn.org/stable/modules/density.html).

# In[18]:


from sklearn.neighbors.kde import KernelDensity


# First, we need to create a model.
# 
# Here we assume to know that the arms are Gaussian, so fitting a Gaussian kernel will probably work the best.
# The `bandwidth` parameter should be of the order of the variances $\sigma_k$ of each arm (we used $0.2$).

# In[19]:


kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
kde


# Then, we will feed it the initial data, obtained from the initial phase of uniform exploration, from $t = 1, \dots, T_0$.

# In[20]:


draws
draws.shape


# We need to use the transpose of this array, as the data should have shape `(n_samples, n_features)`, i.e., of shape `(10, 3)` here.

# In[21]:


get_ipython().run_line_magic('pinfo', 'kde.fit')


# In[22]:


kde.fit(draws.T)


# The [`score_samples(X)`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity.score_samples) method can be used to evaluate the density on sample data (i.e., the likelihood of each observation).

# In[23]:


kde.score_samples(draws.T)


# For instance, based on the means $[0.45, 0.5, 0.55]$, the sample $[10, -10, 0]$ should be *very* unlikely, while $[0.4, 0.5, 0.6]$ will be *more* likely.
# And the vector of empirical means is a very likely observation as well.

# In[24]:


kde.score(np.array([10, -10, 0]).reshape(1, -1))
kde.score(np.array([0.4, 0.5, 0.6]).reshape(1, -1))
kde.score(empirical_means.reshape(1, -1))


# ----
# ## Using the prediction to decide the next arm to sample
# 
# Now that we have a model of [Kernel Density](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html) estimation, we can use it to *generate some random samples*.

# In[25]:


get_ipython().run_line_magic('pinfo', 'kde.sample')


# Basically, that means we can use this model to predict what the next output of the 3 arms (constituting the Gaussian problem) will be.
# 
# Let see this with one example.

# In[26]:


np.random.seed(1)
one_sample = kde.sample()
one_sample


# In[27]:


one_draw = M.draw_each()
one_draw


# Of course, the next random rewards from the arms have no reason to be close to predicted ones...
# 
# But maybe we can use the prediction to choose the arm with highest sample?
# And hopefully this will be the best arm, *at least in average*!

# In[28]:


best_arm_sampled = np.argmax(one_sample)
best_arm_sampled
assert best_arm_sampled == best_arm, "Error: the best arm is wrongly estimated from a random sample, even after {} observations.".format(T_0)


# ## Manual implementation of basic Gaussian kernel fitting
# 
# We can also implement manually a simple 1D Unsupervised Learning algorithm, which fits a Gaussian kernel (i.e., a distribution $\mathcal{N}(\mu,\sigma)$) on the 1D data, for each arm.
# 
# Let start with a base class, showing the signature any Unsupervised Learning should have to be used in our policy (defined below).

# In[47]:


# --- Unsupervised fitting models

class FittingModel(object):
    """ Base class for any fitting model"""

    def __init__(self, *args, **kwargs):
        """ Nothing to do here."""
        pass

    def __repr__(self):
        return str(self)

    def fit(self, data):
        """ Nothing to do here."""
        return self

    def sample(self, shape=1):
        """ Always 0., for instance."""
        return 0.

    def score_samples(self, data):
        """ Always 1., for instance."""
        return 1.

    def score(self, data):
        """ Log likelihood of the point (or the vector of data), under the current Gaussian model."""
        return np.log(np.sum(self.score_samples(data)))


# And then, the `SimpleGaussianKernel` class, using [`scipy.stats.norm.pdf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html) to evaluate the log-probability of an observation.

# In[48]:


import scipy.stats as st


class SimpleGaussianKernel(FittingModel):
    """ Basic Unsupervised Learning algorithm, which simply fits a 1D Gaussian on some 1D data."""

    def __init__(self, loc=0., scale=1., *args, **kwargs):
        r""" Starts with :math:`\mathcal{N}(0, 1)`, by default."""
        self.loc = float(loc)
        self.scale = float(scale)

    def __str__(self):
        return "N({:.3g}, {:.3g})".format(self.loc, self.scale)

    def fit(self, data):
        """ Use the mean and variance from the 1D vector data (of shape `n_samples` or `(n_samples, 1)`)."""
        self.loc, self.scale = np.mean(data), np.std(data)
        return self

    def sample(self, shape=1):
        """ Return one or more sample, from the current Gaussian model."""
        if shape == 1:
            return np.random.normal(self.loc, self.scale)
        else:
            return np.random.normal(self.loc, self.scale, shape)

    def score_samples(self, data):
        """ Likelihood of the point (or the vector of data), under the current Gaussian model, component-wise."""
        return st.norm.pdf(data, loc=self.loc, scale=np.sqrt(self.scale))


# ----
# ## Implementing a Policy from that idea
# 
# Based on that idea, we can implement a policy, following the [common API of all the policies](https://smpybandits.github.io/docs/Policies.BasePolicy.html#Policies.BasePolicy.BasePolicy) of my framework.

# ### Basic algorithm
# - Initially : create $K$ Unsupervised Learning algorithms $\mathcal{U}_k(0)$, $k\in\{1,\dots,K\}$, for instance `KernelDensity` estimators.
# - For the first $K \times T_0$ time steps, each arm $k \in \{1, \dots, K\}$ is sampled exactly $T_0$ times, to get a lot of initial observations for each arm.
# - With these first $T_0$ (e.g., $50$) observations, train a first version of the Unsupervised Learning algorithms $\mathcal{U}_k(t)$, $k\in\{1,\dots,K\}$.
# - Then, for the following time steps, $t \geq T_0 + 1$ :
#     + Once in a while (every $T_1 =$ `fit_every` steps, e.g., $100$), retrain all the Unsupervised Learning algorithms :
#         * For each arm $k\in\{1,\dots,K\}$, use all the previous observations of that arm
#           to train the model $\mathcal{U}_k(t)$.
#     + Otherwise, use the previously trained model :
#         * Get a random sample, $s_k(t)$ from the $K$ Unsupervised Learning algorithms $\mathcal{U}_k(t)$, $k\in\{1,\dots,K\}$ :
#           $$ \forall k\in\{1,\dots,K\}, \;\; s_k(t) \sim \mathcal{U}_k(t). $$
#         * Chose the arm $A(t)$ with *highest* sample :
#           $$ A(t) \in \arg\max_{k\in\{1,\dots,K\}} s_k(t). $$
#         * Play that arm $A(t)$, receive a reward $r_{A(t)}(t)$ from its (unknown) distribution, and store it.

# ### A variant, by aggregating samples
# A more robust (and so more correct) variant could be to use a bunch of samples, and use their mean to give $s_k(t)$ :
# 
# * Get a bunch of $M$ random samples (e.g., $50$), $s_k^i(t)$ from the $K$ Unsupervised Learning algorithms $\mathcal{U}_k(t)$, $k\in\{1,\dots,K\}$ :
#   $$ \forall k\in\{1,\dots,K\}, \;\; \forall i\in\{1,\dots,M\}, \;\; s_k^i(t) \sim \mathcal{U}_k(t). $$
# * Average them to get $\hat{s_k}(t)$ :
#   $$ \forall k\in\{1,\dots,K\}, \;\; \hat{s_k}(t) := \frac{1}{M} \sum_{i=1}^{M} s_k^i(t). $$
# * Chose the arm $A(t)$ with *highest* mean sample :
#   $$ A(t) \in \arg\max_{k\in\{1,\dots,K\}} \hat{s_k}(t). $$

# ### Implementation
# In code, this gives the following:

# In[30]:


class UnsupervisedLearning(object):
    """ Generic policy using an Unsupervised Learning algorithm, from scikit-learn.
    
    - Warning: still highly experimental!
    """
    
    def __init__(self, nbArms, estimator=KernelDensity,
                 T_0=10, fit_every=100, meanOf=50,
                 lower=0., amplitude=1.,  # not used, but needed for my framework
                 *args, **kwargs):
        self.nbArms = nbArms
        self.t = -1
        T_0 = int(T_0)
        self.T_0 = int(max(1, T_0))
        self.fit_every = int(fit_every)
        self.meanOf = int(meanOf)
        # Unsupervised Learning algorithm
        self._was_fitted = False
        self._estimator = estimator
        self._args = args
        self._kwargs = kwargs
        self.ests = [ self._estimator(*self._args, **self._kwargs) for _ in range(nbArms) ]
        # Store all the observations
        self.observations = [ [] for _ in range(nbArms) ]
    
    def __str__(self):
        return "UnsupervisedLearning({.__name__}, $T_0={:.3g}$, $T_1={:.3g}$, $M={:.3g}$)".format(self._estimator, self.T_0, self.fit_every, self.meanOf)
            
    def startGame(self):
        """ Reinitialize everything."""
        self.t = -1
        self.ests = [ self._estimator(*self._args, **self._kwargs) for _ in range(self.nbArms) ]
    
    def getReward(self, armId, reward):
        """ Store this observation."""
        # print("   - At time {}, we saw {} from arm {} ...".format(self.t, reward, armId))  # DEBUG
        self.observations[armId].append(reward)
    
    def choice(self):
        """ Choose an arm."""
        self.t += 1
        # Start by sampling each arm a certain number of times
        if self.t < self.nbArms * self.T_0:
            # print("- First phase: exploring arm {} at time {} ...".format(self.t % self.nbArms, self.t))  # DEBUG
            return self.t % self.nbArms
        else:
            # print("- Second phase: at time {} ...".format(self.t))  # DEBUG
            # 1. Fit the Unsupervised Learning on *all* the data observed so far, but do it once in a while only
            if not self._was_fitted:
                # print("   - Need to first fit the model of each arm with the first {} observations, now of shape {} ...".format(self.fit_every, np.shape(self.observations)))  # DEBUG
                self.fit(self.observations)
                self._was_fitted = True
            elif self.t % self.fit_every == 0:
                # print("   - Need to refit the model of each arm with {} more observations, now of shape {} ...".format(self.fit_every, np.shape(self.observations)))  # DEBUG
                self.fit(self.observations)
            # 2. Sample a random prediction for next output of the arms
            prediction = self.sample_with_mean()
            # 3. Use this sample to select next arm to play
            best_arm_predicted = np.argmax(prediction)
            # print("   - So the best arm seems to be = {} ...".format(best_arm_predicted))  # DEBUG
            return best_arm_predicted
    
    # --- Shortcut methods
    
    def fit(self, data):
        """ Fit each of the K models, with the data accumulated up-to now."""
        for armId in range(self.nbArms):
            # print(" - Fitting the #{} model, with observations of shape {} ...".format(armId + 1, np.shape(self.observations[armId])))  # DEBUG
            est = self.ests[armId]
            est.fit(np.asarray(data[armId]).reshape(-1, 1))
            self.ests[armId] = est
    
    def sample(self):
        """ Return a vector of random sample from each of the K models."""
        return [ float(est.sample()) for est in self.ests ]
    
    def sample_with_mean(self, meanOf=None):
        """ Return a vector of random sample from each of the K models, by averaging a lot of samples (reduce variance)."""
        if meanOf is None:
            meanOf = self.meanOf
        return [ float(np.mean(est.sample(meanOf))) for est in self.ests ]
    
    def score(self, obs):
        """ Return a vector of scores, for each of the K models on its observation."""
        return [ float(est.score(o)) for est, o in zip(self.ests, obs) ]

    def estimatedOrder(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means."""
        return np.argsort(self.sample_with_mean())


# In[31]:


get_ipython().run_line_magic('pinfo', 'UnsupervisedLearning')


# For example, we can chose these values for the numerical parameters :

# In[32]:


nbArms = M.nbArms
T_0 = 100
fit_every = 1000
meanOf = 200


# And use the same Unsupervised Learning algorithm as previously.

# In[50]:


estimator = KernelDensity
kwargs = dict(kernel='gaussian', bandwidth=0.2)


# In[51]:


estimator2 = SimpleGaussianKernel
kwargs2 = dict()


# This gives the following policy:

# In[35]:


policy = UnsupervisedLearning(nbArms, T_0=T_0, fit_every=fit_every, meanOf=meanOf, estimator=estimator, **kwargs)
get_ipython().run_line_magic('pinfo', 'policy')


# ----
# ## Comparing its performance on this Gaussian problem
# 
# We can compare the performance of this `UnsupervisedLearning(kde)` policy, on the same Gaussian problem, against three strategies:
# 
# - [`EmpiricalMeans`](https://smpybandits.github.io/docs/Policies.EmpiricalMeans.html#Policies.EmpiricalMeans.EmpiricalMeans), which only uses the empirical mean estimators $\hat{\mu_k}(t)$. It is known to be insufficient.
# - [`UCB`](https://smpybandits.github.io/docs/Policies.UCB.html#Policies.UCB.UCB), the UCB1 algorithm. It is known to be quite efficient.
# - [`Thompson`](https://smpybandits.github.io/docs/Policies.Thompson.html#Policies.Thompson.Thompson), the Thompson Sampling algorithm. It is known to be very efficient.
# - [`klUCB`](https://smpybandits.github.io/docs/Policies.klUCB.html#Policies.klUCB.klUCB), the kl-UCB algorithm, for Gaussian arms (`klucb = klucbGauss`). It is also known to be very efficient.

# ### Configuring an experiment
# I implemented in the [`Environment`](http://https://smpybandits.github.io/docs/Environment.html) module an [`Evaluator`](http://https://smpybandits.github.io/docs/Environment.Evaluator.html#Environment.Evaluator.Evaluator) class, very convenient to run experiments of Multi-Armed Bandit games without a sweat.
# 
# Let us use it!

# In[36]:


from SMPyBandits.Environment import Evaluator


# We will start with a small experiment, with a small horizon.

# In[54]:


HORIZON = 30000
REPETITIONS = 100
N_JOBS = min(REPETITIONS, 4)
means = [0.45, 0.5, 0.55]
ENVIRONMENTS = [ [Gaussian(mu, sigma=0.2) for mu in means] ]


# In[55]:


from SMPyBandits.Policies import EmpiricalMeans, UCB, Thompson, klUCB
from SMPyBandits.Policies import klucb_mapping, klucbGauss as _klucbGauss

sigma = 0.2
# Custom klucb function
def klucbGauss(x, d, precision=0.):
    """klucbGauss(x, d, sig2) with the good variance (= sigma)."""
    return _klucbGauss(x, d, sigma)

klucb = klucbGauss


# In[56]:


POLICIES = [
        # --- Naive algorithms
        {
            "archtype": EmpiricalMeans,
            "params": {}
        },
        # --- Our algorithm, with two Unsupervised Learning algorithms
        {
            "archtype": UnsupervisedLearning,
            "params": {
                "estimator": KernelDensity,
                "kernel": 'gaussian',
                "bandwidth": sigma,
                "T_0": T_0,
                "fit_every": fit_every,
                "meanOf": meanOf,
            }
        },
        {
            "archtype": UnsupervisedLearning,
            "params": {
                "estimator": SimpleGaussianKernel,
                "T_0": T_0,
                "fit_every": fit_every,
                "meanOf": meanOf,
            }
        },
        # --- Basic UCB1 algorithm
        {
            "archtype": UCB,
            "params": {}
        },
        # --- Thompson sampling algorithm
        {
            "archtype": Thompson,
            "params": {}
        },
        # --- klUCB algorithm, with Gaussian klucb function
        {
            "archtype": klUCB,
            "params": {
                "klucb": klucb
            }
        },
    ]


# In[57]:


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


# In[58]:


evaluation = Evaluator(configuration)


# ### Running an experiment
# 
# We asked to repeat the experiment $100$ times, so it will take a while... (about 10 minutes maximum).

# In[59]:


from SMPyBandits.Environment import tqdm


# In[60]:


get_ipython().run_cell_magic('time', '', 'for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):\n    # Evaluate just that env\n    evaluation.startOneEnv(envId, env)')


# ### Visualizing the results
# Now, we can plot some performance measures, like the regret, the best arm selection rate, the average reward etc.

# In[61]:


def plotAll(evaluation, envId=0):
    evaluation.printFinalRanking(envId)
    evaluation.plotRegrets(envId)
    evaluation.plotRegrets(envId, semilogx=True)
    evaluation.plotRegrets(envId, meanRegret=True)
    evaluation.plotBestArmPulls(envId)


# In[62]:


get_ipython().run_line_magic('pinfo', 'evaluation')


# In[63]:


plotAll(evaluation)


# ----
# ## Another experiment, with just more Gaussian arms

# In[89]:


HORIZON = 30000
REPETITIONS = 100
N_JOBS = min(REPETITIONS, 4)
means = [0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70]
ENVIRONMENTS = [ [Gaussian(mu, sigma=0.25) for mu in means] ]


# In[90]:


POLICIES = [
        # --- Our algorithm, with two Unsupervised Learning algorithms
        {
            "archtype": UnsupervisedLearning,
            "params": {
                "estimator": KernelDensity,
                "kernel": 'gaussian',
                "bandwidth": sigma,
                "T_0": T_0,
                "fit_every": fit_every,
                "meanOf": meanOf,
            }
        },
        {
            "archtype": UnsupervisedLearning,
            "params": {
                "estimator": SimpleGaussianKernel,
                "T_0": T_0,
                "fit_every": fit_every,
                "meanOf": meanOf,
            }
        },
        # --- Basic UCB1 algorithm
        {
            "archtype": UCB,
            "params": {}
        },
        # --- Thompson sampling algorithm
        {
            "archtype": Thompson,
            "params": {}
        },
        # --- klUCB algorithm, with Gaussian klucb function
        {
            "archtype": klUCB,
            "params": {
                "klucb": klucb
            }
        },
    ]


# In[91]:


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


# In[92]:


evaluation2 = Evaluator(configuration)


# ### Running the experiment
# 
# We asked to repeat the experiment $100$ times, so it will take a while...

# In[93]:


get_ipython().run_cell_magic('time', '', 'for envId, env in tqdm(enumerate(evaluation2.envs), desc="Problems"):\n    # Evaluate just that env\n    evaluation2.startOneEnv(envId, env)')


# ### Visualizing the results
# Now, we can plot some performance measures, like the regret, the best arm selection rate, the average reward etc.

# In[94]:


plotAll(evaluation2)


# ### Very good performance!
# Whoo, on this last experiment, the simple `UnsupervisedLearning(SimpleGaussianKernel)` works as well as Thompson Sampling (`Thompson`) !!
# 
# ... In fact, it was almost obvious : Thompson Sampling uses a Gamma posterior, while `UnsupervisedLearning(SimpleGaussianKernel)` works very similarly to Thompson Sampling (start with a flat kernel, fit it to the data, and to take decision, sample it and play the arm with the highest sample). `UnsupervisedLearning(SimpleGaussianKernel)` basically uses a Gaussian posterior, which is obviously better than a Gamma posterior for Gaussian arms!

# ----
# ## Another experiment, with Bernoulli arms
# 
# Let also try the same algorithms but on Bernoulli arms.

# In[95]:


from SMPyBandits.Arms import Bernoulli


# In[96]:


HORIZON = 30000
REPETITIONS = 100
N_JOBS = min(REPETITIONS, 4)
means = [0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70]
ENVIRONMENTS = [ [Bernoulli(mu) for mu in means] ]


# In[97]:


POLICIES = [
        # --- Our algorithm, with two Unsupervised Learning algorithms
        {
            "archtype": UnsupervisedLearning,
            "params": {
                "estimator": KernelDensity,
                "kernel": 'gaussian',
                "bandwidth": 0.1,
                "T_0": T_0,
                "fit_every": fit_every,
                "meanOf": meanOf,
            }
        },
        {
            "archtype": UnsupervisedLearning,
            "params": {
                "estimator": SimpleGaussianKernel,
                "T_0": T_0,
                "fit_every": fit_every,
                "meanOf": meanOf,
            }
        },
        # --- Basic UCB1 algorithm
        {
            "archtype": UCB,
            "params": {}
        },
        # --- Thompson sampling algorithm
        {
            "archtype": Thompson,
            "params": {}
        },
    ]


# In[98]:


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


# In[99]:


evaluation3 = Evaluator(configuration)


# ### Running the experiment
# 
# We asked to repeat the experiment $100$ times, so it will take a while...

# In[100]:


get_ipython().run_cell_magic('time', '', 'for envId, env in tqdm(enumerate(evaluation3.envs), desc="Problems"):\n    # Evaluate just that env\n    evaluation3.startOneEnv(envId, env)')


# ### Visualizing the results
# Now, we can plot some performance measures, like the regret, the best arm selection rate, the average reward etc.

# In[101]:


plotAll(evaluation3)


# ----
# ## Conclusion
# 
# This small simulation shows that with the appropriate tweaking of parameters, and on reasonably easy Gaussian Multi-Armed Bandit problems, one can use a **Unsupervised Learning** learning algorithm, being a **non on-line** algorithm (i.e., updating it at time step $t$ has a time complexity about $\mathcal{O}(K t)$ instead of $\mathcal{O}(K)$).
# 
# By tweaking cleverly the algorithm, mainly without refitting the model at every steps (e.g., but once every $T_1 = 1000$ steps), it works as well as the best-possible algorithm, here we compared against `Thompson` (Thompson Sampling) and `klUCB` (kl-UCB with Gaussian $\mathrm{KL}(x,y)$ function).
# 
# When comparing in terms of mean rewards, accumulated rewards, best-arm selection, and regret (loss against the best fixed-arm policy), this `UnsupervisedLearning(KernelDensity, ...)` algorithm performs as well as the others.

# ### Non-logarithmic regret
# But in terms of regret, it seems that the profile for `UnsupervisedLearning(KernelDensity, ...)` is **not** *asymptotically logarithmic*, contrarily to `Thompson` and `klUCB` (*cf.* see the first curve above, at the end on the right).
# 
# - Note that the horizon is not that large, $T = 30000$ is not very long.
# - And note that the four parameters, $T_0, T_1, M$ for the `UnsupervisedLearning` part, and $\mathrm{bandwidth}$ for the `KernelDensity` part, have all been (manually) *tweaked* for this setting. For instance, $\mathrm{bandwidth} = \sigma = 0.2$ is the same as the one used for the arms (but in a real-world scenario, this would be unknown), $T_0,T_1$ is adapted to $T$, and $M$ is adapted to $\sigma$ also (to reduce variance of the samples for the models).

# ### Comparing *time complexity*
# Another aspect is the *time complexity* of the `UnsupervisedLearning(KernelDensity, ...)` algorithm.
# In the simulation above, we saw that it took about $42\;\mathrm{min}$ to do $1000$ experiments of horizon $T = 30000$ (about $8.4 10^{-5} \; \mathrm{s}$ by time step), against $5.5\;\mathrm{min}$ for Thompson Sampling : even with fitting the unsupervised learning model only *once every $T_1 = 1000$ steps*, it is still about $8$ times slower than Thompson Sampling or UCB !
# 
# It is not that much, but still...

# In[102]:


time_by_loop_UL_KD = (42 * 60.) / (REPETITIONS * HORIZON)
time_by_loop_UL_KD


# In[103]:


time_by_loop_TS = (5.5 * 60.) / (REPETITIONS * HORIZON)
time_by_loop_TS


# In[104]:


42 / 5.5


# ### Not so efficient for Bernoulli arms
# Similarly, the last experiment showed that this `UnsupervisedLearning` policy was not so efficient on Bernoulli problems, with a Gaussian kernel.
# 
# A better approach could have been to use a Bernoulli "kernel", i.e., fitting a Bernoulli distribution on each arm.
# 
# > I implemented this for my framework, see [here the documentation for `SimpleBernoulliKernel`](https://smpybandits.github.io/docs/Policies.UnsupervisedLearning.html#Policies.UnsupervisedLearning.SimpleBernoulliKernel), but I will not present it here.

# ----
# This notebook is here to illustrate my [SMPyBandits](https://smpybandits.github.io/) library, for which a complete documentation is available, [here at https://smpybandits.github.io/](https://smpybandits.github.io/).
# 
# > That's it for this demo! See you, folks!
