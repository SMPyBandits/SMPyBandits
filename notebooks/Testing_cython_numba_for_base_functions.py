
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Requirements" data-toc-modified-id="Requirements-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Requirements</a></div><div class="lev1 toc-item"><a href="#KL-Functions" data-toc-modified-id="KL-Functions-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>KL Functions</a></div><div class="lev2 toc-item"><a href="#klGauss" data-toc-modified-id="klGauss-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>klGauss</a></div><div class="lev2 toc-item"><a href="#klBern" data-toc-modified-id="klBern-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>klBern</a></div><div class="lev1 toc-item"><a href="#Threshold-functions" data-toc-modified-id="Threshold-functions-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Threshold functions</a></div><div class="lev2 toc-item"><a href="#Threshold-for-GLR-Bernoulli" data-toc-modified-id="Threshold-for-GLR-Bernoulli-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Threshold for GLR Bernoulli</a></div>

# # Requirements

# In[1]:


import numpy as np

import numba


# In[2]:


import cython
get_ipython().run_line_magic('load_ext', 'cython')


# # KL Functions

# ## klGauss

# Generating some data:

# In[3]:


def random_Gaussian(sig2=0.25):
    return sig2 * np.random.randn()


# In[4]:


get_ipython().run_line_magic('timeit', '(random_Gaussian(), random_Gaussian())')


# In pure Python:

# In[5]:


def klGauss(x: float, y: float, sig2=0.25) -> float:
    return (x - y)**2 / (2 * sig2**x)


# In[6]:


get_ipython().run_line_magic('timeit', 'klGauss(random_Gaussian(), random_Gaussian())')


# With numba:

# In[7]:


@numba.jit(nopython=True)
def klGauss_numba(x: float, y: float, sig2=0.25) -> float:
    return (x - y)**2 / (2 * sig2**x)


# In[8]:


help(klGauss_numba)


# In[9]:


get_ipython().run_line_magic('timeit', 'klGauss_numba(random_Gaussian(), random_Gaussian())')


# In[10]:


get_ipython().run_line_magic('timeit', 'klGauss_numba(random_Gaussian(), random_Gaussian())')


# In[11]:


print(f"Speed up using Numba for klGauss was: {(1290-993)/(20300-993):.2g} faster!")


# With Cython

# In[12]:


get_ipython().run_cell_magic('cython', '--annotate', '\ndef klGauss_cython(double x, double y, double sig2=0.25) -> double:\n    return (x - y)**2 / (2 * sig2**x)')


# In[13]:


help(klGauss_cython)


# In[14]:


get_ipython().run_line_magic('timeit', 'klGauss_cython(random_Gaussian(), random_Gaussian())')


# In[15]:


print(f"Speed up using Cython for klGauss was: {(1290-993)/(1100-993):.2g} faster!")


# ## klBern

# Generating some data:

# In[16]:


def random_Bern():
    return np.random.random()


# In[17]:


get_ipython().run_line_magic('timeit', '(random_Bern(), random_Bern())')


# In pure Python:

# In[18]:


from math import log

def klBern(x: float, y: float) -> float:
    x = max(1e-7, min(1 - 1e-7, x))
    x = max(1e-7, min(1 - 1e-7, x))
    return x * log(x/y) + (1-x) * log((1-x)/(1-y))


# In[19]:


get_ipython().run_line_magic('timeit', 'klBern(random_Bern(), random_Bern())')


# With numba:

# In[20]:


from math import log

@numba.jit(nopython=True)
def klBern_numba(x: float, y: float) -> float:
    x = max(1e-7, min(1 - 1e-7, x))
    x = max(1e-7, min(1 - 1e-7, x))
    return x * log(x/y) + (1-x) * log((1-x)/(1-y))


# In[21]:


help(klBern_numba)


# In[22]:


get_ipython().run_line_magic('timeit', 'klBern_numba(random_Bern(), random_Bern())')


# In[23]:


get_ipython().run_line_magic('timeit', 'klBern_numba(random_Bern(), random_Bern())')


# In[24]:


print(f"Speed up using Numba for klBern was: {(1740-753)/(996-753):.2g} faster!")


# With Cython

# In[25]:


get_ipython().run_line_magic('load_ext', 'cython')


# In[26]:


get_ipython().run_cell_magic('cython', '--annotate', 'from libc.math cimport log\n\ndef klBern_cython(double x, double y) -> double:\n    x = max(1e-7, min(1 - 1e-7, x))\n    x = max(1e-7, min(1 - 1e-7, x))\n    return x * log(x/y) + (1-x) * log((1-x)/(1-y))')


# In[27]:


help(klBern_cython)


# In[28]:


get_ipython().run_line_magic('timeit', 'klBern_cython(random_Bern(), random_Bern())')


# In[29]:


print(f"Speed up using Cython for klBern was: {(1740-753)/(861-753):.2g} faster!")


# # Threshold functions

# ## Threshold for GLR Bernoulli

# Generating some data:

# In[30]:


def random_t0_s_t_delta(min_t: int=100, max_t: int=1000) -> (int, int, int, float):
    t0 = 0
    t = np.random.randint(min_t, max_t + 1)
    s = np.random.randint(t0, t)
    delta = np.random.choice([0.1, 0.05, 0.01, 0.005, 0.001, max(0.0005, 1/t)])
    return (t0, s, t, delta)


# In[31]:


get_ipython().run_line_magic('timeit', 'random_t0_s_t_delta()')


# In pure Python:

# In[32]:


def threshold(t0: int, s: int, t: int, delta: float) -> float:
    return np.log((s - t0 + 1) * (t - s) / delta)


# In[33]:


get_ipython().run_line_magic('timeit', 'threshold(*random_t0_s_t_delta())')


# It's *way* faster to use `math.log` instead of `numpy.log` (of course)!

# In[34]:


from math import log

def threshold2(t0: int, s: int, t: int, delta: float) -> float:
    return log((s - t0 + 1) * (t - s) / delta)


# In[35]:


get_ipython().run_line_magic('timeit', 'threshold2(*random_t0_s_t_delta())')


# In numba:

# In[36]:


from math import log

@numba.jit(nopython=True)
def threshold_numba(t0: int, s: int, t: int, delta: float) -> float:
    return log((s - t0 + 1) * (t - s) / delta)


# In[37]:


help(threshold_numba)


# In[38]:


get_ipython().run_line_magic('timeit', 'threshold_numba(*random_t0_s_t_delta())')


# In[39]:


print(f"Speed up using Cython for thresold was: {(7510-7200)/(7750-7200):.2g} faster!")


# In Cython:

# In[40]:


get_ipython().run_cell_magic('cython', '--annotate', 'from libc.math cimport log\n\ncpdef double threshold_cython(int t0, int s, int t, double delta):\n    return log((s - t0 + 1) * (t - s) / delta)')


# In[41]:


get_ipython().run_cell_magic('cython', '--annotate', 'from libc.math cimport log\n\ndef threshold_cython2(int t0, int s, int t, double delta) -> double:\n    return log((s - t0 + 1) * (t - s) / delta)')


# In[42]:


help(threshold_cython)


# In[43]:


get_ipython().run_line_magic('timeit', 'threshold_cython(*random_t0_s_t_delta())')


# In[44]:


get_ipython().run_line_magic('timeit', 'threshold_cython2(*random_t0_s_t_delta())')


# In[45]:


print(f"Speed up using Cython for thresold was: {abs(7510-7200)/abs(7070-7200):.2g} faster!")

