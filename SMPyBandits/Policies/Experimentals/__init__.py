# -*- coding: utf-8 -*-
""" ``Policies.Experimentals`` module : contains experimental or unfinished (single-player) bandits algorithms:

- Index based UCB algorithms: :class:`UCBlog10`, :class:`UCBwrong`, :class:`UCBlog10alpha`, :class:`UCBcython`, :classl:`UCBjulia`,

- Based on Kullback-Leibler divergence: :class:`klUCBlog10`, :class:`klUCBloglog10`,

- Empirical KL-UCB algorithm: :class:`KLempUCB` (does not work with the C optimized version of :mod:`kullback`),

- An *experimental* policy, using Unsupervised Learning: :class:`UnsupervisedLearning`,

- An *experimental* policy, using Black-box optimization: :class:`BlackBoxOpt`,

- Bayesian algorithms: :class:`ThompsonRobust`,

- **New!** The UCBoost (Upper Confidence bounds with Boosting) policies, first with no boosting: :class:`UCB_sq_faster`, :class:`UCB_bq_faster`, :class:`UCB_h_faster`, :class:`UCB_lb_faster`, :class:`UCB_t_faster`, and then the ones with non-adaptive boosting: :class:`UCBoost_bq_h_lb_faster`, :class:`UCBoost_bq_h_lb_t_faster`, :class:`UCBoost_bq_h_lb_t_sq_faster`, :class:`UCBoost_faster`, and finally the epsilon-approximation boosting with :class:`UCBoostEpsilon_faster`. These versions use Cython for some functions.

- **New!** The UCBoost (Upper Confidence bounds with Boosting) policies, first with no boosting: :class:`UCB_sq_cython`, :class:`UCB_bq_cython`, :class:`UCB_h_cython`, :class:`UCB_lb_cython`, :class:`UCB_t_cython`, and then the ones with non-adaptive boosting: :class:`UCBoost_bq_h_lb_cython`, :class:`UCBoost_bq_h_lb_t_cython`, :class:`UCBoost_bq_h_lb_t_sq_cython`, :class:`UCBoost_cython`, and finally the epsilon-approximation boosting with :class:`UCBoostEpsilon_cython`. These versions use Cython for the whole code.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from sys import path; path.insert(0, '..')

# --- Using unsupervised learning, from scikit-learn
from .UnsupervisedLearning import FittingModel, SimpleGaussianKernel, SimpleBernoulliKernel, UnsupervisedLearning

from .BlackBoxOpt import default_estimator, default_optimizer, BlackBoxOpt

# --- Simple UCB policies
from .UCBlog10 import UCBlog10  # With log10(t) instead of log(t) = ln(t)
from .UCBwrong import UCBwrong  # With a volontary typo!
from .UCBlog10alpha import UCBlog10alpha  # Different indexes

from .UCBjulia import UCBjulia  # XXX Experimental!

try:
    import pyximport; pyximport.install()
    from .UCBcython import UCBcython  # XXX Experimental!
except:
    print("Warning: the 'UCBcython' module failed to be imported. Maybe there is something wrong with your installation of Cython?")  # DEBUG
    from .UCB import UCB as UCBcython

# --- Thompson sampling index policy
from .ThompsonRobust import ThompsonRobust

# --- Kullback-Leibler based index policy
from .klUCBlog10 import klUCBlog10  # With log10(t) instead of log(t) = ln(t)
from .klUCBloglog10 import klUCBloglog10  # With log10(t) + c log10(log10(t)) and c = 1 (variable)

from .KLempUCB import KLempUCB  # Empirical KL UCB

# From [Fang Liu et al, 2018]
from .UCBoost_faster import UCB_sq as UCB_sq_faster, UCB_bq as UCB_bq_faster, UCB_h as UCB_h_faster, UCB_lb as UCB_lb_faster, UCB_t as UCB_t_faster, UCBoost_bq_h_lb as UCBoost_bq_h_lb_faster, UCBoost_bq_h_lb_t as UCBoost_bq_h_lb_t_faster, UCBoost_bq_h_lb_t_sq as UCBoost_bq_h_lb_t_sq_faster, UCBoost as UCBoost_faster, UCBoostEpsilon as UCBoostEpsilon_faster

try:
    import pyximport; pyximport.install()
    from .UCBoost_cython import UCB_sq as UCB_sq_cython, UCB_bq as UCB_bq_cython, UCB_h as UCB_h_cython, UCB_lb as UCB_lb_cython, UCB_t as UCB_t_cython, UCBoost_bq_h_lb as UCBoost_bq_h_lb_cython, UCBoost_bq_h_lb_t as UCBoost_bq_h_lb_t_cython, UCBoost_bq_h_lb_t_sq as UCBoost_bq_h_lb_t_sq_cython, UCBoost as UCBoost_cython, UCBoostEpsilon as UCBoostEpsilon_cython
except ImportError:
    print("Warning: the 'UCBoost_cython' module failed to be imported. Maybe there is something wrong with your installation of Cython?")  # DEBUG
    from .UCBoost_faster import UCB_sq as UCB_sq_cython, UCB_bq as UCB_bq_cython, UCB_h as UCB_h_cython, UCB_lb as UCB_lb_cython, UCB_t as UCB_t_cython, UCBoost_bq_h_lb as UCBoost_bq_h_lb_cython, UCBoost_bq_h_lb_t as UCBoost_bq_h_lb_t_cython, UCBoost_bq_h_lb_t_sq as UCBoost_bq_h_lb_t_sq_cython, UCBoost as UCBoost_cython, UCBoostEpsilon as UCBoostEpsilon_cython

del path
del pyximport
