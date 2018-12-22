# -*- coding: utf-8 -*-
""" ``Policies.Experimentals`` module : contains experimental or unfinished (single-player) bandits algorithms:

- Index based UCB algorithms: :class:`UCBlog10`, :class:`UCBwrong`, :class:`UCBlog10alpha`, :class:`UCBcython`, :class:`UCBjulia`,

- Based on Kullback-Leibler divergence: :class:`klUCBlog10`, :class:`klUCBloglog10`,

- Empirical KL-UCB algorithm: :class:`KLempUCB` (does not work with the C optimized version of :mod:`kullback`),

- An *experimental* policy, using Unsupervised Learning: :class:`UnsupervisedLearning`,

- An *experimental* policy, using Black-box optimization: :class:`BlackBoxOpt`,

- Bayesian algorithms: :class:`ThompsonRobust`,

- **New!** The UCBoost (Upper Confidence bounds with Boosting) policies, first with no boosting, in module :mod:`UCBoost_faster`: :class:`UCBoost_faster.UCB_sq`, :class:`UCBoost_faster.UCB_bq`, :class:`UCBoost_faster.UCB_h`, :class:`UCBoost_faster.UCB_lb`, :class:`UCBoost_faster.UCB_t`, and then the ones with non-adaptive boosting: :class:`UCBoost_faster.UCBoost_bq_h_lb`, :class:`UCBoost_faster.UCBoost_bq_h_lb_t`, :class:`UCBoost_faster.UCBoost_bq_h_lb_t_sq`, :class:`UCBoost_faster.UCBoost`, and finally the epsilon-approximation boosting with :class:`UCBoost_faster.UCBoostEpsilon`. These versions use Cython for some functions.

- **New!** The UCBoost (Upper Confidence bounds with Boosting) policies, first with no boosting, in module :mod:`UCBoost_cython`: :class:`UCBoost_cython.UCB_sq`, :class:`UCBoost_cython.UCB_bq`, :class:`UCBoost_cython.UCB_h`, :class:`UCBoost_cython.UCB_lb`, :class:`UCBoost_cython.UCB_t`, and then the ones with non-adaptive boosting: :class:`UCBoost_cython.UCBoost_bq_h_lb`, :class:`UCBoost_cython.UCBoost_bq_h_lb_t`, :class:`UCBoost_cython.UCBoost_bq_h_lb_t_sq`, :class:`UCBoost_cython.UCBoost`, and finally the epsilon-approximation boosting with :class:`UCBoost_cython.UCBoostEpsilon`. These versions use Cython for the whole code.
"""

from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

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
    try:
        from .UCB import UCB as UCBcython
    except ImportError:
        import sys; sys.path.insert(0, '..')
        from UCB import UCB as UCBcython

# --- Thompson sampling index policy
from .ThompsonRobust import ThompsonRobust

# --- Kullback-Leibler based index policy
from .klUCBlog10 import klUCBlog10  # With log10(t) instead of log(t) = ln(t)
from .klUCBloglog10 import klUCBloglog10  # With log10(t) + c log10(log10(t)) and c = 1 (variable)

from .KLempUCB import KLempUCB  # Empirical KL UCB

# --- Using unsupervised learning, from scikit-learn
from .UnsupervisedLearning import FittingModel, SimpleGaussianKernel, SimpleBernoulliKernel, UnsupervisedLearning

from .BlackBoxOpt import default_estimator, default_optimizer, BlackBoxOpt

# From [Fang Liu et al, 2018]
from .UCBoost_faster import UCB_sq as UCB_sq_faster, UCB_bq as UCB_bq_faster, UCB_h as UCB_h_faster, UCB_lb as UCB_lb_faster, UCB_t as UCB_t_faster, UCBoost_bq_h_lb as UCBoost_bq_h_lb_faster, UCBoost_bq_h_lb_t as UCBoost_bq_h_lb_t_faster, UCBoost_bq_h_lb_t_sq as UCBoost_bq_h_lb_t_sq_faster, UCBoost as UCBoost_faster, UCBoostEpsilon as UCBoostEpsilon_faster

try:
    import pyximport; pyximport.install()
    from .UCBoost_cython import UCB_sq as UCB_sq_cython, UCB_bq as UCB_bq_cython, UCB_h as UCB_h_cython, UCB_lb as UCB_lb_cython, UCB_t as UCB_t_cython, UCBoost_bq_h_lb as UCBoost_bq_h_lb_cython, UCBoost_bq_h_lb_t as UCBoost_bq_h_lb_t_cython, UCBoost_bq_h_lb_t_sq as UCBoost_bq_h_lb_t_sq_cython, UCBoost as UCBoost_cython, UCBoostEpsilon as UCBoostEpsilon_cython
except ImportError:
    print("Warning: the 'UCBoost_cython' module failed to be imported. Maybe there is something wrong with your installation of Cython?")  # DEBUG
    from .UCBoost_faster import UCB_sq as UCB_sq_cython, UCB_bq as UCB_bq_cython, UCB_h as UCB_h_cython, UCB_lb as UCB_lb_cython, UCB_t as UCB_t_cython, UCBoost_bq_h_lb as UCBoost_bq_h_lb_cython, UCBoost_bq_h_lb_t as UCBoost_bq_h_lb_t_cython, UCBoost_bq_h_lb_t_sq as UCBoost_bq_h_lb_t_sq_cython, UCBoost as UCBoost_cython, UCBoostEpsilon as UCBoostEpsilon_cython

del pyximport
