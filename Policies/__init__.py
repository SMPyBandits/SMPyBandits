# -*- coding: utf-8 -*-
""" Policies module : contains various bandits algorithms:

- "Stupid" algorithms: :class:`Uniform`, :class:`UniformOnSome`, :class:`TakeFixedArm`, :class:`TakeRandomFixedArm`,

- Greedy algorithms: :class:`EpsilonGreedy`, :class:`EpsilonFirst`, :class:`EpsilonDecreasing`,

- Probabilistic weighting algorithms: :class:`Softmax`, :class:`SoftmaxDecreasing`, :class:`SoftMix`, :class:`SoftmaxWithHorizon`, Exp3, Exp3Decreasing, Exp3SoftMix, Exp3WithHorizon, :class:`ProbabilityPursuit`,

- Index based and UCB algorithms: :class:`EmpiricalMeans`, :class:`UCB`, UCBlog10, :class:`UCBwrong`, UCBlog10alpha, :class:`UCBalpha`, :class:`UCBmin`, :class:`UCBplus`, :class:`UCBrandomInit`, :class:`UCBV`, :class:`UCBVtuned`, :class:`UCBH`, :class:`MOSS`, :class:`MOSSH`, :class:`CPUCB`,

- Bayesian algorithms: :class:`Thompson`, :class:`ThompsonRobust`, :class:`BayesUCB`,

- Based on Kullback-Leibler divergence: :class:`klUCB`, klUCBlog10, :class:`klUCBloglog`, klUCBloglog10, :class:`klUCBPlus`, :class:`klUCBH`, :class:`klUCBHPlus`, :class:`klUCBPlusPlus`,

- Empirical KL-UCB algorithm: :class:`KLempUCB`,

- Other index algorithms: :class:`DMED`, :class:`DMEDPlus`, :class:`OCUCB`,

- Hybrids algorithms, mixing Bayesian and UCB indexes: :class:`AdBandit`,

- Aggregation algorithms: :class:`Aggr`,

- Finite-Horizon Gittins index, approximated version: :class:`ApproximatedFHGittins`,

- *New!* An experimental policy, using Unsupervised Learning: :class:`UnsupervisedLearning`,

- *New!* An experimental policy, using Black-box optimization: :class:`BlackBoxOpt`,

- Some are designed only for (fully decentralized) multi-player games: :class:`MusicalChair`, :class:`MEGA`.


All policies have the same interface, as described in :class:`BasePolicy`,
in order to use them in any experiment with the following approach:

>>> my_policy = Policy(nbArms, *args, lower=0, amplitude=1, **kwargs)
>>> my_policy.startGame()  # start the game
>>> for t in range(T):
>>>     chosen_arm_t = k_t = my_policy.choice()  # chose one arm
>>>     reward_t     = sampled from an arm k_t   # sample a reward
>>>     my_policy.getReward(k_t, reward_t)       # give it the the policy
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

from .Posterior import Beta, Gamma, Gauss

# --- Mine, uniform ones or fixed arm / fixed subset ones
from .Uniform import Uniform
from .UniformOnSome import UniformOnSome
from .TakeFixedArm import TakeFixedArm
from .TakeRandomFixedArm import TakeRandomFixedArm

# --- Mine, simple exploratory policies
from .EpsilonGreedy import EpsilonGreedy
from .EpsilonFirst import EpsilonFirst
# --- Mine, simple exploratory policies
from .EpsilonDecreasing import EpsilonDecreasing
from .EpsilonDecreasingMEGA import EpsilonDecreasingMEGA
from .EpsilonExpDecreasing import EpsilonExpDecreasing
from .EmpiricalMeans import EmpiricalMeans

# --- Mine, Softmax and Exp3 policies
from .Softmax import Softmax, SoftmaxDecreasing, SoftMix, SoftmaxWithHorizon
from .Exp3 import Exp3, Exp3Decreasing, Exp3SoftMix, Exp3WithHorizon, Exp3ELM
from .ProbabilityPursuit import ProbabilityPursuit

# --- Using unsupervised learning, from scikit-learn
from .UnsupervisedLearning import FittingModel, SimpleGaussianKernel, SimpleBernoulliKernel, UnsupervisedLearning

from .BlackBoxOpt import default_estimator, default_optimizer, BlackBoxOpt

# --- Simple UCB policies
from .UCB import UCB
from .UCBH import UCBH          # With log(T) instead of log(t)
from .UCBlog10 import UCBlog10  # With log10(t) instead of log(t) = ln(t)
from .UCBwrong import UCBwrong  # With a volontary typo!
from .UCBalpha import UCBalpha  # Different indexes
from .UCBlog10alpha import UCBlog10alpha  # Different indexes
from .UCBmin import UCBmin      # Different indexes
from .UCBplus import UCBplus    # Different indexes
from .UCBrandomInit import UCBrandomInit

# --- UCB policies with variance terms
from .UCBV import UCBV          # Different indexes
from .UCBVtuned import UCBVtuned  # Different indexes

# --- Clopper-Pearson UCB
from .CPUCB import CPUCB        # Different indexes

# --- MOSS index policy
from .MOSS import MOSS
from .MOSSH import MOSSH  # Knowing the horizon

# --- Thompson sampling index policy
from .Thompson import Thompson
from .ThompsonRobust import ThompsonRobust

# --- Bayesian index policy
from .BayesUCB import BayesUCB

# --- Kullback-Leibler based index policy
from .klUCB import klUCB
from .klUCBlog10 import klUCBlog10  # With log10(t) instead of log(t) = ln(t)
from .klUCBloglog import klUCBloglog  # With log(t) + c log(log(t)) and c = 1 (variable)
from .klUCBloglog10 import klUCBloglog10  # With log10(t) + c log10(log10(t)) and c = 1 (variable)
from .klUCBPlus import klUCBPlus    # Different indexes
from .klUCBH import klUCBH          # Knowing the horizon
from .klUCBHPlus import klUCBHPlus  # Different indexes
from .klUCBPlusPlus import klUCBPlusPlus  # Different indexes
from .KLempUCB import KLempUCB  # Empirical KL UCB

# From [Honda & Takemura, COLT 2010]
from .DMED import DMED, DMEDPlus

# From [Lattimore, 2016]
from .OCUCB import OCUCB

# From https://github.com/flaviotruzzi/AdBandits/
from .AdBandits import AdBandits

# --- Mine, aggregated ones, like Exp4  FIXME give it a better name!
from .Aggr import Aggr

# --- Gittins index policy
from .ApproximatedFHGittins import ApproximatedFHGittins  # Approximated Finite-Horizon Gittins index


# --- Mine, implemented from state-of-the-art papers on multi-player policies

from .MusicalChair import MusicalChair  # Cf. [Shamir et al., 2015](https://arxiv.org/abs/1512.02866)
# from .DynamicMusicalChair import DynamicMusicalChair  # FIXME write it! Can be just a subclass of MusicalChair

from .MEGA import MEGA  # Cf. [Avner & Mannor, 2014](https://arxiv.org/abs/1404.5421)


# --- KL-UCB index functions
from .usenumba import jit

from .kullback import klucbBern, klucbExp, klucbGauss, klucbPoisson, klucbGamma

#: Maps name of arms to kl functions
klucb_mapping = {
    "Bernoulli": klucbBern,
    "Exponential": klucbExp,
    "Gaussian": klucbGauss,
    "Poisson": klucbPoisson,
    "Gamma": klucbGamma,
}
