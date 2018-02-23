# -*- coding: utf-8 -*-
""" Policies module : contains various bandits algorithms:

- "Stupid" algorithms: :class:`Uniform`, :class:`UniformOnSome`, :class:`TakeFixedArm`, :class:`TakeRandomFixedArm`,

- Greedy algorithms: :class:`EpsilonGreedy`, :class:`EpsilonFirst`, :class:`EpsilonDecreasing`,
- And two variants of the Explore-Then-Commit policy: :class:`ExploreThenCommit.ETC_KnownGap`, :class:`ExploreThenCommit.ETC_RandomStop`,

- Probabilistic weighting algorithms: :class:`Hedge`, :class:`Softmax`, :class:`Softmax.SoftmaxDecreasing`, :class:`Softmax.SoftMix`, :class:`Softmax.SoftmaxWithHorizon`, :class:`Exp3`, :class:`Exp3.Exp3Decreasing`, :class:`Exp3.Exp3SoftMix`, :class:`Exp3.Exp3WithHorizon`, :class:`Exp3.Exp3ELM`, :class:`ProbabilityPursuit`, :class:`Exp3PlusPlus`, and a smart variant :class:`BoltzmannGumbel`,

- Index based UCB algorithms: :class:`EmpiricalMeans`, :class:`UCB`, :class:`UCBlog10`, :class:`UCBwrong`, :class:`UCBlog10alpha`, :class:`UCBalpha`, :class:`UCBmin`, :class:`UCBplus`, :class:`UCBrandomInit`, :class:`UCBV`, :class:`UCBVtuned`, :class:`UCBH`, :class:`CPUCB`,

- Index based MOSS algorithms: :class:`MOSS`, :class:`MOSSH`, :class:`MOSSAnytime`, :class:`MOSSExperimental`,

- Bayesian algorithms: :class:`Thompson`, :class:`ThompsonRobust`, :class:`BayesUCB`,

- Based on Kullback-Leibler divergence: :class:`klUCB`, :class:`klUCBlog10`, :class:`klUCBloglog`, :class:`klUCBloglog10`, :class:`klUCBPlus`, :class:`klUCBH`, :class:`klUCBHPlus`, :class:`klUCBPlusPlus`,

- Empirical KL-UCB algorithm: :class:`KLempUCB` (FIXME),

- Other index algorithms: :class:`DMED`, :class:`DMED.DMEDPlus`, :class:`OCUCB`, :class:`UCBdagger`,

- Hybrids algorithms, mixing Bayesian and UCB indexes: :class:`AdBandits`,

- Aggregation algorithms: :class:`Aggregator` (mine, it's awesome, go on try it!), and :class:`CORRAL`, :class:`LearnExp`,

- Finite-Horizon Gittins index, approximated version: :class:`ApproximatedFHGittins`,

- An *experimental* policy, using Unsupervised Learning: :class:`UnsupervisedLearning`,

- An *experimental* policy, using Black-box optimization: :class:`BlackBoxOpt`,

- An experimental policy, using a sliding window of for instance 100 draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible), :class:`SlidingWindowRestart`, and 3 versions for UCB, UCBalpha and klUCB: :class:`SlidingWindowRestart.SWR_UCB`, :class:`SlidingWindowRestart.SWR_UCBalpha`, :class:`SlidingWindowRestart.SWR_klUCB` (my algorithm, unpublished yet),

- An experimental policy, using just a sliding window of for instance 100 draws, :class:`SlidingWindowUCB.SWUCB`, and :class:`SlidingWindowUCB.SWUCBPlus` if the horizon is known.

- Another experimental policy with a discount factor, :class:`DiscountedUCB` and :class:`DiscountedUCB.DiscountedUCBPlus`.

- A policy designed to tackle sparse stochastic bandit problems, :class:`SparseUCB`, :class:`SparseklUCB`, and :class:`SparseWrapper` that can be used with *any* index policy.

- A policy that implements a "smart doubling trick" to turn any horizon-dependent policy into a horizon-independent policy without loosing in performances: :class:`DoublingTrickWrapper`,

- An *experimental* policy, implementing a another kind of doubling trick to turn any policy that needs to know the range :math:`[a,b]` of rewards a policy that don't need to know the range, and that adapt dynamically from the new observations, :class:`WrapRange`,

- The *Optimal Sampling for Structured Bandits* (OSSB) policy: :class:`OSSB` (it is more generic and can be applied to almost any kind of bandit problem, it works fine for classical stationary bandits but it is not optimal),

- **New!** The Best Empirical Sampled Average (BESA) policy: :class:`BESA` (it works crazily well),

- Some are designed only for (fully decentralized) multi-player games: :class:`MusicalChair`, :class:`MEGA`.


All policies have the same interface, as described in :class:`BasePolicy`,
in order to use them in any experiment with the following approach: ::

    my_policy = Policy(nbArms, *args, lower=0, amplitude=1, **kwargs)
    my_policy.startGame()  # start the game
    for t in range(T):
        chosen_arm_t = k_t = my_policy.choice()  # chose one arm
        reward_t     = sampled from an arm k_t   # sample a reward
        my_policy.getReward(k_t, reward_t)       # give it the the policy
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

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

# --- Variants on EpsilonFirst, Explore-Then-Commit from E.Kaufmann's slides at IEEE ICC 2017
from .ExploreThenCommit import ETC_KnownGap, ETC_RandomStop

# --- Mine, Softmax and Exp3 policies
from .Softmax import Softmax, SoftmaxDecreasing, SoftMix, SoftmaxWithHorizon
from .Exp3 import Exp3, Exp3Decreasing, Exp3SoftMix, Exp3WithHorizon, Exp3ELM
from .Exp3PlusPlus import Exp3PlusPlus
from .ProbabilityPursuit import ProbabilityPursuit
from .BoltzmannGumbel import BoltzmannGumbel
from .Hedge import Hedge, HedgeDecreasing, HedgeWithHorizon

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

from .UCBjulia import UCBjulia  # XXX Experimental!

# --- UCB policies with variance terms
from .UCBV import UCBV          # Different indexes
from .UCBVtuned import UCBVtuned  # Different indexes

# --- SparseUCB and variants policies for sparse stochastic bandit
from .SparseUCB import SparseUCB
from .SparseklUCB import SparseklUCB
from .SparseWrapper import SparseWrapper  # generic wrapper class

# --- Clopper-Pearson UCB
from .CPUCB import CPUCB        # Different indexes

# --- MOSS index policy
from .MOSS import MOSS
from .MOSSH import MOSSH  # Knowing the horizon
from .MOSSAnytime import MOSSAnytime  # Without knowing the horizon
from .MOSSExperimental import MOSSExperimental  # Without knowing the horizon, experimental

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

# From [Lattimore, 2017]
from .UCBdagger import UCBdagger

# From [Combes et al, 2017]
from .OSSB import OSSB

# From [Baransi et al, 2014]
from .BESA import BESA

# From https://github.com/flaviotruzzi/AdBandits/
from .AdBandits import AdBandits

# --- Mine, aggregation algorithm, like Exp4
from .Aggregator import Aggregator
# --- Others aggregation algorithms
from .CORRAL import CORRAL
from .LearnExp import LearnExp

# --- Gittins index policy
from .ApproximatedFHGittins import ApproximatedFHGittins  # Approximated Finite-Horizon Gittins index

# --- Smart policies trying to adapt to dynamically changing environments
from .SlidingWindowRestart import SlidingWindowRestart, SWR_UCB, SWR_UCBalpha, SWR_klUCB
from .SlidingWindowUCB import SWUCB, SWUCBPlus

from .DiscountedUCB import DiscountedUCB, DiscountedUCBPlus

from .DoublingTrickWrapper import DoublingTrickWrapper, next_horizon__arithmetic, next_horizon__geometric, next_horizon__exponential, next_horizon__exponential_fast, next_horizon__exponential_slow, next_horizon__exponential_generic, breakpoints

from .WrapRange import WrapRange

# --- Mine, implemented from state-of-the-art papers on multi-player policies

from .MusicalChair import MusicalChair, optimalT0  # Cf. [Shamir et al., 2015](https://arxiv.org/abs/1512.02866)
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
