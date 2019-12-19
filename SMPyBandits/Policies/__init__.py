# -*- coding: utf-8 -*-
""" ``Policies`` module : contains all the (single-player) bandits algorithms:

- "Stupid" algorithms: :class:`Uniform`, :class:`UniformOnSome`, :class:`TakeFixedArm`, :class:`TakeRandomFixedArm`,

- Greedy algorithms: :class:`EpsilonGreedy`, :class:`EpsilonFirst`, :class:`EpsilonDecreasing`, :class:`EpsilonDecreasingMEGA`, :class:`EpsilonExpDecreasing`,
- And variants of the Explore-Then-Commit policy: :class:`ExploreThenCommit.ETC_KnownGap`, :class:`ExploreThenCommit.ETC_RandomStop`, :class:`ExploreThenCommit.ETC_FixedBudget`, :class:`ExploreThenCommit.ETC_SPRT`, :class:`ExploreThenCommit.ETC_BAI`, :class:`ExploreThenCommit.DeltaUCB`,

- Probabilistic weighting algorithms: :class:`Hedge`, :class:`Softmax`, :class:`Softmax.SoftmaxDecreasing`, :class:`Softmax.SoftMix`, :class:`Softmax.SoftmaxWithHorizon`, :class:`Exp3`, :class:`Exp3.Exp3Decreasing`, :class:`Exp3.Exp3SoftMix`, :class:`Exp3.Exp3WithHorizon`, :class:`Exp3.Exp3ELM`, :class:`ProbabilityPursuit`, :class:`Exp3PlusPlus`, a smart variant :class:`BoltzmannGumbel`, and a recent extension :class:`TsallisInf`,

- Index based UCB algorithms: :class:`EmpiricalMeans`, :class:`UCB`, :class:`UCBalpha`, :class:`UCBmin`, :class:`UCBplus`, :class:`UCBrandomInit`, :class:`UCBV`, :class:`UCBVtuned`, :class:`UCBH`, :class:`CPUCB`, :class:`UCBimproved`,

- Index based MOSS algorithms: :class:`MOSS`, :class:`MOSSH`, :class:`MOSSAnytime`, :class:`MOSSExperimental`,

- Bayesian algorithms: :class:`Thompson`, :class:`BayesUCB`, and :class:`DiscountedThompson`,

- Based on Kullback-Leibler divergence: :class:`klUCB`, :class:`klUCBloglog`, :class:`klUCBPlus`, :class:`klUCBH`, :class:`klUCBHPlus`, :class:`klUCBPlusPlus`, :class:`klUCBswitch`,

- Other index algorithms: :class:`DMED`, :class:`DMED.DMEDPlus`, :class:`IMED`, :class:`OCUCBH`, :class:`OCUCBH.AOCUCBH`, :class:`OCUCB`, :class:`UCBdagger`,

- Hybrids algorithms, mixing Bayesian and UCB indexes: :class:`AdBandits`,

- Aggregation algorithms: :class:`Aggregator` (mine, it's awesome, go on try it!), and :class:`CORRAL`, :class:`LearnExp`,

- Finite-Horizon Gittins index, approximated version: :class:`ApproximatedFHGittins`,

- An experimental policy, using a sliding window of for instance 100 draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible), :class:`SlidingWindowRestart`, and 3 versions for UCB, UCBalpha and klUCB: :class:`SlidingWindowRestart.SWR_UCB`, :class:`SlidingWindowRestart.SWR_UCBalpha`, :class:`SlidingWindowRestart.SWR_klUCB` (my algorithm, unpublished yet),

- An experimental policy, using just a sliding window of for instance 100 draws, :class:`SlidingWindowUCB.SWUCB`, and :class:`SlidingWindowUCB.SWUCBPlus` if the horizon is known. There is also :class:`SlidingWindowUCB.SWklUCB`.

- Another experimental policy with a discount factor, :class:`DiscountedUCB` and :class:`DiscountedUCB.DiscountedUCBPlus`, as well as versions using klUCB, :class:`DiscountedUCB.DiscountedklUCB`, and :class:`DiscountedUCB.DiscountedklUCBPlus`.

- Other policies for the non-stationary problems: :class:`LM_DSEE`, :class:`SWHash_UCB.SWHash_IndexPolicy`, :class:`CD_UCB.CUSUM_IndexPolicy`, :class:`CD_UCB.PHT_IndexPolicy`, :class:`CD_UCB.UCBLCB_IndexPolicy`, :class:`CD_UCB.GaussianGLR_IndexPolicy`, :class:`CD_UCB.BernoulliGLR_IndexPolicy`, :class:`Monitored_UCB.Monitored_IndexPolicy`, :class:`OracleSequentiallyRestartPolicy`, :class:`AdSwitch`.

- A policy designed to tackle sparse stochastic bandit problems, :class:`SparseUCB`, :class:`SparseklUCB`, and :class:`SparseWrapper` that can be used with *any* index policy.

- A policy that implements a "smart doubling trick" to turn any horizon-dependent policy into a horizon-independent policy without loosing in performances: :class:`DoublingTrickWrapper`,

- An *experimental* policy, implementing a another kind of doubling trick to turn any policy that needs to know the range :math:`[a,b]` of rewards a policy that don't need to know the range, and that adapt dynamically from the new observations, :class:`WrapRange`,

- The *Optimal Sampling for Structured Bandits* (OSSB) policy: :class:`OSSB` (it is more generic and can be applied to almost any kind of bandit problem, it works fine for classical stationary bandits but it is not optimal), a variant for gaussian problem :class:`GaussianOSSB`, and a variant for sparse bandits :class:`SparseOSSB`. There is also two variants with decreasing rates, :class:`OSSB_DecreasingRate` and :class:`OSSB_AutoDecreasingRate`,

- The Best Empirical Sampled Average (BESA) policy: :class:`BESA` (it works crazily well),

- **New!** The UCBoost (Upper Confidence bounds with Boosting) policies, first with no boosting: :class:`UCBoost.UCB_sq`, :class:`UCBoost.UCB_bq`, :class:`UCBoost.UCB_h`, :class:`UCBoost.UCB_lb`, :class:`UCBoost.UCB_t`, and then the ones with non-adaptive boosting: :class:`UCBoost.UCBoost_bq_h_lb`, :class:`UCBoost.UCBoost_bq_h_lb_t`, :class:`UCBoost.UCBoost_bq_h_lb_t_sq`, :class:`UCBoost.UCBoost`, and finally the epsilon-approximation boosting with :class:`UCBoost.UCBoostEpsilon`,


- Some are designed only for (fully decentralized) multi-player games: :class:`MusicalChair`, :class:`MEGA`, :class:`TrekkingTSN`, :class:`MusicalChairNoSensing`, :class:`SIC_MMAB`...

.. note::

    The list above might not be complete, see the details below.


All policies have the same interface, as described in :class:`BasePolicy`,
in order to use them in any experiment with the following approach: ::

    my_policy = Policy(nbArms)
    my_policy.startGame()  # start the game
    for t in range(T):
        chosen_arm_t = k_t = my_policy.choice()  # chose one arm
        reward_t     = sampled from an arm k_t   # sample a reward
        my_policy.getReward(k_t, reward_t)       # give it the the policy
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from .BasePolicy import BasePolicy
from .BaseWrapperPolicy import BaseWrapperPolicy

from .Posterior import Beta, Gamma, Gauss, DiscountedBeta

# --- Mine, uniform ones or fixed arm / fixed subset ones
from .Uniform import Uniform
from .UniformOnSome import UniformOnSome
from .TakeFixedArm import TakeFixedArm
from .TakeRandomFixedArm import TakeRandomFixedArm

# --- Naive or less naive epsilon-greedy policies
from .EpsilonGreedy import EpsilonGreedy, EpsilonFirst, EpsilonDecreasing, EpsilonDecreasingMEGA, EpsilonExpDecreasing
# --- Mine, simple exploratory policies
from .EmpiricalMeans import EmpiricalMeans

# --- Variants on EpsilonFirst, Explore-Then-Commit from E.Kaufmann's slides at IEEE ICC 2017
from .ExploreThenCommit import ETC_KnownGap, ETC_RandomStop, ETC_FixedBudget, ETC_SPRT, ETC_BAI, DeltaUCB

# --- Mine, Softmax and Exp3 policies
from .Softmax import Softmax, SoftmaxDecreasing, SoftMix, SoftmaxWithHorizon
from .Exp3 import Exp3, Exp3Decreasing, Exp3SoftMix, Exp3WithHorizon, Exp3ELM
from .Exp3PlusPlus import Exp3PlusPlus
from .ProbabilityPursuit import ProbabilityPursuit
from .BoltzmannGumbel import BoltzmannGumbel
from .Hedge import Hedge, HedgeDecreasing, HedgeWithHorizon

from .TsallisInf import TsallisInf

# --- Simple UCB policies
from .UCB import UCB
from .UCBH import UCBH          # With log(T) instead of log(t)
from .UCBalpha import UCBalpha  # Different indexes
from .UCBmin import UCBmin      # Different indexes
from .UCBplus import UCBplus    # Different indexes
from .UCBrandomInit import UCBrandomInit

# --- UCB with successive eliminations
from .UCBimproved import UCBimproved          # Different indexes

# --- UCB policies with variance terms
from .UCBV import UCBV          # Different indexes
from .UCBVtuned import UCBVtuned  # Different indexes

from .RCB import RCB  # Randomized Confidence Bounds
from .PHE import PHE  # Perturbed-History Exploration

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
from .DiscountedThompson import DiscountedThompson

# --- Bayesian index policy
from .BayesUCB import BayesUCB

# --- Kullback-Leibler based index policy
from .klUCB import klUCB
from .klUCBloglog import klUCBloglog  # With log(t) + c log(log(t)) and c = 1 (variable)
from .klUCB_forGLR import klUCB_forGLR  # With f(t - tau_i(t)) different for each arm
from .klUCBloglog_forGLR import klUCBloglog_forGLR
from .klUCBPlus import klUCBPlus    # Different indexes
from .klUCBH import klUCBH          # Knowing the horizon
from .klUCBHPlus import klUCBHPlus  # Different indexes
from .klUCBPlusPlus import klUCBPlusPlus  # Different indexes
from .klUCBswitch import klUCBswitch, klUCBswitchAnytime  # Different indexes

# From [Honda & Takemura, COLT 2010]
from .DMED import DMED, DMEDPlus

# From [Honda & Takemura, JMLR 2015]
from .IMED import IMED

# From [Lattimore, 2015]
from .OCUCBH import OCUCBH, AOCUCBH
# From [Lattimore, 2016]
from .OCUCB import OCUCB

# From [Lattimore, 2017]
from .UCBdagger import UCBdagger

# From [Combes et al, 2017] and my own work on the OSSB algorithm
from .OSSB import OSSB, GaussianOSSB, SparseOSSB, OSSB_DecreasingRate, OSSB_AutoDecreasingRate

# From [Baransi et al, 2014]
from .BESA import BESA

# From [Fang Liu et al, 2018]
from .UCBoost import UCB_sq, UCB_bq, UCB_h, UCB_lb, UCB_t, UCBoost_bq_h_lb, UCBoost_bq_h_lb_t, UCBoost_bq_h_lb_t_sq, UCBoost, UCBoostEpsilon

# From https://github.com/flaviotruzzi/AdBandits/
from .AdBandits import AdBandits

# --- Mine, aggregation algorithm, like Exp4
from .Aggregator import Aggregator
# --- Others aggregation algorithms
from .CORRAL import CORRAL
from .LearnExp import LearnExp
from .GenericAggregation import GenericAggregation

# --- Gittins index policy
from .ApproximatedFHGittins import ApproximatedFHGittins  # Approximated Finite-Horizon Gittins index

# --- Smart policies trying to adapt to dynamically changing environments
from .SlidingWindowUCB import SWUCB, SWUCBPlus, SWklUCB, SWklUCBPlus
from .DiscountedUCB import DiscountedUCB, DiscountedUCBPlus, DiscountedklUCB, DiscountedklUCBPlus

from .SlidingWindowRestart import SlidingWindowRestart, SWR_UCB, SWR_UCBalpha, SWR_klUCB

from .LM_DSEE import LM_DSEE
from .SWHash_UCB import SWHash_IndexPolicy
from .CD_UCB import SlidingWindowRestart_IndexPolicy, UCBLCB_IndexPolicy
from .CUSUM_UCB import CUSUM_IndexPolicy, PHT_IndexPolicy
from .GLR_UCB import GLR_IndexPolicy, GLR_IndexPolicy_WithTracking, GLR_IndexPolicy_WithDeterministicExploration, GaussianGLR_IndexPolicy, BernoulliGLR_IndexPolicy, OurGaussianGLR_IndexPolicy, GaussianGLR_IndexPolicy_WithTracking, BernoulliGLR_IndexPolicy_WithTracking, OurGaussianGLR_IndexPolicy_WithTracking, GaussianGLR_IndexPolicy_WithDeterministicExploration, BernoulliGLR_IndexPolicy_WithDeterministicExploration, OurGaussianGLR_IndexPolicy_WithDeterministicExploration, SubGaussianGLR_IndexPolicy

from .Exp3R import Exp3R, Exp3RPlusPlus
from .Exp3S import Exp3S
from .Monitored_UCB import Monitored_IndexPolicy
from .OracleSequentiallyRestartPolicy import OracleSequentiallyRestartPolicy
from .AdSwitch import AdSwitch
from .AdSwitchNew import AdSwitchNew

from .DoublingTrickWrapper import DoublingTrickWrapper, next_horizon__arithmetic, next_horizon__geometric, next_horizon__exponential, next_horizon__exponential_fast, next_horizon__exponential_slow, next_horizon__exponential_generic, breakpoints, Ti_geometric, Ti_exponential, Ti_intermediate_sqrti, Ti_intermediate_i13, Ti_intermediate_i23, Ti_intermediate_i12_logi12, Ti_intermediate_i_by_logi

from .WrapRange import WrapRange

# --- Mine, implemented from state-of-the-art papers on multi-player policies

from .MusicalChair import MusicalChair, optimalT0  # Cf. [Shamir et al., 2015](https://arxiv.org/abs/1512.02866)
from .MusicalChairNoSensing import MusicalChairNoSensing  # Cf. [Lugosi and Mehrabian, 2018](https://arxiv.org/abs/1808.08416)
from .SIC_MMAB import SIC_MMAB, SIC_MMAB_UCB, SIC_MMAB_klUCB  # Cf. [Boursier and Perchet, 2018](https://arxiv.org/abs/1809.08151)

from .TrekkingTSN import TrekkingTSN  # Cf. [R.Kumar, A.Yadav, S.J.Darak, M.K.Hanawal, Trekking based Distributed Algorithm for Opportunistic Spectrum Access in Infrastructure-less Network, 2018](https://ieeexplore.ieee.org/abstract/document/8362858/)

from .MEGA import MEGA  # Cf. [Avner & Mannor, 2014](https://arxiv.org/abs/1404.5421)

# --- Rotting bandits
from .SWA import SWA, wSWA
from .FEWA import FEWA, EFF_FEWA
from .RAWUCB import RAWUCB, EFF_RAWUCB, EFF_RAWUCB_asymptotic, EFF_RAWklUCB
from .GreedyOracle import GreedyOracle, GreedyPolicy

# --- Utility functions

from .with_proba import with_proba

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

# # Experimentals policies
# try:
#     from .Experimentals import *
# except ImportError as e:
#     from traceback import print_exc
#     print("Warning: not able to import some policies from Experimentals subpackage.\nError was: {}...".format(e))  # DEBUG
#     print_exc()  # DEBUG
#     del print_exc
