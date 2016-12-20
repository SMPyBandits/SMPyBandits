# -*- coding: utf-8 -*-
""" Policies : contains various bandits algorithms:

- Stupid ones: Uniform, UniformOnSome, TakeFixedArm, TakeRandomFixedArm
- Greedy ones: EpsilonGreedy, EpsilonFirst, EpsilonDecreasing
- Probabilist ones: Softmax
- Index based: UCB, UCBalpha, UCBopt, UCBplus, UCBtuned, UCBrandomInit, UCBV
- Bayesian: Thompson, BayesUCB
- Based on Kullback-Leibler ones: klUCB, klUCBPlus, klUCBHPlus
- Hybrids: AdBandit
- Aggregated ones: Aggr
- Designed for multi-player games: MusicalChair, MEGA
"""

__author__ = "Lilian Besson"
__version__ = "0.3"

# --- Mine, stupid ones
from .Uniform import Uniform
from .UniformOnSome import UniformOnSome
from .TakeFixedArm import TakeFixedArm
from .TakeRandomFixedArm import TakeRandomFixedArm

# --- Mine, simple exploratory policies
from .EpsilonGreedy import EpsilonGreedy
from .EpsilonFirst import EpsilonFirst
from .EpsilonDecreasing import EpsilonDecreasing
from .EpsilonDecreasingMEGA import EpsilonDecreasingMEGA
from .EpsilonExpDecreasing import EpsilonExpDecreasing

# --- Mine, Exp3-like policies
from .Softmax import Softmax, SoftmaxDecreasing, SoftmaxWithHorizon

# --- From pymaBandits v1.0
from .UCB import UCB
from .UCBalpha import UCBalpha  # Different indexes
from .UCBopt import UCBopt      # Different indexes
from .UCBplus import UCBplus    # Different indexes
from .UCBtuned import UCBtuned  # Different indexes
from .UCBrandomInit import UCBrandomInit      # Different indexes
from .UCBV import UCBV          # Different indexes

from .MOSS import MOSS

from .Thompson import Thompson

from .BayesUCB import BayesUCB

from .klUCB import klUCB
from .klUCBPlus import klUCBPlus    # Different indexes
from .klUCBHPlus import klUCBHPlus  # Different indexes
# from .KLempUCB import KLempUCB  # XXX fix it before importing it

# From https://github.com/flaviotruzzi/AdBandits/
from .AdBandits import AdBandits

# --- Mine, aggregated ones
from .Aggr import Aggr

# --- Mine, implemented from state-of-the-art papers

from .MusicalChair import MusicalChair  # Cf. [Shamir et al., 2015](https://arxiv.org/abs/1512.02866)
# from .DynamicMusicalChair import DynamicMusicalChair  # FIXME write it! Can be just a subclass of MusicalChair

from .MEGA import MEGA  # Cf. [Avner & Mannor, 2014](https://arxiv.org/abs/1404.5421)
