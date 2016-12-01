# -*- coding: utf-8 -*-
""" Policies : contains various bandits algorithms:
Uniform, EpsilonGreedy, EpsilonFirst, EpsilonDecreasing, Softmax, UCB, UCBV, Thompson, BayesUCB, klUCB, KLempUCB, Aggr, AdBandit.
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

# Mine, stupid ones
from .Uniform import Uniform
from .UniformOnSome import UniformOnSome
from .TakeFixedArm import TakeFixedArm
from .TakeRandomFixedArm import TakeRandomFixedArm

# Mine, simple exploratory policies
from .EpsilonGreedy import EpsilonGreedy
from .EpsilonFirst import EpsilonFirst
from .EpsilonDecreasing import EpsilonDecreasing
from .Softmax import Softmax

# From pymaBandits v1.0
from .UCB import UCB
from .UCBalpha import UCBalpha
from .UCBV import UCBV
from .Thompson import Thompson
from .BayesUCB import BayesUCB
from .klUCB import klUCB
# from .KLempUCB import KLempUCB  # XXX fix it before importing it

# From https://github.com/flaviotruzzi/AdBandits/
from .AdBandits import AdBandit

# Mine, aggregated ones
from .Aggr import Aggr

# Mine, implemented from state-of-the-art papers
from .MusicalChair import MusicalChair  # Cf. [Shamir et al., 2015](https://arxiv.org/abs/1512.02866)
# from .DynamicMusicalChair import DynamicMusicalChair  # FIXME write it! Can be just a subclass of MusicalChair
