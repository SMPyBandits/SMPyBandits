# -*- coding: utf-8 -*-
""" Policies : contains various bandits algorithms:
Dummy, UCB, Thompson, BayesUCB, klUCB, Aggr, AdBandit.
"""

__author__ = "Lilian Besson, Emilie Kaufmann"
__version__ = "0.1"

from .Dummy import Dummy
from .UCB import UCB
from .Thompson import Thompson
from .BayesUCB import BayesUCB
from .klUCB import klUCB
from .Aggr import Aggr
from .AdBandits import AdBandit
