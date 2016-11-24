# -*- coding: utf-8 -*-
""" PoliciesMultiPlayers : contains various bandits algorithms:
Dummy.
"""
# Dummy, EpsilonGreedy, EpsilonFirst, EpsilonDecreasing, Softmax, UCB, UCBV, Thompson, BayesUCB, klUCB, KLempUCB, Aggr, AdBandit.

__author__ = "Lilian Besson"
__version__ = "0.1"

# Mine, stupid ones
from .Dummy import Dummy

# from .EpsilonGreedy import EpsilonGreedy
# from .EpsilonFirst import EpsilonFirst
# from .EpsilonDecreasing import EpsilonDecreasing
# from .Softmax import Softmax

# # From pymaBandits v1.0
# from .UCB import UCB
# from .UCBalpha import UCBalpha
# from .UCBV import UCBV
# from .Thompson import Thompson
# from .BayesUCB import BayesUCB
# from .klUCB import klUCB
# from .KLempUCB import KLempUCB

# # From https://github.com/flaviotruzzi/AdBandits/
# from .AdBandits import AdBandit

# # Mine, aggregated ones
# from .Aggr import Aggr
