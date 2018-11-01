# -*- coding: utf-8 -*-
""" Posteriors for Bayesian Index policies:

- :class:`Beta` is the default for :class:`Thompson` Sampling and :class:`BayesUCB`, ideal for Bernoulli experiments,
- :class:`Gamma` and :class:`Gauss` are more suited for respectively Poisson and Gaussian arms,
- :class:`DiscountedBeta` is the default for :class:`Policies.DiscountedThompson` Sampling, ideal for Bernoulli experiments on non stationary bandits.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

# from .Posterior import Posterior

from .Beta import Beta
from .DiscountedBeta import DiscountedBeta
from .Gamma import Gamma
from .Gauss import Gauss
