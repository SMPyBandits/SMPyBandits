# -*- coding: utf-8 -*-
""" Posteriors for Bayesian Index policies:

- :class:`Beta` is the default for :class:`Thomspon` Sampling and :class:`BayesUCB`, ideal for Bernoulli experiments,
- :class:`Gamma` and :class:`Gauss` are more suited for respectively Poisson and Gaussian arms.
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

# from .Posterior import Posterior

from .Beta import Beta
from .Gamma import Gamma
from .Gauss import Gauss
