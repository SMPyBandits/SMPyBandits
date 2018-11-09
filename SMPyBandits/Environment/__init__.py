# -*- coding: utf-8 -*-
""" ``Environment`` module:

- :class:`MAB`, :class:`MarkovianMAB`, :class:`ChangingAtEachRepMAB`, :class:`IncreasingMAB`, :class:`PieceWiseStationaryMAB`, :class:`NonStationaryMAB` objects, used to wrap the problems (essentially a list of arms).
- :class:`Result` and :class:`ResultMultiPlayers` objects, used to wrap simulation results (list of decisions and rewards).
- :class:`Evaluator` environment, used to wrap simulation, for the single player case.
- :class:`EvaluatorMultiPlayers` environment, used to wrap simulation, for the multi-players case.
- :class:`EvaluatorSparseMultiPlayers` environment, used to wrap simulation, for the multi-players case with sparse activated players.
- :mod:`CollisionModels` implements different collision models.

And useful constants and functions for the plotting and stuff:

- :data:`DPI`, :func:`signature`, :func:`maximizeWindow`, :func:`palette`, :func:`makemarkers`, :func:`wraptext`: for plotting,
- :func:`notify`: send a desktop notification,
- :func:`Parallel`, :func:`delayed`: joblib related,
- :mod:`tqdm`: pretty range() loops,
- :mod:`sortedDistance`, :mod:`fairnessMeasures`: science related,
- :func:`getCurrentMemory`, :func:`sizeof_fmt`: to measure and pretty print memory consumption.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from .MAB import MAB, MarkovianMAB, ChangingAtEachRepMAB, IncreasingMAB, PieceWiseStationaryMAB, NonStationaryMAB

from .Result import Result
from .Evaluator import Evaluator

from .CollisionModels import *
from .ResultMultiPlayers import ResultMultiPlayers
from .EvaluatorMultiPlayers import EvaluatorMultiPlayers
from .EvaluatorSparseMultiPlayers import EvaluatorSparseMultiPlayers

from .plotsettings import DPI, signature, maximizeWindow, palette, makemarkers, wraptext

from .notify import notify

from .usejoblib import USE_JOBLIB, Parallel, delayed
from .usetqdm import USE_TQDM, tqdm

from .sortedDistance import weightedDistance, manhattan, kendalltau, spearmanr, gestalt, meanDistance, sortedDistance
from .fairnessMeasures import amplitude_fairness, std_fairness, rajjain_fairness, mo_walrand_fairness, mean_fairness, fairnessMeasure, fairness_mapping

from .memory_consumption import getCurrentMemory, sizeof_fmt, start_tracemalloc, display_top_tracemalloc
