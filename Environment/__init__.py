# -*- coding: utf-8 -*-
""" Environment module :

- :class:`MAB`, :class:`MarkovianMAB`, :class:`DynamicMAB` and :class:`IncreasingMAB` objects, used to wrap the problems (list of arms).
- :class:`Result` and :class:`ResultMultiPlayers` objects, used to wrap simulation results (list of decisions and rewards).
- :class:`Evaluator` environment, used to wrap simulation, for the single player case.
- :class:`EvaluatorMultiPlayers` environment, used to wrap simulation, for the multi-players case.
- :class:`EvaluatorSparseMultiPlayers` environment, used to wrap simulation, for the multi-players case with sparse activated players.
- :mod:`CollisionModels` implements different collision models.

And useful constants and functions for the plotting and stuff:

- :func:`DPI`, :func:`signature`, :func:`maximizeWindow`, :func:`palette`, :func:`makemarkers`, :func:`wraptext`: for plotting
- :func:`notify`: send a desktop notification
- :func:`Parallel`, :func:`delayed`: joblib related
- :func:`tqdm`: pretty range() loops
- :func:`sortedDistance`, :func:`fairnessMeasures`: science related
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from .MAB import MAB, MarkovianMAB, DynamicMAB, IncreasingMAB

from .Result import Result
from .Evaluator import Evaluator

from .CollisionModels import *
from .ResultMultiPlayers import ResultMultiPlayers
from .EvaluatorMultiPlayers import EvaluatorMultiPlayers
from .EvaluatorSparseMultiPlayers import EvaluatorSparseMultiPlayers

from .plotsettings import DPI, signature, maximizeWindow, palette, makemarkers, wraptext

from .notify import notify

from .usejoblib import *
from .usetqdm import *

from .sortedDistance import *
from .fairnessMeasures import *
