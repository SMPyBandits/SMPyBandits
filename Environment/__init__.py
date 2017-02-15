# -*- coding: utf-8 -*-
""" Environment :

- MAB environment, used to wrap the problems (list of arms).
- Result environment, used to wrap simulation results (list of decisions and rewards).
- Evaluator environment, used to wrap simulation, for the single player case.
- Evaluator environment, used to wrap simulation, for the multi-players case.
"""

__author__ = "Lilian Besson"
__version__ = "0.5"

from .MAB import MAB

from .Result import Result
from .ResultMultiPlayers import ResultMultiPlayers

from .Evaluator import Evaluator
from .EvaluatorMultiPlayers import EvaluatorMultiPlayers
# from .CollisionModels import *

from .plotsettings import DPI, signature, maximizeWindow, palette, makemarkers, wraptext

from .notify import notify
