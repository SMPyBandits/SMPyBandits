# -*- coding: utf-8 -*-
""" Environment :

- MAB environment, used to wrap the problems (list of arms).
- Result environment, used to wrap simulation results (list of decisions and rewards).
- Evaluator environment, used to wrap simulation, for the single player case.
- Evaluator environment, used to wrap simulation, for the multi-players case.
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

from .MAB import MAB, DynamicMAB

from .Result import Result
from .Evaluator import Evaluator

from .CollisionModels import *
from .ResultMultiPlayers import ResultMultiPlayers
from .EvaluatorMultiPlayers import EvaluatorMultiPlayers

from .plotsettings import DPI, signature, maximizeWindow, palette, makemarkers, wraptext

from .notify import notify

from .usejoblib import *
from .usetqdm import *

from .sortedDistance import *
from .fairnessMeasures import *
