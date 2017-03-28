# -*- coding: utf-8 -*-
""" Environment :

- MAB and DynamicMAB objects, used to wrap the problems (list of arms).
- Result and ResultMultiPlayers objects, used to wrap simulation results (list of decisions and rewards).
- Evaluator environment, used to wrap simulation, for the single player case.
- EvaluatorMultiPlayers environment, used to wrap simulation, for the multi-players case.

And useful functions for the plotting and stuff:

- DPI, signature, maximizeWindow, palette, makemarkers, wraptext: for plotting
- notify: send a notificaiton
- Parallel, delayed: joblib related
- tqdm: pretty range() loops
- sortedDistance, fairnessMeasures: science related
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
