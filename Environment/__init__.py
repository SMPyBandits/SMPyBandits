# -*- coding: utf-8 -*-
""" Environment :

- MAB environment, used to wrap the problems (list of arms).
- Evaluator environment, used to wrap simulation.
- Result environment, used to wrap simulation results (list of decisions and rewards).
"""

__author__ = "Lilian Besson, Emilie Kaufmann"
__version__ = "0.1"

from .Evaluator import Evaluator
from .MAB import MAB
from .Result import Result
