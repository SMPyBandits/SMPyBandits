#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Empty __init__.py to define the package as a package and its version.
"""
# from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9.2"

try:
    # from .Arms import *
    from SMPyBandits import Arms
except ImportError:
    pass

try:
    # from .Environment import *
    from SMPyBandits import Environment
except ImportError:
    pass

try:
    # from .Policies import *
    from SMPyBandits import Policies
except ImportError:
    pass

# try:
#     # from .Policies.Posterior import *
#     from SMPyBandits.Policies import Posterior
# except ImportError:
#     pass

try:
    # from .PoliciesMultiPlayers import *
    from SMPyBandits import PoliciesMultiPlayers
except ImportError:
    pass
