# -*- coding: utf-8 -*-
""" Import numba.jit or a dummy decorator.
"""
from __future__ import division, print_function

__author__ = "Lilian Besson"
__version__ = "0.5"

# Configure the use of numba
USE_NUMBA = False
USE_NUMBA = True   # XXX Experimental

# DONE I tried numba.jit() on these functions, and it DOES not give any speedup...:-( sad sad !
try:
    from numba.decorators import jit
    # from numba.decorators import jit as numbajit
    import locale  # See this bug, http://numba.pydata.org/numba-doc/dev/user/faq.html#llvm-locale-bug
    locale.setlocale(locale.LC_NUMERIC, 'C')
    print("Info: numba seems to be available.")
except ImportError:
    print("Warning: numba seems to not be available. Using a dummy decorator for numba.jit() ...")
    USE_NUMBA = False

if not USE_NUMBA:
    print("Warning: numba seems to be disabled. Using a dummy decorator for numba.jit() ...")
    def jit(f):
        return f
