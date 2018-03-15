# -*- coding: utf-8 -*-
""" Import numba.jit or a dummy decorator.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

#: Configure the use of numba
USE_NUMBA = False
USE_NUMBA = True   # XXX Experimental

if not USE_NUMBA:
    print("Warning: numba.jit seems to be disabled. Using a dummy decorator for numba.jit() ...")  # DEBUG

# DONE I tried numba.jit() on these functions, and it DOES not give any speedup...:-( sad sad !
try:
    from numba.decorators import jit
    import locale  # See this bug, http://numba.pydata.org/numba-doc/dev/user/faq.html#llvm-locale-bug
    locale.setlocale(locale.LC_NUMERIC, 'C')
    # print("Info: numba.jit seems to be available.")  # DEBUG
except ImportError:
    print("Warning: numba.jit seems to not be available. Using a dummy decorator for numba.jit() ...\nIf you want the speed up brought by numba.jit, try to manually install numba and check that it works (installing llvmlite can be tricky, cf. https://github.com/numba/numba#custom-python-environments")  # DEBUG
    USE_NUMBA = False

if not USE_NUMBA:
    from functools import wraps

    def jit(f):
        """Fake numba.jit decorator."""
        return f  # XXX isn't it enough?!
        # @wraps(f)
        # def wrapper(*args, **kwargs):
        #     """Fake docstring, shouldn't be used thanks to wraps."""
        #     return f(*args, **kwargs)
        # return wrapper


# Only export and expose the useful functions defined here
__all__ = ["USE_NUMBA", "jit"]
