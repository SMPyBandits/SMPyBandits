# -*- coding: utf-8 -*-
""" Import Parallel and delayed from joblib, safely.
"""
from __future__ import division, print_function

__author__ = "Lilian Besson"
__version__ = "0.6"

try:
    from joblib import Parallel, delayed
    USE_JOBLIB = True
except ImportError:
    print("joblib not found. Install it from pypi ('pip install joblib') or conda.\n  Info: Not mandatory, but it improves speed computation on multi-core machines.")
    USE_JOBLIB = False

    # In case the code uses Parallel and delayed, even if USE_JOBLIB is False
    def Parallel(*args, **kwargs):
        """Fake joblib.Parallel implementation."""
        def fakeParallelWrapper(iterator):
            """ Just a list(iterator)."""
            return list(iterator)
        return fakeParallelWrapper

    def delayed(f, *args, **kwargs):
        """Fake joblib.delayed implementation."""
        return f


# Only export and expose the useful functions defined here
__all__ = [
    "USE_JOBLIB",
    "Parallel",
    "delayed"
]
