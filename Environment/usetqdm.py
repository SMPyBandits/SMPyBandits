# -*- coding: utf-8 -*-
""" Import tqdm from tqdm, safely.
"""
from __future__ import division, print_function

__author__ = "Lilian Besson"
__version__ = "0.6"

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    print("tqdm not found. Install it from pypi ('pip install tqdm') or conda.\n  Info: Not mandatory, but it's pretty!")
    USE_TQDM = False

    def tqdm(iterator, *args, **kwargs):
        """Fake tqdm.tqdm wrapper, ignore **kwargs like desc='...', and return iterator."""
        return iterator


# Only export and expose the useful functions defined here
__all__ = [USE_TQDM, tqdm]
