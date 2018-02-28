# -*- coding: utf-8 -*-
""" Import tqdm from tqdm, safely.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


def in_notebook():
    """Check if the code is running inside a Jupyter notebook or not. Cf. http://stackoverflow.com/a/39662359/.

    >>> in_notebook()
    False
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole?
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal running IPython?
            return False
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


try:
    if in_notebook():
        from tqdm import tqdm_notebook as tqdm
        print("Info: Using the Jupyter notebook version of the tqdm() decorator, tqdm_notebook() ...")  # DEBUG
    else:
        from tqdm import tqdm
        # print("Info: Using the regular tqdm() decorator ...")  # DEBUG
    USE_TQDM = True
except ImportError:
    print("Warning: tqdm not found. Install it from pypi ('pip install tqdm') or conda.\n  Info: Not mandatory, but it's pretty!")
    USE_TQDM = False

    def tqdm(iterator, *args, **kwargs):
        """Fake tqdm.tqdm wrapper, ignore **kwargs like desc='...', and return iterator."""
        return iterator


# Only export and expose the useful functions defined here
__all__ = [
    "USE_TQDM",
    "tqdm",
]
