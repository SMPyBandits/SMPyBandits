# -*- coding: utf-8 -*-
"""Lempel-Ziv complexity for a binary sequence, in naive Python code.

- How to use it? From Python, it's easy:

>>> from lempel_ziv_complexity import lempel_ziv_complexity
>>> s = '1001111011000010'
>>> lempel_ziv_complexity(s)  # 1 / 0 / 01 / 1110 / 1100 / 0010
6

- From others folders:

>>> from LempelZiv import lempel_ziv_complexity

- Note: there is also a Cython-powered version, for speedup, see :download:`lempel_ziv_complexity_cython.pyx`.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.6"


#: Configure the use of numba
USE_NUMBA = True   # XXX Experimental
USE_NUMBA = False


# DONE I tried numba.jit() on these functions, and it DOES not give any speedup...:-( sad sad !
try:
    from numba.decorators import jit
    # from numba.decorators import jit as numbajit
    import locale  # See this bug, http://numba.pydata.org/numba-doc/dev/user/faq.html#llvm-locale-bug
    locale.setlocale(locale.LC_NUMERIC, 'C')
    print("Info: numba.jit seems to be available.")
except ImportError:
    print("Warning: numba.jit seems to not be available. Using a dummy decorator for numba.jit() ...")

    USE_NUMBA = False

if not USE_NUMBA:
    print("Warning: numba.jit seems to be disabled. Using a dummy decorator for numba.jit() ...")

    def jit(f):
        """Fake numba.jit decorator."""
        return f


# Can be numba, can be not numba, depending on USE_NUMBA
@jit
def lempel_ziv_complexity(binary_sequence):
    """ Manual implementation of the Lempel-Ziv complexity.

    It is defined as the number of different substrings encountered as the stream is viewed from begining to the end.
    As an example:

    >>> s = '1001111011000010'
    >>> lempel_ziv_complexity(s)  # 1 / 0 / 01 / 1110 / 1100 / 0010
    6

    Marking in the different substrings the sequence complexity :math:`\mathrm{Lempel-Ziv}(s) = 6`: :math:`s = 1 / 0 / 01 / 1110 / 1100 / 0010`.

    - See the page https://en.wikipedia.org/wiki/Lempel-Ziv_complexity for more details.


    Other examples:

    >>> lempel_ziv_complexity('1010101010101010')  # 1 / 0 / 10
    3
    >>> lempel_ziv_complexity('1001111011000010000010')  # 1 / 0 / 01 / 1110 / 1100 / 0010 / 000 / 010
    7
    >>> lempel_ziv_complexity('100111101100001000001010')  # 1 / 0 / 01 / 1110 / 1100 / 0010 / 000 / 010 / 10
    8

    - Note: it is faster to give the sequence as a string of characters, like `'10001001'`, instead of a list or a numpy array.
    - Note: see this notebook for more details, comparison, benchmarks and experiments: http://banditslilian.gforge.inria.fr/notebooks/Short_study_of_the_Lempel-Ziv_complexity.html
    - Note: there is also a Cython-powered version, for speedup, see :download:`lempel_ziv_complexity_cython.pyx`.
    """
    u, v, w = 0, 1, 1
    v_max = 1
    length = len(binary_sequence)
    complexity = 1
    while True:
        if binary_sequence[u + v - 1] == binary_sequence[w + v - 1]:
            v += 1
            if w + v >= length:
                complexity += 1
                break
        else:
            if v > v_max:
                v_max = v
            u += 1
            if u == w:
                complexity += 1
                w += v_max
                if w > length:
                    break
                else:
                    u = 0
                    v = 1
                    v_max = 1
            else:
                v = 1
    return complexity
