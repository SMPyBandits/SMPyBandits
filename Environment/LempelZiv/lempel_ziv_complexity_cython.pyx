# %%cython
"""Lempel-Ziv complexity for a binary sequence, in simple Cython code (C extension).

- How to build it? Simply use the file :download:`Makefile` provided in this folder.

- How to use it? From Python, it's easy:

>>> from lempel_ziv_complexity_cython import lempel_ziv_complexity
>>> s = '1001111011000010'
>>> lempel_ziv_complexity(s)  # 1 / 0 / 01 / 1110 / 1100 / 0010
6

- Requirements: you need to have [Cython](http://Cython.org/) installed, and use [CPython](https://www.Python.org/).
"""

__author__ = "Lilian Besson"
__version__ = "0.6"


import cython

# Define the type of unsigned int32
ctypedef unsigned int DTYPE_t


# turn off bounds-checking for entire function, quicker but less safe
@cython.boundscheck(False)
def lempel_ziv_complexity(str binary_sequence):
    """Lempel-Ziv complexity for a binary sequence, in simple Cython code (C extension).

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
    - Note: there is also a naive Python version, for speedup, see :download:`lempel_ziv_complexity.py`.
    """
    cdef DTYPE_t u = 0
    cdef DTYPE_t v = 1
    cdef DTYPE_t w = 1
    cdef DTYPE_t v_max = 1
    cdef DTYPE_t length = len(binary_sequence)
    cdef DTYPE_t complexity = 1
    # that was the only needed part, typing statically all the variables
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
