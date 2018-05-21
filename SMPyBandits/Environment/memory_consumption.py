# -*- coding: utf-8 -*-
""" Tiny module that defines a utility function to get the memory consumes in a thread, and the function :func:`sizeof_fmt`."""

from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


try:
    import resource
except ImportError as e:
    print("ERROR: 'resource' module not available, but it is in the standard library.\nHave you messed up your Python installation?\nPlease submit a new bug on https://github.com/SMPyBandits/SMPyBandits/issues/new")  # DEBUG
    raise e


def getCurrentMemory(thread=False):
    """ Get the current memory consumption of the process, or the thread.

    - Example, before and after creating a huge random matrix in Numpy, and asking to invert it:

    >>> currentMemory = getCurrentMemory()
    >>> print("Consumed {} memory".format(sizeof_fmt(currentMemory)))  # doctest: +SKIP
    Consumed 16.8 KiB memory

    >>> import numpy as np; x = np.random.randn(1000, 1000)  # doctest: +SKIP
    >>> diffMemory = getCurrentMemory() - currentMemory; currentMemory += diffMemory
    >>> print("Consumed {} more memory".format(sizeof_fmt(diffMemory)))  # doctest: +SKIP
    Consumed 18.8 KiB more memory

    >>> y = np.linalg.pinv(x)  # doctest: +SKIP
    >>> diffMemory = getCurrentMemory() - currentMemory; currentMemory += diffMemory
    >>> print("Consumed {} more memory".format(sizeof_fmt(diffMemory)))  # doctest: +SKIP
    Consumed 63.9 KiB more memory
    """
    if thread:
        try:
            return resource.getrusage(resource.RUSAGE_THREAD).ru_maxrss
        except ValueError:
            print("Warning: resource.RUSAGE_THREAD is not available.")  # DEBUG
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def sizeof_fmt(num, suffix='B', longsuffix=True, usespace=True, base=1024):
    """ Returns a string representation of the size ``num``.

    - Examples:
    >>> sizeof_fmt(1020)
    '1020 B'
    >>> sizeof_fmt(1024)
    '1 KiB'
    >>> sizeof_fmt(12011993)
    '11.5 MiB'
    >>> sizeof_fmt(123456789)
    '117.7 MiB'
    >>> sizeof_fmt(123456789911)
    '115 GiB'

    Options include:

    - No space before unit:
    >>> sizeof_fmt(123456789911, usespace=False)
    '115GiB'

    - French style, with short suffix, the "O" suffix for "octets", and a base 1000:
    >>> sizeof_fmt(123456789911, longsuffix=False, suffix='O', base=1000)
    '123.5 GO'

    - Reference: https://stackoverflow.com/a/1094933/5889533
    """
    num = float(num)  # force typecast
    base = float(base)

    suffixes = ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    if longsuffix:
        suffixes = ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi']

    for unit in suffixes[:-1]:
        if abs(num) < base:
            return "{num:3.1f}{space}{unit}{suffix}".format(
                num=num,
                space=' ' if usespace else '',
                unit=unit,
                suffix=suffix,
            ).replace(".0", "")
        num /= base
    return "{num:.1f}{space}{unit}{suffix}".format(
        num=num,
        space=' ' if usespace else '',
        unit=suffixes[-1],
        suffix=suffix,
    ).replace(".0", "")


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
