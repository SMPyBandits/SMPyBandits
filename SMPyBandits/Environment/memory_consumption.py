# -*- coding: utf-8 -*-
""" Tiny module to measure and work on memory consumption.

It defines a utility function to get the memory consumes in the current process or the current thread (:func:`getCurrentMemory`), and a function to pretty print memory size (:func:`sizeof_fmt`).

It also imports :mod:`tracemalloc` and define a convenient function that pretty print the most costly lines after a run.

- Reference: https://docs.python.org/3/library/tracemalloc.html#pretty-top
- Example:

>>> return_code = start_tracemalloc()
Starting to trace memory allocations...
>>> # ... run your application ...
>>> display_top_tracemalloc()
<BLANKLINE>
Top 10 lines ranked by memory consumption:
#1: python3.6/doctest.py:1330: 636 B
    compileflags, 1), test.globs)
#2: <doctest __main__[1]>:1: 568 B
    display_top_tracemalloc()
#3: python3.6/doctest.py:1346: 472 B
    if check(example.want, got, self.optionflags):
#4: python3.6/doctest.py:1374: 464 B
    self.report_success(out, test, example, got)
#5: python3.6/doctest.py:1591: 456 B
    got = self._toAscii(got)
#6: ./memory_consumption.py:168: 448 B
    snapshot = tracemalloc.take_snapshot()
#7: python3.6/doctest.py:1340: 440 B
    self._fakeout.truncate(0)
#8: python3.6/doctest.py:1339: 440 B
    got = self._fakeout.getvalue()  # the actual output
#9: python3.6/doctest.py:1331: 432 B
    self.debugger.set_continue() # ==== Example Finished ====
#10: python3.6/doctest.py:251: 89 B
    result = StringIO.getvalue(self)
2 others: 78 B
<BLANKLINE>
Total allocated size: 4.4 KiB
4523

.. warning:: This is automatically used (for ``main.py`` at least) when ``DEBUGMEMORY=True`` (cli env).

.. warning:: This is experimental and does *not* work as well on Mac OS X and Windows as it works on GNU/Linux systems.
"""

from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import os
from linecache import getline

HAS_resource = False
try:
    import resource
    HAS_resource = True
except ImportError as e:
    print("ERROR: 'resource' module not available, but it is in the standard library.\nHave you messed up your Python installation?\nAre you on Windows? In this case, it's okay.\nPlease submit a new bug on https://github.com/SMPyBandits/SMPyBandits/issues/new")  # DEBUG


def getCurrentMemory(thread=False, both=False):
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

    .. warning:: This is still experimental for multi-threaded code.
    .. warning:: It can break on some systems, see for instance [the issue #142](https://github.com/SMPyBandits/SMPyBandits/issues/142).

    .. warning:: FIXME even on my own system, it works for the *last* few policies I test, but fails for the first??

    .. warning:: This returns 0 on Microsoft Windows, because the :mod:`resource` module is not available on non-UNIX systems (see https://docs.python.org/3/library/unix.html).
    """
    if not HAS_resource:
        return 0
    if thread:
        try:
            # https://docs.python.org/3/library/resource.html#resource.RUSAGE_THREAD
            return resource.getrusage(resource.RUSAGE_THREAD).ru_maxrss
        except (ValueError, AttributeError):
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


#: Max number of lines to show with :func:`display_top_tracemalloc`.
LIMIT = 10

try:
    import tracemalloc

    def start_tracemalloc():
        """Wrapper function around :func:`tracemalloc.start`, to log the start of tracing memory allocation."""
        tracemalloc.start()
        print("Starting to trace memory allocations...")  # DEBUG
        return 0

    def display_top_tracemalloc(snapshot=None, key_type='lineno', limit=LIMIT):
        """ Display detailed information on the ``limit`` most costly lines in this memory snapshot.
        """
        if snapshot is None:
            snapshot = tracemalloc.take_snapshot()

        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
            tracemalloc.Filter(False, "<unknown>"),
            # XXX add my own filter if needed!
        ))
        top_stats = snapshot.statistics(key_type)

        print("\nTop {} lines ranked by memory consumption:".format(limit))
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback[0]
            # replace "/path/to/module/file.py" with "module/file.py"
            filename = os.sep.join(frame.filename.split(os.sep)[-2:])
            print("#{index}: {filename}:{lineno}: {size}".format(
                index=index,
                filename=filename,
                lineno=frame.lineno,
                size=sizeof_fmt(stat.size)
            ))
            line = getline(frame.filename, frame.lineno).strip()
            if line:
                print("    {}".format(line))

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            print("{other} others: {size}".format(other=len(other), size=sizeof_fmt(size)))

        total = sum(stat.size for stat in top_stats)
        print("\nTotal allocated size: {size}".format(size=sizeof_fmt(total)))
        return total

# ----------------------------------------------------------------------------
except ImportError:
    print("'tracemalloc' module not available. Are you on Python 2?\nYou should use Python 3 for this feature.")  # WARNING

    def start_tracemalloc():
        """Fake :func:`start_tracemalloc` function, if :mod:`tracemalloc` is not available."""
        print("Warning: tracemalloc module is not available, no information can be printed about memory consumption.")  # DEBUG
        return -1

    def display_top_tracemalloc(snapshot=None, key_type='lineno', limit=LIMIT):
        """Fake :func:`display_top_tracemalloc` function, if :mod:`tracemalloc` is not available."""
        print("Warning: tracemalloc module is not available, no information can be printed about memory consumption.")  # DEBUG
        return -1


# Only export and expose the useful functions defined here
__all__ = [
    "getCurrentMemory",
    "sizeof_fmt",
    "start_tracemalloc",
    "display_top_tracemalloc",
]

# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
