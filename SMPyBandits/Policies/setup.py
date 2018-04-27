# -*- coding: utf-8 -*-
"""
Basic setup.py to compile a Cython extension.
It is used to compile the ``kullback_cython`` extension, by running::

    $ python setup.py build_ext --inplace

You can also use [pyximport](http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html#pyximport-cython-compilation-for-developers) to import the ``kullback_cython`` module transparently:

>>> import pyximport; pyximport.install()
>>> import kullback_cython as kullback
>>> # then use kullback.klucbBern or others, as if they came from the pure Python version!
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("kullback_cython", ["kullback_cython.pyx"]),
    # XXX also build the extension with full name?
    # Extension("SMPyBandits.Policies.kullback_cython", ["kullback_cython.pyx"]),
    Extension("UCBoost_faster_cython", ["UCBoost_faster_cython.pyx"]),
    # XXX also build the extension with full name?
    # Extension("SMPyBandits.Policies.UCBoost_faster_cython", ["UCBoost_faster_cython.pyx"]),
    Extension("UCBoost_cython", ["UCBoost_cython.pyx"]),
    # XXX also build the extension with full name?
    # Extension("SMPyBandits.Policies.UCBoost_cython", ["UCBoost_cython.pyx"]),
]

setup(
    ext_modules = cythonize(extensions, compiler_directives={
        'embedsignature': True,
        'language_level': 3,
        'warn.undeclared': True,
        'warn.unreachable': True,
        'warn.maybe_uninitialized': True,
        'warn.unused': True,
        'warn.unused_arg': True,
        'warn.unused_result': True,
        'warn.multiple_declarators': True,
    })
)
