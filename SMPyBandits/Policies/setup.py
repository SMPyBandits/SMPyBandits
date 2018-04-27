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
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("kullback_cython.pyx")
)
