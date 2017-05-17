#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Utility for building the C library for Python 2/3."""

__author__ = "Lilian Besson"
__version__ = "0.6"

from distutils.core import setup
from Cython.Build import cythonize


setup(name='Lempel-Ziv complexity',
      version='1.0',
      description='Fast implementation of the Lempel-Ziv complexity function',
      ext_modules=cythonize('lempel_ziv_complexity_cython.pyx')
      )
