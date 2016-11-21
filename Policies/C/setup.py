# -*- coding: utf-8 -*-
""" Utility for building the C library for Python 2."""

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.3 $"

from distutils.core import setup, Extension

module1 = Extension('kullback', sources=['kullback.c'])


setup(name='Kullback utilities',
      version='1.0',
      description='computes various KL divergences',
      ext_modules=[module1]
      )
