# -*- coding: utf8 -*-
""" Test module to be called from C++"""

# from __future__ import print_function

def add(a, b):
    """ Returns the sum of two numbers."""
    a, b = int(a), int(b)
    c = str(a + b)
    print("a = {} and b = {} and a + b = {}".format(a, b, c))
    return c
