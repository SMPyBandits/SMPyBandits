# -*- coding: utf-8 -*-
""" The UCB1 (UCB-alpha) index policy, using a Cython extension.

- Reference: [Auer et al. 02].

.. warning::

    This extension should be used with the ``setup.py`` script, by running::

        $ python setup.py build_ext --inplace

    You can also use [pyximport](http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html#pyximport-cython-compilation-for-developers) to import the ``kullback_cython`` module transparently:

    >>> import pyximport; pyximport.install()  # instantaneous  # doctest: +ELLIPSIS
    (None, <pyximport.pyximport.PyxImporter at 0x...>)
    >>> from UCBcython import *     # takes about two seconds
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from libc.math cimport log, sqrt, exp, ceil, floor

import numpy as np
# cimport numpy as np  # WARNING might be deprecated
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!
from sys import path; path.insert(0, '..')

try:
    # from IndexPolicy import IndexPolicy
    import IndexPolicy as INDEXPOLICY
    IndexPolicy = INDEXPOLICY.IndexPolicy
except ImportError:
    from .IndexPolicy import IndexPolicy

try:
    import UCB as UCBMODULE
    UCB = UCBMODULE.UCB
except ImportError:
    from .UCB import UCB

#: Default parameter for alpha
cdef float ALPHA
ALPHA = 1
ALPHA = 4


cdef float UCBindex(float reward, float pull, float t, int arm, float alpha=ALPHA):
    if pull < 1:
        return float('+inf')
    else:
        return (reward / pull) + sqrt((alpha * log(t)) / (2 * pull))


class UCBcython(UCB):
    """ The UCB1 (UCB-alpha) index policy, using a Cython extension.

    - Reference: [Auer et al. 02].
    """

    def __init__(self, nbArms, alpha=ALPHA, lower=0., amplitude=1.):
        super(UCBcython, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert alpha >= 0, "Error: the alpha parameter for UCBcython class has to be >= 0."  # DEBUG
        self.alpha = alpha  #: Parameter alpha

    def __str__(self):
        return r"UCBcython($\alpha={:.3g}$)".format(self.alpha)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{\alpha \log(t)}{2 N_k(t)}}.
        """
        return UCBindex(self.rewards[arm], self.pulls[arm], self.t, self.alpha)
        # if self.pulls[arm] < 1:
        #     return float('+inf')
        # else:
        #     return (self.rewards[arm] / self.pulls[arm]) + sqrt((self.alpha * log(self.t)) / (2 * self.pulls[arm]))

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)
        # indexes = (self.rewards / self.pulls) + np.sqrt((self.alpha * np.log(self.t)) / (2 * self.pulls))
        # indexes[self.pulls < 1] = float('+inf')
        # self.index[:] = indexes
