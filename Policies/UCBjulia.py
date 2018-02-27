# -*- coding: utf-8 -*-
""" The UCB policy for bounded bandits, with UCB indexes computed with Julia.
Reference: [Lai & Robbins, 1985].

.. warning::

    Using a Julia function *from* Python will not speed up anything, as there is a lot of overhead in the "bridge" protocol used by pyjulia.
    The idea of using naively a tiny Julia function to speed up computations is basically useless.

    A naive benchmark showed that in this approach, :class:`UCBjulia` (used withing Python) is about 125 times slower (!) than :class:`UCB`.

.. warning:: This is only experimental, and purely useless. See https://github.com/SMPyBandits/SMPyBandits/issues/98
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from .IndexPolicy import IndexPolicy


class UCBjulia(IndexPolicy):
    """ The UCB policy for bounded bandits, with UCB indexes computed with Julia.
    Reference: [Lai & Robbins, 1985].

    .. warning:: This is only experimental, and purely useless. See https://github.com/SMPyBandits/SMPyBandits/issues/98
    """

    def __init__(self, nbArms, lower=0., amplitude=1.):
        """ Will fail directly if the bridge with julia is unavailable or buggy."""
        super(UCBjulia, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.t = 0
        # Importing the julia module and creating the bridge
        try:
            import julia
        except ImportError:
            print("Error: unable to load the 'julia' Python module. Install with 'pip install julia', or see https://github.com/JuliaPy/pyjulia/")  # DEBUG
        _j = julia.Julia()
        try:
            self._index_function = _j.evalfile("Policies/UCBjulia.jl")
        except RuntimeError:
            try:
                self._index_function = _j.evalfile("UCBjulia.jl")
            except RuntimeError:
                raise ValueError("Error: Unable to load 'UCBjulia.jl' julia file.")  # WARNING
        try:
            self._index_function([1], [1], 1, 1)
        except (RuntimeError, ValueError):
            raise ValueError("Error: the index function loaded from 'UCBjulia.jl' is bugged or unavailable.")  # WARNING

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{2 \log(t)}{N_k(t)}}.
        """
        return self._index_function(self.rewards, self.pulls, self.t, arm + 1)