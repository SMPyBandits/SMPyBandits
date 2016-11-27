# -*- coding: utf-8 -*-
""" Manipulate posteriors of Bernoulli/Beta experiments.
"""

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann"
__version__ = "$Revision: 1.7 $"

try:
    from numpy.random import beta as betavariate  # Faster
except ImportError:
    from random import betavariate
from scipy.special import btdtri


class Beta:
    """ Manipulate posteriors of Bernoulli/Beta experiments."""

    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b
        self.N = [a, b]

    def __repr__(self):
        return 'Beta(' + repr(self.a) + ', ' + repr(self.b) + ')'

    def reset(self, a=0, b=0):
        if a == 0:
            a = self.a
        if b == 0:
            b = self.b
        self.N = [a, b]

    def sample(self):
        return betavariate(self.N[1], self.N[0])

    def quantile(self, p):
        return btdtri(self.N[1], self.N[0], p)
        # Bug: do not call btdtri with (0.5,0.5,0.5) in scipy version < 0.9 (old)

    def mean(self):
        return self.N[1] / float(sum(self.N))

    def forget(self, obs):
        # print("Info: calling Beta.forget() with obs = {} ...".format(obs))  # DEBUG
        # FIXME update this code, to accept obs that are FLOAT in [0, 1] and not just in {0, 1}...
        self.N[int(obs)] -= 1

    def update(self, obs):
        # print("Info: calling Beta.update() with obs = {} ...".format(obs))  # DEBUG
        # FIXME update this code, to accept obs that are FLOAT in [0, 1] and not just in {0, 1}...
        self.N[int(obs)] += 1
