# -*- coding: utf-8 -*-
""" Manipulate posteriors of Bernoulli/Beta experiments.

Rewards not in `{0, 1}` are handled with a trick, with a "random binarization", cf., [[Agrawal & Goyal, 2012]](http://jmlr.org/proceedings/papers/v23/agrawal12/agrawal12.pdf) (algorithm 2).
When reward `r_t in [0, 1]` is observed, the player receives the result of a Bernoulli sample of average `r_t`: `r_t <- sample from Bernoulli(r_t)` so it is well in `{0, 1}`.
"""

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann, Lilian Besson"
__version__ = "0.5"

from random import random
try:
    from numpy.random import beta as betavariate  # Faster? Not sure!
except ImportError:
    from random import betavariate
from scipy.special import btdtri

from .Posterior import Posterior


def bernoulliBinarization(r_t):
    """ Return a (random) binarization of a reward r_t in the continuous interval [0, 1] as an observation in discrete {0, 1}."""
    if r_t == 0:
        return 0  # Returns a int!
    elif r_t == 1:
        return 1  # Returns a int!
    else:
        assert 0 <= r_t <= 1, "Error: only bounded rewards in [0, 1] are supported by this Beta posterior right now."
        return int(random() < r_t)


class Beta(Posterior):
    """ Manipulate posteriors of Bernoulli/Beta experiments."""

    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b
        self.N = [a, b]

    def __str__(self):
        return "Beta({}, {})".format(self.N[1], self.N[0])

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
        # FIXED update this code, to accept obs that are FLOAT in [0, 1] and not just in {0, 1}...
        self.N[bernoulliBinarization(obs)] -= 1

    def update(self, obs):
        # print("Info: calling Beta.update() with obs = {} ...".format(obs))  # DEBUG
        # FIXED update this code, to accept obs that are FLOAT in [0, 1] and not just in {0, 1}...
        self.N[bernoulliBinarization(obs)] += 1
