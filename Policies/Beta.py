# -*- coding: utf-8 -*-
""" Manipulate posteriors of Bernoulli/Beta experiments.

Rewards not in `{0, 1}` are handled with a trick, with a "random binarization", cf., [[Agrawal & Goyal, 2012]](http://jmlr.org/proceedings/papers/v23/agrawal12/agrawal12.pdf) (algorithm 2).
When reward `r_t in [0, 1]` is observed, the player receives the result of a Bernoulli sample of average `r_t`: `r_t <- sample from Bernoulli(r_t)` so it is well in `{0, 1}`.

- See https://en.wikipedia.org/wiki/Bernoulli_distribution#Related_distributions
- And https://en.wikipedia.org/wiki/Conjugate_prior#Discrete_distributions
"""

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann, Lilian Besson"
__version__ = "0.5"

from random import random
try:
    from numpy.random import beta as betavariate  # Faster! Yes!
except ImportError:
    from random import betavariate
from scipy.special import btdtri

# Local imports
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
        """Create a Beta posterior with no observation, i.e. alpha = 1 and beta = 1."""
        assert a >= 0, "Error: parameter 'a' for Beta posterior has to be >= 0."
        self.a = a  #: Parameter alpha
        assert b >= 0, "Error: parameter 'b' for Beta posterior has to be >= 0."
        self.b = b  #: Parameter beta
        self.N = [a, b]  #: [a, b]

    def __str__(self):
        return "Beta({}, {})".format(self.N[1], self.N[0])

    def reset(self, a=None, b=None):
        """Reset alpha and beta, both to 1 as when creating a new default Beta."""
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        self.N = [a, b]

    def sample(self):
        """Get a random sample from the Beta posterior (using :func:`numpy.random.betavariate`).

        - Used only by Thompson Sampling so far.
        """
        return betavariate(self.N[1], self.N[0])

    def quantile(self, p):
        """Return the p quantile of the Beta posterior (using :func:`scipy.stats.btdtri`).

        - Used only by BayesUCB so far.
        """
        return btdtri(self.N[1], self.N[0], p)
        # Bug: do not call btdtri with (0.5,0.5,0.5) in scipy version < 0.9 (old)

    def mean(self):
        """Compute the mean of the Beta posterior (should be useless)."""
        return self.N[1] / float(sum(self.N))

    def forget(self, obs):
        """Forget the last observation."""
        # print("Info: calling Beta.forget() with obs = {} ...".format(obs))  # DEBUG
        # FIXED update this code, to accept obs that are FLOAT in [0, 1] and not just in {0, 1}...
        self.N[bernoulliBinarization(obs)] -= 1

    def update(self, obs):
        """Add an observation. If obs is 1, update alpha, if it is 0, update beta, otherwise, a trick with :func:`bernoulliBinarization` has to be used."""
        # print("Info: calling Beta.update() with obs = {} ...".format(obs))  # DEBUG
        # FIXED update this code, to accept obs that are FLOAT in [0, 1] and not just in {0, 1}...
        self.N[bernoulliBinarization(obs)] += 1
