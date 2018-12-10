# -*- coding: utf-8 -*-
r""" Manipulate posteriors of Bernoulli/Beta experiments., for discounted Bayesian policies (:class:`Policies.DiscountedBayesianIndexPolicy`).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

# Local imports
try:
    from .Beta import Beta, bernoulliBinarization

    from .with_proba import with_proba
except (ImportError, SystemError):
    from Beta import Beta, bernoulliBinarization

    from with_proba import with_proba

try:
    from numpy.random import beta as betavariate  # Faster! Yes!
except ImportError:
    from random import betavariate
from scipy.special import btdtri


# --- Constants

#: Default value for the discount factor :math:`\gamma\in(0,1)`.
#: ``0.95`` is empirically a reasonable value for short-term non-stationary experiments.
GAMMA = 0.95


# --- Class

class DiscountedBeta(Beta):
    r""" Manipulate posteriors of Bernoulli/Beta experiments, for discounted Bayesian policies (:class:`Policies.DiscountedBayesianIndexPolicy`).

    - It keeps :math:`\tilde{S}(t)` and :math:`\tilde{F}(t)` the *discounted* counts of successes and failures (S and F).
    """

    def __init__(self, gamma=GAMMA, a=1, b=1):
        r""" Create a Beta posterior :math:`\mathrm{Beta}(\alpha, \beta)` with no observation, i.e., :math:`\alpha = 1` and :math:`\beta = 1` by default."""
        assert a >= 0, "Error: parameter 'a' for Beta posterior has to be >= 0."  # DEBUG
        self._a = a
        assert b >= 0, "Error: parameter 'b' for Beta posterior has to be >= 0."  # DEBUG
        self._b = b
        self.N = [0, 0]  #: List of two parameters [a, b]
        assert 0 < gamma <= 1, "Error: for a DiscountedBayesianIndexPolicy policy, the discount factor has to be in (0,1], but it was {}.".format(gamma)  # DEBUG
        if gamma == 1:
            print("Warning: gamma = 1 is stupid, just use a regular Beta posterior!")  # DEBUG
        self.gamma = gamma  #: Discount factor :math:`\gamma\in(0,1)`.

    def __str__(self):
        return r"DiscountedBeta(\alpha={:.3g}, \beta={:.3g})".format(self.N[1], self.N[0])

    def reset(self, a=None, b=None):
        """Reset alpha and beta, both to 0 as when creating a new default DiscountedBeta."""
        if a is None:
            a = self._a
        if b is None:
            b = self._b
        self.N = [0, 0]

    def sample(self):
        """Get a random sample from the DiscountedBeta posterior (using :func:`numpy.random.betavariate`).

        - Used only by :class:`Thompson` Sampling and :class:`AdBandits` so far.
        """
        return betavariate(self._a + self.N[1], self._b + self.N[0])

    def quantile(self, p):
        """Return the p quantile of the DiscountedBeta posterior (using :func:`scipy.stats.btdtri`).

        - Used only by :class:`BayesUCB` and :class:`AdBandits` so far.
        """
        return btdtri(self._a + self.N[1], self._b + self.N[0], p)
        # Bug: do not call btdtri with (0.5,0.5,0.5) in scipy version < 0.9 (old)

    def forget(self, obs):
        """Forget the last observation, and undiscount the count of observations."""
        # print("Info: calling DiscountedBeta.forget() with obs = {}, self.N = {} and self.gamma = {} ...".format(obs, self.N, self.gamma))  # DEBUG
        # FIXED update this code, to accept obs that are FLOAT in [0, 1] and not just in {0, 1}...
        binaryObs = bernoulliBinarization(obs)
        self.N[binaryObs] = (self.N[binaryObs] - 1) / self.gamma
        otherObs = 1 - binaryObs
        self.N[otherObs] = self.N[otherObs] / self.gamma

    def update(self, obs):
        r""" Add an observation, and discount the previous observations.

        - If obs is 1, update :math:`\alpha` the count of positive observations,
        - If it is 0, update :math:`\beta` the count of negative observations.

        - But instead of using :math:`\tilde{S}(t) = S(t)` and :math:`\tilde{N}(t) = N(t)`, they are updated at each time step using the discount factor :math:`\gamma`:

        .. math::
            \tilde{S}(t+1) &= \gamma \tilde{S}(t) + r(t),
            \tilde{F}(t+1) &= \gamma \tilde{F}(t) + (1 - r(t)).

        .. note:: Otherwise, a trick with :func:`bernoulliBinarization` has to be used.
        """
        # print("Info: calling DiscountedBeta.update() with obs = {}, self.N = {} and self.gamma = {} ...".format(obs, self.N, self.gamma))  # DEBUG
        # FIXED update this code, to accept obs that are FLOAT in [0, 1] and not just in {0, 1}...
        binaryObs = bernoulliBinarization(obs)
        self.N[binaryObs] = self.gamma * self.N[binaryObs] + 1
        otherObs = 1 - binaryObs
        self.N[otherObs] = self.gamma * self.N[otherObs]

    def discount(self):
        r""" Simply discount the old observation, when no observation is given at this time.

        .. math::
            \tilde{S}(t+1) &= \gamma \tilde{S}(t),
            \tilde{F}(t+1) &= \gamma \tilde{F}(t).
        """
        # print("Info: calling DiscountedBeta.discount() self.N = {} and self.gamma = {} ...".format(self.N, self.gamma))  # DEBUG
        self.N[0] = max(0, self.gamma * self.N[0])
        self.N[1] = max(0, self.gamma * self.N[1])

    def undiscount(self):
        r""" Simply cancel the discount on the old observation, when no observation is given at this time.

        .. math::
            \tilde{S}(t+1) &= \frac{1}{\gamma} \tilde{S}(t),
            \tilde{F}(t+1) &= \frac{1}{\gamma} \tilde{F}(t).
        """
        # print("Info: calling DiscountedBeta.undiscount() self.N = {} and self.gamma = {} ...".format(self.N, self.gamma))  # DEBUG
        self.N[0] = max(0, self.N[0] / self.gamma)
        self.N[1] = max(0, self.N[1] / self.gamma)
