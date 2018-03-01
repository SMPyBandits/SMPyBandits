# -*- coding: utf-8 -*-
r""" The basic Probability Pursuit algorithm.

- We use the simple version of the pursuit algorithm, as described in the seminal book by Sutton and Barto (1998), https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html.
- Initially, a uniform probability is set on each arm, :math:`p_k(0) = 1/k`.
- At each time step :math:`t`, the probabilities are *all* recomputed, following this equation:

  .. math::

     p_k(t+1) = \begin{cases}
     (1 - \beta) p_k(t) + \beta \times 1 & \text{if}\; \hat{\mu}_k(t) = \max_j \hat{\mu}_j(t) \\
     (1 - \beta) p_k(t) + \beta \times 0 & \text{otherwise}.
     \end{cases}

- :math:`\beta \in (0, 1)` is a *learning rate*, default is `BETA = 0.5`.
- And then arm :math:`A_k(t+1)` is randomly selected from the distribution :math:`(p_k(t+1))_{1 \leq k \leq K}`.

- References: [Kuleshov & Precup - JMLR, 2000](http://www.cs.mcgill.ca/~vkules/bandits.pdf#page=6), [Sutton & Barto, 1998]
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!
import numpy.random as rn

from .BasePolicy import BasePolicy

#: Default value for the beta parameter
BETA = 0.5


class ProbabilityPursuit(BasePolicy):
    r""" The basic Probability pursuit algorithm.

    - References: [Kuleshov & Precup - JMLR, 2000](http://www.cs.mcgill.ca/~vkules/bandits.pdf#page=6), [Sutton & Barto, 1998]
    """

    def __init__(self, nbArms, beta=BETA, prior='uniform', lower=0., amplitude=1.):
        super(ProbabilityPursuit, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # Constant _beta
        assert 0 <= beta <= 1, "Error: the 'beta' parameter for ProbabilityPursuit class has to be in [0, 1]."  # DEBUG
        self._beta = beta
        # Initialize the probabilities
        self._prior = prior
        if prior is not None and prior != 'uniform':
            assert len(prior) == self.nbArms, "Error: the 'prior' argument given to ProbabilityPursuit has to be an array of the good size ({}).".format(nbArms)  # DEBUG
            self.probabilities = prior  #: Probabilities of each arm
        else:   # Assume uniform prior if not given or if = 'uniform'
            self.probabilities = np.full(nbArms, 1. / nbArms)

    def startGame(self):
        """Reinitialize probabilities."""
        super(ProbabilityPursuit, self).startGame()
        if self._prior is not None and self._prior != 'uniform':
            self.probabilities = self._prior  #: Probabilities of each arm
        else:   # Assume uniform prior if not given or if = 'uniform'
            self.probabilities = np.full(self.nbArms, 1. / self.nbArms)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def beta(self):  # Allow child classes to use time-dependent beta coef
        r"""Constant parameter :math:`\beta(t) = \beta(0)`."""
        return self._beta

    def __str__(self):
        return "Pursuit({})".format(self.beta)

    def getReward(self, arm, reward):
        r"""Give a reward: accumulate rewards on that arm k, then update the probabilities :math:`p_k(t)` of each arm. """
        super(ProbabilityPursuit, self).getReward(arm, reward)  # XXX Call to BasePolicy
        # Update all probabilities
        means = self.rewards / self.pulls
        self.probabilities = (1 - self.beta) * self.probabilities + self.beta * (means == np.max(means))
        # Renormalize probabilities at each step
        self.probabilities /= np.sum(self.probabilities)

    # --- Choice methods

    def choice(self):
        r"""One random selection, with probabilities :math:`(p_k(t))_{1 \leq k \leq K}`, thank to :func:`numpy.random.choice`."""
        return rn.choice(self.nbArms, p=self.probabilities)

    def choiceWithRank(self, rank=1):
        r"""Multiple (rank >= 1) random selection, with probabilities :math:`(p_k(t))_{1 \leq k \leq K}`, thank to :func:`numpy.random.choice`, and select the last one (less probable)."""
        if (self.t < self.nbArms) or (rank == 1):
            return self.choice()
        else:
            return rn.choice(self.nbArms, size=rank, replace=False, p=self.probabilities)[rank - 1]

    def choiceFromSubSet(self, availableArms='all'):
        r"""One random selection, from availableArms, with probabilities :math:`(p_k(t))_{1 \leq k \leq K}`, thank to :func:`numpy.random.choice`."""
        if (self.t < self.nbArms) or (availableArms == 'all') or (len(availableArms) == self.nbArms):
            return self.choice()
        else:
            return rn.choice(availableArms, p=self.probabilities[availableArms])

    def choiceMultiple(self, nb=1):
        r"""Multiple (nb >= 1) random selection, with probabilities :math:`(p_k(t))_{1 \leq k \leq K}`, thank to :func:`numpy.random.choice`."""
        if (self.t < self.nbArms) or (nb == 1):
            return np.array([self.choice() for _ in range(nb)])  # good size if nb > 1 but t < nbArms
        else:
            return rn.choice(self.nbArms, size=nb, replace=False, p=self.probabilities)
