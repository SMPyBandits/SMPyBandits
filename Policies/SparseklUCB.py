# -*- coding: utf-8 -*-
""" The SparseklUCB policy, designed to tackle sparse stochastic bandit problems:

- This means that only a small subset of size ``s`` of the ``K`` arms has non-zero means.
- The SparseklUCB algorithm requires to known **exactly** the value of ``s``.

- Reference: [["Sparse Stochastic Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)] who introduced SparseUCB.
- This SparseklUCB is my version.
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

from math import sqrt, log
from enum import Enum  # For the different states
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!


from .kullback import klucbBern
from .klUCB import klUCB


#: Different states during the SparseklUCB algorithm.
#:
#: - ``RoundRobin`` means all are sampled once.
#: - ``ForceLog`` uniformly explores arms that are in the set :math:`\mathcal{J}(t) \setminus \mathcal{K}(t)`.
#: - ``UCB`` is the phase that the algorithm should converge to, when a normal UCB selection is done only on the "good" arms, i.e., :math:`\mathcal{K}(t)`.
Phase = Enum('Phase', ['RoundRobin', 'ForceLog', 'UCB'])

#: Default value for the constant c used in the computation of KL-UCB index
c = 1.  #: default value, as it was in pymaBandits v1.0
# c = 1.  #: as suggested in the Theorem 1 in https://arxiv.org/pdf/1102.2490.pdf


class SparseklUCB(klUCB):
    """ The SparseklUCB policy, designed to tackle sparse stochastic bandit problems:

    - This means that only a small subset of size ``s`` of the ``K`` arms has non-zero means.
    - The SparseklUCB algorithm requires to known **exactly** the value of ``s``.

    - By default, assume 'sparsity' = 'nbArms'.
    - Reference: [["Sparse Stochastic Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)].
    """

    def __init__(self, nbArms, sparsity=None, tolerance=1e-4, klucb=klucbBern, c=c, lower=0., amplitude=1.):
        super(SparseklUCB, self).__init__(nbArms, tolerance=tolerance, klucb=klucb, c=c, lower=lower, amplitude=amplitude)
        if sparsity is None:
            sparsity = nbArms
            print("Warning: regular klUCB should be used instead of SparseklUCB if 'sparsity' = 'nbArms' = {} ...".format(nbArms))  # DEBUG
        assert 1 <= sparsity <= nbArms, "Error: 'sparsity' has to be in [1, nbArms = {}] but was {} ...".format(nbArms, sparsity)  # DEBUG
        self.sparsity = sparsity  #: Known value of the sparsity of the current problem.
        self.phase = Phase.RoundRobin  #: Current phase of the algorithm.
        # internal memory
        self.force_to_see = np.full(nbArms, True)  #: Binary array for the set :math:`\mathcal{J}(t)`.
        self.goods = np.full(nbArms, True)  #: Binary array for the set :math:`\mathcal{K}(t)`.
        self.offset = -1  #: Next arm to sample, for the Round-Robin phase

    def __str__(self):
        return r"Sparse-KL-UCB($s={}$, {}{})".format(self.sparsity, "" if self.c == 1 else r"$c={:.3g}$".format(self.c), self.klucb.__name__[5:])

    def startGame(self):
        """ Initialize the policy for a new game."""
        super(SparseklUCB, self).startGame()
        self.phase = Phase.RoundRobin
        self.force_to_see.fill(True)  # faster than sets
        self.goods.fill(True)  # faster than sets
        self.offset = -1

    # --- Update the two sets

    def update_j(self):
        r""" Recompute the set :math:`\mathcal{J}(t)`:

        .. math:: \mathcal{J}(t) = \left\{ k \in [1,...,K]\;, \frac{X_k(t)}{N_k(t)} \geq \sqrt{\frac{\c \log(N_k(t))}{N_k(t)}} \right\}.
        """
        assert np.all(self.pulls >= 1), "Error: at least one arm was not already pulled: pulls = {} ...".format(self.pulls)  # DEBUG
        self.force_to_see.fill(False)  # faster than sets
        means = self.rewards / self.pulls
        # UCB_J = np.sqrt((self.c * np.log(self.pulls)) / self.pulls)
        UCB_J = self.klucb(self.rewards / self.pulls, self.c * np.log(self.pulls) / self.pulls, self.tolerance) - means
        self.force_to_see[means >= UCB_J] = True

    def update_k(self):
        r""" Recompute the set :math:`\mathcal{K}(t)`:

        .. math:: \mathcal{K}(t) = \left\{ k \in [1,...,K]\;, \frac{X_k(t)}{N_k(t)} \geq \sqrt{\frac{\c \log(t)}{N_k(t)}} \right\}.
        """
        assert np.all(self.pulls >= 1), "Error: at least one arm was not already pulled: pulls = {} ...".format(self.pulls)  # DEBUG
        self.goods.fill(False)  # faster than sets
        means = self.rewards / self.pulls
        # UCB_K = np.sqrt((self.c * np.log(self.t)) / self.pulls)
        UCB_K = self.klucb(self.rewards / self.pulls, self.c * np.log(self.t) / self.pulls, self.tolerance) - means
        self.goods[means >= UCB_K] = True

    # --- SparseklUCB choice() method

    def choice(self):
        r""" In an index policy, choose an arm with maximal index (uniformly at random):

        .. math:: A(t) \sim U(\arg\max_{1 \leq k \leq K} I_k(t)).
        """
        # print("  At step t = {} a SparseklUCB algorithm was in phase {} ...".format(self.t, self.phase))  # DEBUG
        if (self.phase == Phase.RoundRobin) and ((1 + self.offset) < self.nbArms):
            # deterministic phase
            self.offset += 1
            return self.offset
        else:
            self.update_j()
            j = self.force_to_see
            # print("    At step t = {}, set j = {} and set k = {} ...".format(self.t, set_j, set_k))  # DEBUG
            # 1st case: Round-Robin phase
            if np.sum(j) < self.sparsity:
                self.phase = Phase.RoundRobin
                self.offset = 0
                return self.offset
            # 2nd case: Force-Log Phase
            else:
                self.update_k()
                k = self.goods
                if np.sum(k) < self.sparsity:
                    self.phase = Phase.ForceLog
                    diff_of_set = j & (~k)  # component-wise boolean operations to the numpy array
                    return np.random.choice(np.nonzero(diff_of_set)[0])
                # 3rd case: UCB phase
                else:
                    self.phase = Phase.UCB
                    return self.choiceFromSubSet(availableArms=np.nonzero(self.goods)[0])

    # --- computeIndex and computeAllIndex are the same as klUCB

    # --- Same as klUCB

    # def computeIndex(self, arm):
    #     r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

    #     .. math::

    #        \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
    #        U_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c \log(t)}{N_k(t)} \right\},\\
    #        I_k(t) &= \hat{\mu}_k(t) + U_k(t).

    #     If rewards are in :math:`[a, b]` (default to :math:`[0, 1]`) and :math:`\mathrm{kl}(x, y)` is the Kullback-Leibler divergence between two distributions of means x and y (see :mod:`Arms.kullback`),
    #     and c is the parameter (default to 1).
    #     """
    #     if self.pulls[arm] < 1:
    #         return float('+inf')
    #     else:
    #         # XXX We could adapt tolerance to the value of self.t
    #         return self.klucb(self.rewards[arm] / self.pulls[arm], self.c * log(self.t) / self.pulls[arm], self.tolerance)

    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     indexes = self.klucb(self.rewards / self.pulls, self.c * np.log(self.t) / self.pulls, self.tolerance)
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index = indexes
