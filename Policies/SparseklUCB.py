# -*- coding: utf-8 -*-
""" The SparseUCB policy, designed to tackle sparse stochastic bandit problems:

- This means that only a small subset of size ``s`` of the ``K`` arms has non-zero means.
- The SparseUCB algorithm requires to known **exactly** the value of ``s``.

- Reference: [["Sparse Stochastic Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)].
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

from math import sqrt, log
from enum import Enum  # For the different states
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .UCBalpha import UCBalpha


#: Different states during the SparseUCB algorithm.
#:
#: - ``RoundRobin`` means all are sampled once.
#: - ``ForceLog`` uniformly explores arms that are in the set :math:`\mathcal{J}(t) \setminus \mathcal{K}(t)`.
#: - ``UCB`` is the phase that the algorithm should converge to, when a normal UCB selection is done only on the "good" arms, i.e., :math:`\mathcal{K}(t)`.
Phase = Enum('Phase', ['RoundRobin', 'ForceLog', 'UCB'])

#: Default parameter for alpha.
ALPHA = 4


class SparseUCB(UCBalpha):
    """ The SparseUCB policy, designed to tackle sparse stochastic bandit problems:

    - This means that only a small subset of size ``s`` of the ``K`` arms has non-zero means.
    - The SparseUCB algorithm requires to known **exactly** the value of ``s``.

    - By default, assume 'sparsity' = 'nbArms'.
    - Reference: [["Sparse Stochastic Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)].
    """

    def __init__(self, nbArms, sparsity=None, alpha=ALPHA, lower=0., amplitude=1.):
        super(SparseUCB, self).__init__(nbArms, alpha=alpha, lower=lower, amplitude=amplitude)
        if sparsity is None:
            sparsity = nbArms
            print("Warning: regular UCBalpha should be used instead of SparseUCB if 'sparsity' = 'nbArms' = {} ...".format(nbArms))  # DEBUG
        assert 1 <= sparsity <= nbArms, "Error: 'sparsity' has to be in [1, nbArms = {}] but was {} ...".format(nbArms, sparsity)  # DEBUG
        self.sparsity = sparsity  #: Known value of the sparsity of the current problem.
        self.phase = Phase.RoundRobin  #: Current phase of the algorithm.
        # internal memory
        self.force_to_see = np.full(nbArms, True)  #: Binary array for the set :math:`\mathcal{J}(t)`.
        self.goods = np.full(nbArms, True)  #: Binary array for the set :math:`\mathcal{K}(t)`.
        self.offset = -1  #: Next arm to sample, for the Round-Robin phase

    def __str__(self):
        return r"SparseUCB($s={}$, $\alpha={:.3g}$)".format(self.sparsity, self.alpha)

    def startGame(self):
        """ Initialize the policy for a new game."""
        super(SparseUCB, self).startGame()
        self.phase = Phase.RoundRobin
        self.force_to_see.fill(True)  # faster than sets
        self.goods.fill(True)  # faster than sets
        self.offset = -1

    # --- Update the two sets

    def update_j(self):
        r""" Recompute the set :math:`\mathcal{J}(t)`:

        .. math:: \mathcal{J}(t) = \left\{ k \in [1,...,K]\;, \frac{X_k(t)}{N_k(t)} \geq \sqrt{\frac{\alpha \log(N_k(t))}{N_k(t)}} \right\}.
        """
        assert np.all(self.pulls >= 1), "Error: at least one arm was not already pulled: pulls = {} ...".format(self.pulls)  # DEBUG
        self.force_to_see.fill(False)  # faster than sets
        means = self.rewards / self.pulls
        UCB_J = np.sqrt((self.alpha * np.log(self.pulls)) / self.pulls)
        self.force_to_see[means >= UCB_J] = True

    def update_k(self):
        r""" Recompute the set :math:`\mathcal{K}(t)`:

        .. math:: \mathcal{K}(t) = \left\{ k \in [1,...,K]\;, \frac{X_k(t)}{N_k(t)} \geq \sqrt{\frac{\alpha \log(t)}{N_k(t)}} \right\}.
        """
        assert np.all(self.pulls >= 1), "Error: at least one arm was not already pulled: pulls = {} ...".format(self.pulls)  # DEBUG
        self.goods.fill(False)  # faster than sets
        means = self.rewards / self.pulls
        UCB_K = np.sqrt((self.alpha * np.log(self.t)) / self.pulls)
        self.goods[means >= UCB_K] = True

    # --- SparseUCB choice() method

    def choice(self):
        r""" In an index policy, choose an arm with maximal index (uniformly at random):

        .. math:: A(t) \sim U(\arg\max_{1 \leq k \leq K} I_k(t)).
        """
        # print("  At step t = {} a SparseUCB algorithm was in phase {} ...".format(self.t, self.phase))  # DEBUG
        if (self.phase == Phase.RoundRobin) and ((1 + self.offset) < self.nbArms):
            # deterministic phase
            self.offset += 1
            return self.offset
        else:
            self.update_j()
            j = self.force_to_see
            # FIXME small checks, to remove soon
            # assert np.all(j[k]), "Error: set k = {} was not found include in set j = {} but it should be...".format(k, j)  # DEBUG
            # set_j = set(np.nonzero(j)[0])
            # set_k = set(np.nonzero(k)[0])
            # assert set_k <= set_j, "Error: set k = {} was not found include in set j = {} but it should be...".format(set_k, set_j)  # DEBUG
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
                    diff_of_set = j & (~k)
                    return np.random.choice(np.nonzero(diff_of_set)[0])
                # 3rd case: UCB phase
                else:
                    self.phase = Phase.UCB
                    return self.choiceFromSubSet(availableArms=np.nonzero(self.goods)[0])

    # --- Same as UCBalpha

    # def computeIndex(self, arm):
    #     r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

    #     .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{\alpha \log(t)}{N_k(t)}}.
    #     """
    #     if self.pulls[arm] < 1:
    #         return float('+inf')
    #     else:
    #         return (self.rewards[arm] / self.pulls[arm]) + sqrt((self.alpha * log(self.t)) / self.pulls[arm])

    # def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.pulls) + np.sqrt((self.alpha * np.log(self.t)) / self.pulls)
        indexes[self.pulls < 1] = float('+inf')
        self.index = indexes
