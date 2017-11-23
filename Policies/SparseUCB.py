# -*- coding: utf-8 -*-
""" The SparseUCB policy, designed to tackle sparse stochastic bandit problems:

- This means that only a small subset of size ``s`` of the ``K`` arms has non-zero means.
- The SparseUCB algorithm requires to known **exactly** the value of ``s``.

- Reference: [["Sparse Stochastic Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)].
"""
from __future__ import division, print_function  # Python 2 compatibility

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

#: Default parameter for :math:`\alpha` for the UCB indexes.
ALPHA = 4


# --- The interesting class


class SparseUCB(UCBalpha):
    """ The SparseUCB policy, designed to tackle sparse stochastic bandit problems.

    - By default, assume ``sparsity`` = ``nbArms``.
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

    # --- pretty printing

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
        # assert np.all(self.pulls >= 1), "Error: at least one arm was not already pulled: pulls = {} ...".format(self.pulls)  # DEBUG
        self.force_to_see.fill(False)  # faster than sets
        means = self.rewards / self.pulls
        means[self.pulls < 1] = float('+inf')
        UCB_J = np.sqrt((self.alpha * np.log(self.pulls)) / self.pulls)
        UCB_J[self.pulls < 1] = float('+inf')
        self.force_to_see[means >= UCB_J] = True

    def update_k(self):
        r""" Recompute the set :math:`\mathcal{K}(t)`:

        .. math:: \mathcal{K}(t) = \left\{ k \in [1,...,K]\;, \frac{X_k(t)}{N_k(t)} \geq \sqrt{\frac{\alpha \log(t)}{N_k(t)}} \right\}.
        """
        # assert np.all(self.pulls >= 1), "Error: at least one arm was not already pulled: pulls = {} ...".format(self.pulls)  # DEBUG
        self.goods.fill(False)  # faster than sets
        means = self.rewards / self.pulls
        means[self.pulls < 1] = float('+inf')
        UCB_K = np.sqrt((self.alpha * np.log(self.t)) / self.pulls)
        UCB_K[self.pulls < 1] = float('+inf')
        self.goods[means >= UCB_K] = True

    # --- SparseUCB choice() method

    def choice(self):
        r""" Choose the next arm to play:

        - If still in a Round-Robin phase, play the next arm,
        - Otherwise, recompute the set :math:`\mathcal{J}(t)`,
        - If it is too small, if :math:`\mathcal{J}(t) < s`:
           + Start a new Round-Robin phase from arm 0.
        - Otherwise, recompute the second set :math:`\mathcal{K}(t)`,
        - If it is too small, if :math:`\mathcal{K}(t) < s`:
           + Play a Force-Log step by choosing an arm uniformly at random from the set :math:`\mathcal{J}(t) \setminus \mathcal{K}(t)`.
        - Otherwise,
           + Play a UCB step by choosing an arm with highest UCB index from the set :math:`\mathcal{K}(t)`.
        """
        # print("  At step t = {} a SparseUCB algorithm was in phase {} ...".format(self.t, self.phase))  # DEBUG
        if (self.phase == Phase.RoundRobin) and ((1 + self.offset) < self.nbArms):
            # deterministic phase
            self.offset += 1
            return self.offset
        else:
            self.update_j()
            j = self.force_to_see
            # DEBUG small checks, to remove soon
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
                    diff_of_set = j & (~k)  # component-wise boolean operations to the numpy array
                    return np.random.choice(np.nonzero(diff_of_set)[0])
                # 3rd case: UCB phase
                else:
                    self.phase = Phase.UCB
                    return self.choiceFromSubSet(availableArms=np.nonzero(self.goods)[0])

    # --- computeIndex and computeAllIndex are the same as UCBalpha
