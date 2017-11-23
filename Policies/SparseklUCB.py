# -*- coding: utf-8 -*-
""" The SparseklUCB policy, designed to tackle sparse stochastic bandit problems:

- This means that only a small subset of size ``s`` of the ``K`` arms has non-zero means.
- The SparseklUCB algorithm requires to known **exactly** the value of ``s``.

- This SparseklUCB is my version. It uses the KL-UCB index for both the decision in the UCB phase and the construction of the sets :math:`\mathcal{J}(t)` and :math:`\mathcal{K}(t)`.
- The usual UCB indexes can be used for the sets by setting the flag ``use_ucb_for_sets`` to true.
- Reference: [["Sparse Stochastic Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)] who introduced SparseUCB.
"""
from __future__ import division, print_function  # Python 2 compatibility

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


#: Default value for the flag controlling whether the usual UCB indexes are used for the sets :math:`\mathcal{J}(t)`
#: and :math:`\mathcal{K}(t)`. Default it to use the KL-UCB indexes, which should be more efficient.
USE_UCB_FOR_SETS = True
USE_UCB_FOR_SETS = False


# --- The interesting class


class SparseklUCB(klUCB):
    """ The SparseklUCB policy, designed to tackle sparse stochastic bandit problems.

    - By default, assume ``sparsity`` = ``nbArms``.
    """

    def __init__(self, nbArms, sparsity=None,
                 tolerance=1e-4, klucb=klucbBern, c=c,
                 use_ucb_for_sets=USE_UCB_FOR_SETS,
                 lower=0., amplitude=1.):
        super(SparseklUCB, self).__init__(nbArms, tolerance=tolerance, klucb=klucb, c=c, lower=lower, amplitude=amplitude)
        if sparsity is None:
            sparsity = nbArms
            print("Warning: regular klUCB should be used instead of SparseklUCB if 'sparsity' = 'nbArms' = {} ...".format(nbArms))  # DEBUG
        assert 1 <= sparsity <= nbArms, "Error: 'sparsity' has to be in [1, nbArms = {}] but was {} ...".format(nbArms, sparsity)  # DEBUG
        self.sparsity = sparsity  #: Known value of the sparsity of the current problem.
        self.use_ucb_for_sets = use_ucb_for_sets  #: Whether the usual UCB indexes are used for the sets :math:`\mathcal{J}(t)` and :math:`\mathcal{K}(t)`.
        self.phase = Phase.RoundRobin  #: Current phase of the algorithm.
        # internal memory
        self.force_to_see = np.full(nbArms, True)  #: Binary array for the set :math:`\mathcal{J}(t)`.
        self.goods = np.full(nbArms, True)  #: Binary array for the set :math:`\mathcal{K}(t)`.
        self.offset = -1  #: Next arm to sample, for the Round-Robin phase

    # --- pretty printing

    def __str__(self):
        return r"Sparse-KL-UCB($s={}$, {}{}{})".format(self.sparsity, "" if self.c == 1 else r"$c={:.3g}$".format(self.c), self.klucb.__name__[5:], ", UCB for sets" if self.use_ucb_for_sets else "")

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

        .. math::

           \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
           U^{\mathcal{J}}_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c \log(N_k(t))}{N_k(t)} \right\},\\
           \mathcal{J}(t) &= \left\{ k \in [1,...,K]\;, \hat{\mu}_k(t) \geq U^{\mathcal{J}}_k(t) - \hat{\mu}_k(t) \right\}.

        - If ``use_ucb_for_sets`` is ``True``, the same formula from :class:`Policies.SparseUCB` is used.
        """
        # assert np.all(self.pulls >= 1), "Error: at least one arm was not already pulled: pulls = {} ...".format(self.pulls)  # DEBUG
        self.force_to_see.fill(False)  # faster than sets
        means = self.rewards / self.pulls
        means[self.pulls < 1] = float('+inf')
        if self.use_ucb_for_sets:
            UCB_J = np.sqrt((self.c * np.log(self.pulls)) / self.pulls)
            UCB_J[self.pulls < 1] = float('+inf')
        else:
            UCB_J = self.klucb(self.rewards / self.pulls, self.c * np.log(self.pulls) / self.pulls, self.tolerance) - means
            UCB_J[self.pulls < 1] = float('+inf')
        self.force_to_see[means >= UCB_J] = True

    def update_k(self):
        r""" Recompute the set :math:`\mathcal{K}(t)`:

        .. math::

           \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
           U^{\mathcal{K}}_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c \log(t)}{N_k(t)} \right\},\\
           \mathcal{J}(t) &= \left\{ k \in [1,...,K]\;, \hat{\mu}_k(t) \geq U^{\mathcal{K}}_k(t) - \hat{\mu}_k(t) \right\}.

        - If ``use_ucb_for_sets`` is ``True``, the same formula from :class:`Policies.SparseUCB` is used.
        """
        # assert np.all(self.pulls >= 1), "Error: at least one arm was not already pulled: pulls = {} ...".format(self.pulls)  # DEBUG
        self.goods.fill(False)  # faster than sets
        means = self.rewards / self.pulls
        means[self.pulls < 1] = float('+inf')
        if self.use_ucb_for_sets:
            UCB_K = np.sqrt((self.c * np.log(self.t)) / self.pulls)
            UCB_K[self.pulls < 1] = float('+inf')
        else:
            UCB_K = self.klucb(self.rewards / self.pulls, self.c * np.log(self.t) / self.pulls, self.tolerance) - means
            UCB_K[self.pulls < 1] = float('+inf')
        self.goods[means >= UCB_K] = True

    # --- SparseklUCB choice() method

    def choice(self):
        r""" Choose the next arm to play:

        - If still in a Round-Robin phase, play the next arm,
        - Otherwise, recompute the set :math:`\mathcal{J}(t)`,
        - If it is too small, if :math:`\mathcal{J}(t) < s`:
           + Start a new Round-Robin phase from arm 0.
        - Otherwise, recompute the second set :math:`\mathcal{K}(t)`,
        - If it is too small, if :math:`\mathcal{K}(t) < s`:
           + Play a Force-Log step by choosing an arm uniformly at random from the set :math:`\mathcal{J}(t) \setminus K(t)`.
        - Otherwise,
           + Play a UCB step by choosing an arm with highest KL-UCB index from the set :math:`\mathcal{K}(t)`.
        """
        # print("  At step t = {} a SparseklUCB algorithm was in phase {} ...".format(self.t, self.phase))  # DEBUG
        if (self.phase == Phase.RoundRobin) and ((1 + self.offset) < self.nbArms):
            # deterministic phase
            self.offset += 1
            return self.offset
        else:
            self.update_j()
            j = self.force_to_see
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
