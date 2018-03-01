# -*- coding: utf-8 -*-
""" The SparseWrapper policy, designed to tackle sparse stochastic bandit problems:

- This means that only a small subset of size ``s`` of the ``K`` arms has non-zero means.
- The SparseWrapper algorithm requires to known **exactly** the value of ``s``.

- This SparseWrapper is a very generic version, and can use *any index policy* for both the decision in the UCB phase and the construction of the sets :math:`\mathcal{J}(t)` and :math:`\mathcal{K}(t)`.
- The usual UCB indexes can be used for the set :math:`\mathcal{K}(t)` by setting the flag ``use_ucb_for_set_K`` to true (but usually the indexes from the underlying policy can be used efficiently for set :math:`\mathcal{K}(t)`), and for the set :math:`\mathcal{J}(t)` by setting the flag ``use_ucb_for_set_J`` to true (as its formula is less easily generalized).
- If used with :class:`Policy.UCBalpha` or :class:`Policy.klUCB`, it should be better to use directly :class:`Policy.SparseUCB` or :class:`Policy.SparseklUCB`.

- Reference: [["Sparse Stochastic Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)] who introduced SparseUCB.

.. warning::

   This is very EXPERIMENTAL! No proof yet!
   But it works fine!!
"""

__author__ = "Lilian Besson"
__version__ = "0.7"

from math import sqrt, log
from enum import Enum  # For the different states
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .BasePolicy import BasePolicy
from .UCBalpha import UCBalpha

#: Default horizon-dependent policy
default_index_policy = UCBalpha


#: Different states during the SparseWrapper algorithm.
#:
#: - ``RoundRobin`` means all are sampled once.
#: - ``ForceLog`` uniformly explores arms that are in the set :math:`\mathcal{J}(t) \setminus \mathcal{K}(t)`.
#: - ``UCB`` is the phase that the algorithm should converge to, when a normal UCB selection is done only on the "good" arms, i.e., :math:`\mathcal{K}(t)`.
Phase = Enum('Phase', ['RoundRobin', 'ForceLog', 'UCB'])


#: Default value for the flag controlling whether the usual UCB indexes are used for the set :math:`\mathcal{K}(t)`.
#: Default it to use the indexes of the underlying policy, which could be more efficient.
USE_UCB_FOR_SET_K = True
USE_UCB_FOR_SET_K = False


#: Default value for the flag controlling whether the usual UCB indexes are used for the set :math:`\mathcal{J}(t)`.
#: Default it to use the UCB indexes as there is no clean and generic formula to obtain the indexes for :math:`\mathcal{J}(t)` from the indexes of the underlying policy.
#: Note that I found a formula, it's just durty. See below.
USE_UCB_FOR_SET_J = True
USE_UCB_FOR_SET_J = False

#: Default parameter for :math:`\alpha` for the UCB indexes.
ALPHA = 1


# --- The interesting class


class SparseWrapper(BasePolicy):
    """ The SparseWrapper policy, designed to tackle sparse stochastic bandit problems.

    - By default, assume ``sparsity`` = ``nbArms``.
    """

    def __init__(self, nbArms, sparsity=None,
                 use_ucb_for_set_K=USE_UCB_FOR_SET_K,
                 use_ucb_for_set_J=USE_UCB_FOR_SET_J,
                 alpha=ALPHA,
                 policy=default_index_policy,
                 lower=0., amplitude=1.,
                 *args, **kwargs):
        super(SparseWrapper, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        if sparsity is None:
            sparsity = nbArms
            print("Warning: regular klUCB should be used instead of SparseWrapper if 'sparsity' = 'nbArms' = {} ...".format(nbArms))  # DEBUG
        assert 1 <= sparsity <= nbArms, "Error: 'sparsity' has to be in [1, nbArms = {}] but was {} ...".format(nbArms, sparsity)  # DEBUG
        self.sparsity = sparsity  #: Known value of the sparsity of the current problem.
        self.use_ucb_for_set_K = use_ucb_for_set_K  #: Whether the usual UCB indexes are used for the set :math:`\mathcal{K}(t)`.
        self.use_ucb_for_set_J = use_ucb_for_set_J  #: Whether the usual UCB indexes are used for the set :math:`\mathcal{J}(t)`.
        self.alpha = alpha  #: Parameter :math:`\alpha` for the UCB indexes for the two sets, if not using the indexes of the underlying policy.
        self.phase = Phase.RoundRobin  #: Current phase of the algorithm.
        # --- Policy
        self._policy = policy  # Class to create the underlying policy
        self._args = args  # To keep them
        if 'params' in kwargs:
            kwargs.update(kwargs['params'])
            del kwargs['params']
        self._kwargs = kwargs  # To keep them
        self.policy = None  #: Underlying policy
        # --- internal memory
        self.force_to_see = np.full(nbArms, True)  #: Binary array for the set :math:`\mathcal{J}(t)`.
        self.goods = np.full(nbArms, True)  #: Binary array for the set :math:`\mathcal{K}(t)`.
        self.offset = -1  #: Next arm to sample, for the Round-Robin phase
        self.startGame()  # XXX Force it, for pretty printing...

    # --- pretty printing

    def __str__(self):
        ucb_for = ""
        if self.use_ucb_for_set_K or self.use_ucb_for_set_J:
            ucb_for = ", UCB for "
        if self.use_ucb_for_set_J and self.use_ucb_for_set_K:
            ucb_for += "K and J"
        elif self.use_ucb_for_set_K and not self.use_ucb_for_set_J:
            ucb_for += "K"
        elif self.use_ucb_for_set_J and not self.use_ucb_for_set_K:
            ucb_for += "J"
        return r"SparseWrapper(s={})[{}{}]".format(self.sparsity, self.policy, ucb_for)

    # --- Start game by creating new underlying policy

    def startGame(self):
        """ Initialize the policy for a new game."""
        super(SparseWrapper, self).startGame()
        self.phase = Phase.RoundRobin
        self.force_to_see.fill(True)  # faster than sets
        self.goods.fill(True)  # faster than sets
        self.offset = -1
        # now for the underlying policy
        self.policy = self._policy(self.nbArms, lower=self.lower, amplitude=self.amplitude, *self._args, **self._kwargs)
        # now also start game for the underlying policy
        self.policy.startGame()

    # --- Pass the call to the subpolicy

    def getReward(self, arm, reward):
        """ Pass the reward, as usual, update t and sometimes restart the underlying policy."""
        # print(" - At time t = {}, got a reward = {} from arm {} ...".format(self.t, arm, reward))  # DEBUG
        super(SparseWrapper, self).getReward(arm, reward)
        self.policy.getReward(arm, reward)

    # --- Update the two sets

    def update_j(self):
        r""" Recompute the set :math:`\mathcal{J}(t)`:

        .. math::

           \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
           U^{\mathcal{K}}_k(t) &= I_k^{P}(t) - \hat{\mu}_k(t),\\
           U^{\mathcal{J}}_k(t) &= U^{\mathcal{K}}_k(t) \times \sqrt{\frac{\log(N_k(t))}{\log(t)}},\\
           \mathcal{J}(t) &= \left\{ k \in [1,...,K]\;, \hat{\mu}_k(t) \geq U^{\mathcal{J}}_k(t) - \hat{\mu}_k(t) \right\}.

        - Yes, this is a nothing but a *hack*, as there is no generic formula to retrieve the indexes used in the set :math:`\mathcal{J}(t)` from the indexes :math:`I_k^{P}(t)` of the underlying index policy :math:`P`.
        - If ``use_ucb_for_set_J`` is ``True``, the same formula from :class:`Policies.SparseUCB` is used.
        """
        # assert np.all(self.pulls >= 1), "Error: at least one arm was not already pulled: pulls = {} ...".format(self.pulls)  # DEBUG
        self.force_to_see.fill(False)  # faster than sets
        means = self.rewards / self.pulls
        means[self.pulls < 1] = float('+inf')
        if self.use_ucb_for_set_J:
            UCB_J = np.sqrt((self.alpha * np.log(self.pulls)) / self.pulls)
            UCB_J[self.pulls < 1] = float('+inf')
        else:
            self.computeAllIndex()
            UCB_K = self.index - means
            # FIXME hack to convert it to the UCB_J
            UCB_J = np.sqrt( np.log(self.pulls) / np.log(self.t) ) * UCB_K
            UCB_J[self.pulls < 1] = float('+inf')
        self.force_to_see[means >= UCB_J] = True

    def update_k(self):
        r""" Recompute the set :math:`\mathcal{K}(t)`:

        .. math::

           \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
           U^{\mathcal{K}}_k(t) &= I_k^{P}(t) - \hat{\mu}_k(t),\\
           \mathcal{J}(t) &= \left\{ k \in [1,...,K]\;, \hat{\mu}_k(t) \geq U^{\mathcal{K}}_k(t) - \hat{\mu}_k(t) \right\}.

        - If ``use_ucb_for_set_K`` is ``True``, the same formula from :class:`Policies.SparseUCB` is used.
        """
        # assert np.all(self.pulls >= 1), "Error: at least one arm was not already pulled: pulls = {} ...".format(self.pulls)  # DEBUG
        self.goods.fill(False)  # faster than sets
        means = self.rewards / self.pulls
        means[self.pulls < 1] = float('+inf')
        if self.use_ucb_for_set_K:
            UCB_K = np.sqrt((self.alpha * np.log(self.t)) / self.pulls)
            UCB_K[self.pulls < 1] = float('+inf')
        else:
            self.computeAllIndex()
            UCB_K = self.index - means
        self.goods[means >= UCB_K] = True

    # --- SparseWrapper choice() method

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
           + Play a UCB step by choosing an arm with highest index (from the underlying policy) from the set :math:`\mathcal{K}(t)`.
        """
        # print("  At step t = {} a SparseWrapper algorithm was in phase {} ...".format(self.t, self.phase))  # DEBUG
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

    # --- Sub methods

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def index(self):
        r""" Get attribute ``index`` from the underlying policy."""
        return self.policy.index

    def choiceWithRank(self, rank=1):
        r""" Pass the call to ``choiceWithRank`` of the underlying policy."""
        return self.policy.choiceWithRank(rank=rank)

    def choiceFromSubSet(self, availableArms='all'):
        r""" Pass the call to ``choiceFromSubSet`` of the underlying policy."""
        return self.policy.choiceFromSubSet(availableArms=availableArms)

    def choiceMultiple(self, nb=1):
        r""" Pass the call to ``choiceMultiple`` of the underlying policy."""
        return self.policy.choiceMultiple(nb=nb)

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        r""" Pass the call to ``choiceIMP`` of the underlying policy."""
        return self.policy.choiceIMP(nb=nb, startWithChoiceMultiple=startWithChoiceMultiple)

    def estimatedOrder(self):
        r""" Pass the call to ``estimatedOrder`` of the underlying policy."""
        return self.policy.estimatedOrder()

    def estimatedBestArms(self, M=1):
        r""" Pass the call to ``estimatedBestArms`` of the underlying policy."""
        return self.policy.estimatedBestArms(M=M)

    def computeIndex(self, arm):
        r""" Pass the call to ``computeIndex`` of the underlying policy."""
        return self.policy.computeIndex(arm)

    def computeAllIndex(self):
        r""" Pass the call to ``computeAllIndex`` of the underlying policy."""
        return self.policy.computeAllIndex()
