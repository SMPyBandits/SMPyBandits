# -*- coding: utf-8 -*-
""" The EXP3++ randomized index policy, an improved version of the EXP3 policy.

Reference: [[One practical algorithm for both stochastic and adversarial bandits, S.Seldin & A.Slivkins, ICML, 2014](http://www.jmlr.org/proceedings/papers/v32/seldinb14-supp.pdf)].

See also [[An Improved Parametrization and Analysis of the EXP3++ Algorithm for Stochastic and Adversarial Bandits, by Y.Seldin & G.Lugosi, COLT, 2017](https://arxiv.org/pdf/1702.06103)].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.7"

import numpy as np
import numpy.random as rn
from .BasePolicy import BasePolicy


#: Value for the :math:`\alpha` parameter.
ALPHA = 3


#: Value for the :math:`\beta` parameter.
BETA = 256


class Exp3PlusPlus(BasePolicy):
    """ The EXP3++ randomized index policy, an improved version of the EXP3 policy.

    Reference: [[One practical algorithm for both stochastic and adversarial bandits, S.Seldin & A.Slivkins, ICML, 2014](http://www.jmlr.org/proceedings/papers/v32/seldinb14-supp.pdf)].

    See also [[An Improved Parametrization and Analysis of the EXP3++ Algorithm for Stochastic and Adversarial Bandits, by Y.Seldin & G.Lugosi, COLT, 2017](https://arxiv.org/pdf/1702.06103)].
    """

    def __init__(self, nbArms, alpha=ALPHA, beta=BETA,
                 lower=0., amplitude=1.):
        super(Exp3PlusPlus, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.alpha = alpha  #: :math:`\alpha` parameter for computations of :math:`\xi_t(a)`.
        self.beta = beta  #: :math:`\beta` parameter for computations of :math:`\xi_t(a)`.
        # Internal memory
        self.weights = np.full(nbArms, 1. / nbArms)  #: Weights on the arms
        self.losses = np.zeros(nbArms)  #: Cumulative sum of losses estimates for each arm
        self.unweighted_losses = np.zeros(nbArms)  #: Cumulative sum of unweighted losses for each arm
        # trying to randomize the order of the initial visit to each arm; as this determinism breaks its habitility to play efficiently in multi-players games
        # XXX do even more randomized, take a random permutation of the arm ?
        self._initial_exploration = rn.permutation(nbArms)
        # The proba that another player has the same is nbPlayers / factorial(nbArms) : should be SMALL !

    def startGame(self):
        """Start with uniform weights."""
        super(Exp3PlusPlus, self).startGame()
        self.weights.fill(1. / self.nbArms)
        self.losses.fill(0)

    def __str__(self):
        s = "{}{}".format("" if self.alpha == ALPHA else r"$\alpha={}$".format(self.alpha), "" if self.beta == BETA else r"$\beta={}$".format(self.beta))
        return r"Exp3++{}".format("({})".format(s) if s else "")

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def eta(self):
        r"""Decreasing sequence of learning rates, given by :math:`\eta_t = \frac{1}{2} \sqrt{\frac{\log K}{t K}}`."""
        return 0.5 * np.sqrt(np.log(self.nbArms) / float(self.t * self.nbArms))

    @property
    def gap_estimate(self):
        r"""Compute the gap estimate :math:`\widehat{\Delta}^{\mathrm{LCB}}_t(a)` from :

        - Compute the UCB: :math:`\mathrm{UCB}_t(a) = \min\left( 1, \frac{wide\hat{L}_{t-1}(a)}{N_{t-1}(a)} + \sqrt{\frac{a \log(t K^{1/\alpha})}{2 N_{t-1}(a)}} \right)`,
        - Compute the LCB: :math:`\mathrm{LCB}_t(a) = \max\left( 0, \frac{wide\hat{L}_{t-1}(a)}{N_{t-1}(a)} - \sqrt{\frac{a \log(t K^{1/\alpha})}{2 N_{t-1}(a)}} \right)`,
        - Then the gap: :math:`\widehat{\Delta}^{\mathrm{LCB}}_t(a) = \max\left( 0, \mathrm{LCB}_t(a) - \min_{a'} \mathrm{UCB}_t(a') \right)`.
        - The gap should be in :math:`[0, 1]`.
        """
        average_losses = self.unweighted_losses / self.pulls
        exploration_term = np.sqrt((self.alpha * np.log(self.t * self.nbArms**(1./self.alpha))) / (2 * self.pulls))
        UCB = np.minimum(1, average_losses + exploration_term)
        min_UCB = np.min(UCB)
        LCB = np.maximum(0, average_losses - exploration_term)
        Delta = np.maximum(0, LCB - min_UCB)
        assert np.min(Delta) >= 0 and np.max(Delta) <= 1, "Error: a gap estimate of Delta = {} was found to be outside of [0, 1].".format(Delta)  # DEBUG
        return Delta

    @property
    def xi(self):
        r"""Compute the :math:`\xi_t(a) = \frac{\beta \log t}{t \widehat{\Delta}^{\mathrm{LCB}}_t(a)^2}` vector of indexes."""
        return self.beta * np.log(self.t) / (self.t * (self.gap_estimate ** 2))

    @property
    def epsilon(self):
        r"""Compute the vector of parameters :math:`\eta_t(a) = \min\left(\frac{1}{2 K}, \frac{1}{2} \sqrt{\frac{\log K}{t K}}, \xi_t(a) \right)`."""
        return np.minimum(
            0.5 / self.nbArms,
            0.5 * np.sqrt(np.log(self.nbArms) / (self.t * self.nbArms)),
            self.xi
        )

    @property
    def trusts(self):
        r"""Update the trusts probabilities according to Exp3PlusPlus formula, and the parameter :math:`\eta_t`.

        .. math::

           \tilde{\rho}'_{t+1}(a) &= (1 - \sum_{a'=1}^{K}\eta_t(a')) w_t(a) + \eta_t(a), \\
           \tilde{\rho}_{t+1} &= \tilde{\rho}'_{t+1} / \sum_{a=1}^{K} \tilde{\rho}'_{t+1}(a).

        If :math:`rho_t(a)` is the current weight from arm a.
        """
        # Mixture between the weights and the uniform distribution
        trusts = ((1 - np.sum(self.eta)) * self.weights) + self.eta
        # XXX Handle weird cases, slow down everything but safer!
        if not np.all(np.isfinite(trusts)):
            print("WARNING: Exp3PlusPlus.trusts : a trust was infinite...")  # DEBUG
            # XXX some value has non-finite trust, probably on the first steps
            # 1st case: all values are non-finite (nan): set trusts to 1/N uniform choice
            if np.all(~np.isfinite(trusts)):
                trusts = np.full(self.nbArms, 1. / self.nbArms)
            # 2nd case: only few values are non-finite: set them to 0
            else:
                trusts[~np.isfinite(trusts)] = 0
        # Bad case, where the sum is so small that it's only rounding errors
        if np.isclose(np.sum(trusts), 0):
            print("WARNING: Exp3PlusPlus.trusts : the sum of trusts was too close to zero, reinitializing!")  # DEBUG
            trusts = np.full(self.nbArms, 1. / self.nbArms)
        # Normalize it and return it
        return trusts / np.sum(trusts)

    def getReward(self, arm, reward):
        r"""Give a reward: accumulate losses on that arm a, then update the weight :math:`\rho_t(a)` and renormalize the weights.

        - Divide by the trust on that arm a, i.e., the probability of observing arm a: :math:`\tilde{l}_t(a) = \frac{l_t(a)}{\tilde{\rho}_t(a)} 1(A_t = a)`.
        - Add this loss to the cumulative loss: :math:`\tilde{L}_t(a) := \tilde{L}_{t-1}(a) + \tilde{l}_t(a)`.
        - But the un-weighted loss is added to the other cumulative loss: :math:`\widehat{L}_t(a) := \widehat{L}_{t-1}(a) + l_t(a) 1(A_t = a)`.

        .. math::

           \rho'_{t+1}(a) &= \exp\left( - \tilde{L}_t(a) \eta_t \right) \\
           \rho_{t+1} &= \rho'_{t+1} / \sum_{a=1}^{K} \rho'_{t+1}(a).
        """
        super(Exp3PlusPlus, self).getReward(arm, reward)  # XXX Call to IndexPolicy
        # Compute loss estimate
        reward = (reward - self.lower) / self.amplitude
        loss = 1 - reward
        self.unweighted_losses[arm] += loss
        loss = loss / self.trusts[arm]
        self.losses[arm] += loss
        # Update weight of THIS arm, with this biased or unbiased loss estimate, but we need to compute again ALL losses!
        # self.weights[arm] = np.exp(- self.eta * self.losses[arm])
        self.weights = np.exp(- self.eta * self.losses)
        # Renormalize weights at each step
        self.weights /= np.sum(self.weights)

    # --- Choice methods

    def choice(self):
        """One random selection, with probabilities = trusts, thank to :func:`numpy.random.choice`."""
        # Force to first visit each arm once in the first steps
        if self.t < self.nbArms:
            # DONE we could use a random permutation instead of deterministic order!
            return self._initial_exploration[self.t]
        else:
            return rn.choice(self.nbArms, p=self.trusts)

    def choiceWithRank(self, rank=1):
        """Multiple (rank >= 1) random selection, with probabilities = trusts, thank to :func:`numpy.random.choice`, and select the last one (less probable).

        - Note that if not enough entries in the trust vector are non-zero, then :func:`choice` is called instead (rank is ignored).
        """
        if (self.t < self.nbArms) or (rank == 1) or np.sum(~np.isclose(self.trusts, 0)) < rank:
            return self.choice()
        else:
            return rn.choice(self.nbArms, size=rank, replace=False, p=self.trusts)[rank - 1]

    def choiceFromSubSet(self, availableArms='all'):
        """One random selection, from availableArms, with probabilities = trusts, thank to :func:`numpy.random.choice`."""
        if (self.t < self.nbArms) or (availableArms == 'all') or (len(availableArms) == self.nbArms):
            return self.choice()
        else:
            return rn.choice(availableArms, p=self.trusts[availableArms])

    def choiceMultiple(self, nb=1):
        """Multiple (nb >= 1) random selection, with probabilities = trusts, thank to :func:`numpy.random.choice`."""
        return rn.choice(self.nbArms, size=nb, replace=False, p=self.trusts)

    # --- Other methods

    def estimatedOrder(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing trust probabilities."""
        return np.argsort(self.trusts)

    def estimatedBestArms(self, M=1):
        """ Return a (non-necessarily sorted) list of the indexes of the M-best arms. Identify the set M-best."""
        assert 1 <= M <= self.nbArms, "Error: the parameter 'M' has to be between 1 and K = {}, but it was {} ...".format(self.nbArms, M)  # DEBUG
        order = self.estimatedOrder()
        return order[-M:]
