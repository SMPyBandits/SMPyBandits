# -*- coding: utf-8 -*-
""" The DMED policy of [Honda & Takemura, COLT 2010] in the special case of Bernoulli rewards (can be used on any [0,1]-valued rewards, but warning: in the non-binary case, this is not the algorithm of [Honda & Takemura, COLT 2010]) (see note below on the variant).

- Reference: [Garivier & Cappé - COLT, 2011](https://arxiv.org/pdf/1102.2490.pdf).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Olivier Cappé, Aurélien Garivier, Lilian Besson"
__version__ = "0.6"


import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!


from .kullback import klBern
from .BasePolicy import BasePolicy


#: Variant: with this set to false, use a less aggressive list pruning criterion corresponding to the version called DMED in [Garivier & Cappé, COLT 2011]; the default is the original proposal of [Honda & Takemura, COLT 2010] (called DMED+ in [Garivier & Cappé, COLT 2011])
GENUINE = False
GENUINE = True


class DMED(BasePolicy):
    """ The DMED policy of [Honda & Takemura, COLT 2010] in the special case of Bernoulli rewards (can be used on any [0,1]-valued rewards, but warning: in the non-binary case, this is not the algorithm of [Honda & Takemura, COLT 2010]) (see note below on the variant).

    - Reference: [Garivier & Cappé - COLT, 2011](https://arxiv.org/pdf/1102.2490.pdf).
    """

    def __init__(self, nbArms, genuine=GENUINE, tolerance=1e-4, kl=klBern, lower=0., amplitude=1.):
        super(DMED, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.kl = np.vectorize(kl)  #: kl function to use
        self.kl.__name__ = kl.__name__
        self.tolerance = tolerance  #: Numerical tolerance
        self.genuine = genuine  #: Flag to know which variant is implemented, DMED or DMED+
        self.nextActions = list(range(nbArms))  #: List of next actions to play, every next step is playing ``nextActions.pop(0)``

    def __str__(self):
        return r"DMED{}({})".format("$^+$" if self.genuine else "", self.kl.__name__[2:])

    def startGame(self):
        """ Initialize the policy for a new game."""
        super(DMED, self).startGame()
        self.nextActions = list(range(self.nbArms))  # Force exploring the initial actions
        np.random.shuffle(self.nextActions)  # In a random order,

    def choice(self):
        r""" If there is still a next action to play, pop it and play it, otherwise make new list and play first action.

        The list of action is obtained as all the indexes :math:`k` satisfying the following equation.

        - For the naive version (``genuine = False``), DMED:

        .. math::

           \mathrm{kl}(\hat{\mu}_k(t), \hat{\mu}^*(t)) < \frac{\log(t)}{N_k(t)}.


        - For the original version (``genuine = True``), DMED+:

        .. math::

           \mathrm{kl}(\hat{\mu}_k(t), \hat{\mu}^*(t)) < \frac{\log(\frac{t}{N_k(t)})}{N_k(t)}.


        Where :math:`X_k(t)` is the sum of rewards from arm k, :math:`\hat{\mu}_k(t)` is the empirical mean,
        and :math:`\hat{\mu}^*(t)` is the best empirical mean.

        .. math::

           X_k(t) &= \sum_{\sigma=1}^{t} 1(A(\sigma) = k) r_k(\sigma) \\
           \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
           \hat{\mu}^*(t) &= \max_{k=1}^{K} \hat{\mu}_k(t)
        """
        if len(self.nextActions) == 0:
            empiricalMeans = self.rewards / self.pulls
            bestEmpiricalMean = np.max(empiricalMeans)
            if self.genuine:
                self.nextActions = np.nonzero(self.pulls * self.kl(empiricalMeans, bestEmpiricalMean) < np.log(self.t / self.pulls))[0]
            else:
                self.nextActions = np.nonzero(self.pulls * self.kl(empiricalMeans, bestEmpiricalMean) < np.log(self.t))[0]
            self.nextActions = list(self.nextActions)
        # Play next action
        return self.nextActions.pop(0)

    def choiceMultiple(self, nb=1):
        """ If there is still enough actions to play, pop them and play them, otherwise make new list and play nb first actions."""
        choices = [self.choice() for _ in range(nb)]
        assert len(set(choices)) == nb, "Error: choiceMultiple({}) for DMED policy does not work yet...".format(nb)  # DEBUG
        # while len(set(choices)) < nb:  # Not enough different actions? Try again.
        #     choices = [self.choice() for _ in range(nb)]
        return choices


class DMEDPlus(DMED):
    """ The DMED+ policy of [Honda & Takemura, COLT 2010] in the special case of Bernoulli rewards (can be used on any [0,1]-valued rewards, but warning: in the non-binary case, this is not the algorithm of [Honda & Takemura, COLT 2010]).

    - Reference: [Garivier & Cappé - COLT, 2011](https://arxiv.org/pdf/1102.2490.pdf).
    """

    def __init__(self, nbArms, tolerance=1e-4, kl=klBern, lower=0., amplitude=1.):
        super(DMEDPlus, self).__init__(nbArms, genuine=True, tolerance=tolerance, kl=kl, lower=lower, amplitude=amplitude)
