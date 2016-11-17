# -*- coding: utf-8 -*-
'''The Bayes-UCB policy.
 Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012]'''

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann"
__version__ = "$Revision: 1.7 $"


from IndexPolicy import IndexPolicy
from Beta import Beta


class BayesUCB(IndexPolicy):
    """The Bayes-UCB.
      Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012].
    """

    def __init__(self, nbArms, posterior=Beta):
        self.nbArms = nbArms
        self.posterior = dict()
        for arm in range(self.nbArms):
            self.posterior[arm] = posterior()
        self.params = ''

    def startGame(self):
        self.t = 1
        for arm in range(self.nbArms):
            self.posterior[arm].reset()

    def getReward(self, arm, reward):
        self.posterior[arm].update(reward)
        self.t += 1

    def computeIndex(self, arm):
        return self.posterior[arm].quantile(1-1./self.t)

    def __str__(self):
        return "BayesUCB"
