# -*- coding: utf-8 -*-
""" The Boltzmann Exploration (Softmax) index policy.
Reference: [http://www.cs.mcgill.ca/~vkules/bandits.pdf §2.1].
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np
from .BasePolicy import BasePolicy

temperature = 1


class Softmax(BasePolicy):
    """The Boltzmann Exploration (Softmax) index policy.
    Reference: [http://www.cs.mcgill.ca/~vkules/bandits.pdf §2.1].
    """

    def __init__(self, nbArms, temperature=temperature, lower=0., amplitude=1.):
        super(Softmax, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert temperature > 0, "Error: the temperature parameter for Softmax class has to be > 0."
        self.temperature = temperature

    def __str__(self):
        return "Softmax(temp: {})".format(self.temperature)

    def choice(self):
        # Force to first visit each arm once in the first steps
        if self.t < self.nbArms:
            arm = self.t % self.nbArms  # TODO random permutation instead of deterministic order!
        else:
            trusts = np.exp(self.rewards / (self.temperature * self.pulls))
            trusts /= np.sum(trusts)
            arm = np.random.choice(self.nbArms, p=trusts)
        # self.pulls[arm] += 1  # XXX why is it here?
        return arm

    def choiceWithRank(self, rank=1):
        # Force to first visit each arm once in the first steps
        if self.t < self.nbArms:
            arm = self.t % self.nbArms  # TODO random permutation instead of deterministic order!
        else:
            trusts = np.exp(self.rewards / (self.temperature * self.pulls))
            trusts /= np.sum(trusts)
            arms = np.random.choice(self.nbArms, size=rank, replace=False, p=trusts)
            arm = arms[rank - 1]
        return arm
