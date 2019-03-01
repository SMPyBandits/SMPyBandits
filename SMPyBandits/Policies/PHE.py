# -*- coding: utf-8 -*-
""" The PHE, Perturbed-History Exploration, policy for bounded bandits.

- Reference: [[Perturbed-History Exploration in Stochastic Multi-Armed Bandits, by Branislav Kveton, Csaba Szepesvari, Mohammad Ghavamzadeh, Craig Boutilier, 26 Feb 2019, arXiv:1902.10089]](https://arxiv.org/abs/1902.10089)
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

try:
    from .IndexPolicy import IndexPolicy
except ImportError:
    from IndexPolicy import IndexPolicy

from math import ceil
import numpy as np

#: By default, :math:`a` the perturbation scale in PHE is 1, that is, at current time step t, if there is :math:`s = T_{i,t-1}` samples of arm i, PHE generates :math:`s` pseudo-rewards (of mean :math:`1/2`)
DEFAULT_PERTURBATION_SCALE = 1.0


class PHE(IndexPolicy):
    """ The PHE, Perturbed-History Exploration, policy for bounded bandits.

    - Reference: [[Perturbed-History Exploration in Stochastic Multi-Armed Bandits, by Branislav Kveton, Csaba Szepesvari, Mohammad Ghavamzadeh, Craig Boutilier, 26 Feb 2019, arXiv:1902.10089]](https://arxiv.org/abs/1902.10089)

    - They prove that PHE achieves a regret of :math:`\mathcal{O}(K \Delta^{-1} \log(T))` regret for horizon :math:`T`, and if :math:`\Delta` is the minimum gap between the expected rewards of the optimal and suboptimal arms, for :math:`a > 1`.
    - Note that the limit case of :math:`a=0` gives the Follow-the-Leader algorithm (FTL), known to fail.
    """
    def __init__(self, nbArms, perturbation_scale=DEFAULT_PERTURBATION_SCALE, lower=0., amplitude=1.):
        assert perturbation_scale > 0, "Error: for PHE class, the parameter perturbation_scale should be > 0, it was {}.".format(perturbation_scale)  # DEBUG
        self.perturbation_scale = perturbation_scale  #: Perturbation scale, denoted :math:`a` in their paper. Should be a float or int number. With :math:`s` current samples, :math:`\lceil a s \rceil` additional pseudo-rewards are generated.
        super(PHE, self).__init__(nbArms, lower=lower, amplitude=amplitude)

    def __str__(self):
        return r"PHE($a={:.3g}$)".format(self.perturbation_scale)

    def computeIndex(self, arm):
        """ Compute a randomized index by adding :math:`a` pseudo-rewards (of mean :math:`1/2`) to the current observations of this arm."""
        s = self.pulls[arm]
        if s <= 0:
            return float('+inf')
        V_is = self.rewards[arm]
        number_of_perturbation = ceil(self.perturbation_scale * s)
        U_is = np.random.binomial(number_of_perturbation, 0.5)
        perturbated_mean = (V_is + U_is) / (s + number_of_perturbation)
        return perturbated_mean
