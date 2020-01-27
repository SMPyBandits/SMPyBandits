# -*- coding: utf-8 -*-
""" The generic kl-UCB policy for one-parameter exponential distributions with restarted round count t_k.
By default, it assumes Bernoulli arms.
Note: using log(t) + c log(log(t)) for the KL-UCB index of just log(t)
- It is designed to be used with the wrapper :class:`GLR_UCB`.
- By default, it assumes Bernoulli arms.
- Reference: [Garivier & Cappé - COLT, 2011](https://arxiv.org/pdf/1102.2490.pdf).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from math import log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .kullback import klucbBern
    from .klUCBloglog import klUCBloglog
    from .klUCB_forGLR import klUCB_forGLR
except (ImportError, SystemError):
    from kullback import klucbBern
    from klUCBloglog import klUCBloglog
    from klUCB_forGLR import klUCB_forGLR

#: Default value for the constant c used in the computation of KL-UCB index.
c = 3  #: Default value when using :math:`f(t) = \log(t) + c \log(\log(t))`, as :class:`klUCB_forGLR` is inherited from :class:`klUCBloglog`.


#: Default value for the tolerance for computing numerical approximations of the kl-UCB indexes.
TOLERANCE = 1e-4


class klUCBloglog_forGLR(klUCB_forGLR):
    """ The generic KL-UCB policy for one-parameter exponential distributions, using a different exploration time step for each arm (:math:`\log(t_k) + c \log(\log(t_k))` instead of :math:`\log(t) + c \log(\log(t))`).

- It is designed to be used with the wrapper :class:`GLR_UCB`.
    - By default, it assumes Bernoulli arms.
    - Reference: [Garivier & Cappé - COLT, 2011](https://arxiv.org/pdf/1102.2490.pdf).
    """
    def __init__(self, nbArms, tolerance=TOLERANCE, klucb=klucbBern, c=2, lower=0., amplitude=1.):
        super(klUCBloglog_forGLR, self).__init__(nbArms, tolerance=tolerance, klucb=klucb, c=c, lower=lower, amplitude=amplitude)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            U_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{\log(t) + c \log(\max(1, \log(t)))}{N_k(t)} \right\},\\
            I_k(t) &= U_k(t).

        If rewards are in :math:`[a, b]` (default to :math:`[0, 1]`) and :math:`\mathrm{kl}(x, y)` is the Kullback-Leibler divergence between two distributions of means x and y (see :mod:`Arms.kullback`),
        and c is the parameter (default to 1).
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # XXX We could adapt tolerance to the value of self.t
            return self.klucb(self.rewards[arm] / self.pulls[arm], (log(self.t_for_each_arm[arm]) + self.c * log(max(1, log(self.t_for_each_arm[arm])))) / self.pulls[arm], self.tolerance)

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = self.klucb_vect(self.rewards / self.pulls, (np.log(self.t_for_each_arm) + self.c * np.log(np.maximum(1., np.log(self.t_for_each_arm)))) / self.pulls, self.tolerance)
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes