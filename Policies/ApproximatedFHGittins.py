# -*- coding: utf-8 -*-
r""" The approximated Finite-Horizon Gittins index policy for bounded bandits.

- This is not the computationally costly Gittins index, but a simple approximation, using the knowledge of the horizon T.
- Reference: [Lattimore - COLT, 2016](http://www.jmlr.org/proceedings/papers/v49/lattimore16.pdf), and [his COLT presentation](https://youtu.be/p8AwKiudhZ4?t=276)
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

from math import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .IndexPolicy import IndexPolicy

#: Default value for the parameter :math:`\alpha > 0` for ApproximatedFHGittins.
ALPHA = 4


class ApproximatedFHGittins(IndexPolicy):
    r""" The approximated Finite-Horizon Gittins index policy for bounded bandits.

    - This is not the computationally costly Gittins index, but a simple approximation, using the knowledge of the horizon T.
    - Reference: [Lattimore - COLT, 2016](http://www.jmlr.org/proceedings/papers/v49/lattimore16.pdf), and [his COLT presentation](https://youtu.be/p8AwKiudhZ4?t=276)
    """

    def __init__(self, nbArms, horizon=None, alpha=ALPHA, lower=0., amplitude=1.):
        super(ApproximatedFHGittins, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.alpha = alpha  #: Parameter :math:`\alpha > 0`.
        self.horizon = horizon  #: Constant parameter T = horizon of the experiment.

    def __str__(self):
        return r"ApproximatedFHGittins($\alpha={:.3g}$)".format(self.alpha)

    @property
    def m(self):
        r""":math:`m = T - t + 1` is the number of steps to be played until end of the game."""
        return self.horizon - self.t + 1

    # --- Computation

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

           I_k(t) &= \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{\alpha}{2 N_k(t)} \log\left( \frac{m}{N_k(t) \log^{1/2}\left( \frac{m}{N_k(t)} \right)} \right)}, \\
           \text{where}\;\; & m = T - t + 1.


        .. note:: This :math:`\log^{1/2}(\dots) = \sqrt(\log(\dots)))` term can be *undefined*, as soon as :math:`m < N_k(t)`, so empirically, :math:`\sqrt(\max(0, \log(\dots))` is used instead, or a larger horizon can be used to make :math:`m` artificially larger (e.g., :math:`T' = 1.1 T`).

        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            m_by_Nk = float(self.m) / self.pulls[arm]
            loghalf = sqrt(max(0, log(m_by_Nk)))
            return (self.rewards[arm] / self.pulls[arm]) + sqrt(self.alpha / (2. * self.pulls[arm]) * log(m_by_Nk / loghalf))

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        m_by_Nk = float(self.m) / self.pulls
        loghalf = np.sqrt(np.maximum(0, np.log(m_by_Nk)))
        indexes = (self.rewards / self.pulls) + np.sqrt(self.alpha / (2. * self.pulls) * np.log(m_by_Nk / loghalf))
        indexes[self.pulls < 1] = float('+inf')
        self.index = indexes
