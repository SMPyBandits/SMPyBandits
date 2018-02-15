# -*- coding: utf-8 -*-
r""" The approximated Finite-Horizon Gittins index policy for bounded bandits.

- This is not the computationally costly Gittins index, but a simple approximation, using the knowledge of the horizon T.
- Reference: [Lattimore - COLT, 2016](http://www.jmlr.org/proceedings/papers/v49/lattimore16.pdf), and [his COLT presentation](https://youtu.be/p8AwKiudhZ4?t=276)
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .IndexPolicy import IndexPolicy

#: Default value for the parameter :math:`\alpha > 0` for ApproximatedFHGittins.
ALPHA = 0.125
ALPHA = 0.5

#: Default value for the parameter :math:`\tau \geq 1` that is used to artificially increase the horizon, from :math:`T` to :math`\tau T`.
DISTORTION_HORIZON = 1.01


class ApproximatedFHGittins(IndexPolicy):
    r""" The approximated Finite-Horizon Gittins index policy for bounded bandits.

    - This is not the computationally costly Gittins index, but a simple approximation, using the knowledge of the horizon T.
    - Reference: [Lattimore - COLT, 2016](http://www.jmlr.org/proceedings/papers/v49/lattimore16.pdf), and [his COLT presentation](https://youtu.be/p8AwKiudhZ4?t=276)
    """

    def __init__(self, nbArms, horizon=None,
                 alpha=ALPHA, distortion_horizon=DISTORTION_HORIZON,
                 lower=0., amplitude=1.):
        super(ApproximatedFHGittins, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert alpha > 0, "Error: parameter 'alpha' for ApproximatedFHGittins should be > 0."  # DEBUG
        self.alpha = alpha  #: Parameter :math:`\alpha > 0`.
        assert distortion_horizon >= 1, "Error: parameter 'distortion_horizon' for ApproximatedFHGittins should be >= 1."  # DEBUG
        self.distortion_horizon = distortion_horizon  #: Parameter :math:`\tau > 0`.
        self.horizon = int(horizon) if horizon is not None else None  #: Parameter :math:`T` = known horizon of the experiment.

    def __str__(self):
        if self.alpha == ALPHA:
            return r"ApprFHG($T={}$)".format(self.horizon)
        else:
            return r"ApprFHG($T={}$, $\alpha={:.3g}$)".format(self.horizon, self.alpha)

    @property
    def m(self):
        r""":math:`m = T - t + 1` is the number of steps to be played until end of the game.

        .. note:: The article does not explain how to deal with unknown horizon, but eventually if :math:`T` is wrong, this `m` becomes negative. Empirically, I force it to be :math:`\geq 1`, to not mess up with the :math:`\log(m)` used below, by using :math:`\tau T` instead of :math:`T` (e.g., :math:`\tau = 1.01` is enough to not ruin the performance in the last steps of the experiment).
        """
        return max((self.distortion_horizon * self.horizon) - self.t + 1, 1)
        # return max(self.horizon - self.t + 1, 1)

    # --- Computation

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

           I_k(t) &= \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{2 \alpha}{N_k(t)} \log\left( \frac{m}{N_k(t) \log^{1/2}\left( \frac{m}{N_k(t)} \right)} \right)}, \\
           \text{where}\;\; & m = T - t + 1.


        .. note:: This :math:`\log^{1/2}(\dots) = \sqrt(\log(\dots)))` term can be *undefined*, as soon as :math:`m < N_k(t)`, so empirically, :math:`\sqrt(\max(0, \log(\dots))` is used instead, or a larger horizon can be used to make :math:`m` artificially larger (e.g., :math:`T' = 1.1 T`).

        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            m_by_Nk = float(self.m) / self.pulls[arm]
            return (self.rewards[arm] / self.pulls[arm]) + np.sqrt((2. * self.alpha) / self.pulls[arm] * np.log(m_by_Nk / np.sqrt(np.log(m_by_Nk))))

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        m_by_Nk = float(self.m) / self.pulls
        indexes = (self.rewards / self.pulls) + np.sqrt((2. * self.alpha) / self.pulls * np.log(m_by_Nk / np.sqrt(np.maximum(0, np.log(m_by_Nk)))))
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
