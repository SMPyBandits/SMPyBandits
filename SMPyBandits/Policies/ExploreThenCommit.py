# -*- coding: utf-8 -*-
""" Different variants of the Explore-Then-Commit policy.

- Reference: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
- And [Kaufmann & Moy, 2017, ICC](http://icc2017.ieee-icc.org/program/tutorials#TT01), E.Kaufmann's slides at IEEE ICC 2017
- See also: https://github.com/SMPyBandits/SMPyBandits/issues/62

.. warning:: They don't work as well as the theory predicted...
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
from .EpsilonGreedy import EpsilonGreedy


#: Default value for the gap, 0.1 as in many basic experiments
GAP = 0.1


class ETC_KnownGap(EpsilonGreedy):
    r""" Variant of the Explore-Then-Commit policy, with known horizon :math:`T` and gap :math:`\Delta = \min_{i\neq j} \mu_i - \mu_j`.

    - Reference: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, horizon, gap=GAP,
                 lower=0., amplitude=1.):
        super(ETC_KnownGap, self).__init__(nbArms, epsilon=0.5, lower=lower, amplitude=amplitude)
        # Arguments
        assert horizon > 0, "Error: the 'horizon' parameter for ETC_KnownGap class has to be > 0."
        self.horizon = int(horizon) if horizon is not None else None  #: Parameter :math:`T` = known horizon of the experiment.
        assert 0 <= gap <= 1, "Error: the 'gap' parameter for ETC_KnownGap class has to be in [0, 1]."  # DEBUG
        self.gap = gap  #: Known gap parameter for the stopping rule.
        # Compute the time m
        m = int(np.floor(((2. / gap**2) * np.log(horizon * gap**2 / 2.))))
        self.max_t = self.nbArms * m  #: Time until exporation, ``m_`` steps in each arm.

    def __str__(self):
        return r"ETC_KnownGap($T={}$, $\Delta={:.3g}$, $T_0={}$)".format(self.horizon, self.gap, self.max_t)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):
        r""" 1 while :math:`t \leq T_0`, 0 after, where :math:`T_0` is defined by:

        .. math:: T_0 = \lfloor \frac{2}{\Delta^2} \log(\frac{T \Delta^2}{2}) \rfloor.
        """
        if self.t <= self.max_t:
            # First phase: randomly explore!
            return 1
        else:
            # Second phase: just exploit!
            return 0


class ETC_RandomStop(EpsilonGreedy):
    r""" Variant of the Explore-Then-Commit policy, with known horizon :math:`T` and random stopping time.

    - Reference: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, horizon,
                 lower=0., amplitude=1.):
        super(ETC_RandomStop, self).__init__(nbArms, epsilon=0.5, lower=lower, amplitude=amplitude)
        # Arguments
        assert horizon > 0, "Error: the 'horizon' parameter for ETC_RandomStop class has to be > 0."
        self.horizon = int(horizon) if horizon is not None else None  #: Parameter :math:`T` = known horizon of the experiment.
        self.stillRandom = True  #: Still randomly exploring?

    def __str__(self):
        return r"ETC_RandomStop($T={}$)".format(self.horizon)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):
        r""" 1 while :math:`t \leq \tau`, 0 after, where :math:`\tau` is a random stopping time, defined by:

        .. math:: \tau = \inf\{ t \in\mathbb{N},\; \min_{i \neq j} \| \widehat{X_i}(t) - \widehat{X_j}(t) \| > \sqrt{\frac{4 \log(T/t)}{t}} \}.
        """
        if np.min(self.pulls) > 0:
            means = self.rewards / self.pulls
            # smallestDiffMean = np.min(np.diff(np.sort(means)))
            smallestDiffMean = max([abs(mi - mj) for mi in means for mj in means if mi != mj])
            if smallestDiffMean > np.sqrt((4. * np.log(self.horizon / self.t)) / self.t):
                self.stillRandom = False
        # Done
        if self.stillRandom:
            # First phase: randomly explore!
            return 1
        else:
            # Second phase: just exploit!
            return 0
