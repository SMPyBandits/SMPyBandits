# -*- coding: utf-8 -*-
r""" The Drift-Detection algorithm for non-stationary bandits.

- Reference: [["EXP3 with Drift Detection for the Switching Bandit Problem", Robin Allesiardo & Raphael Feraud]](https://www.researchgate.net/profile/Allesiardo_Robin/publication/281028960_EXP3_with_Drift_Detection_for_the_Switching_Bandit_Problem/links/55d1927808aee19936fdac8e.pdf)
- It runs on top of a simple policy like :class:`Exp3`, and :class:`DriftDetection_IndexPolicy` is a wrapper:

    >>> policy = DriftDetection_IndexPolicy(nbArms, C=1)
    >>> # use policy as usual, with policy.startGame(), r = policy.choice(), policy.getReward(arm, r)

- It uses an additional :math:`\mathcal{O}(\tau_\max)` memory for a game of maximum stationary length :math:`\tau_\max`.

.. warning:: It works on :class:`Exp3` or other parametrizations of the Exp3 policy, e.g., :class:`Exp3PlusPlus`.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


import numpy as np
from math import log, sqrt

try:
    from .CD_UCB import CD_IndexPolicy
    from .Exp3 import Exp3
    from .Exp3PlusPlus import Exp3PlusPlus
except ImportError:
    from CD_UCB import CD_IndexPolicy
    from Exp3 import Exp3
    from Exp3PlusPlus import Exp3PlusPlus


VERBOSE = True
#: Whether to be verbose when doing the search for valid parameter :math:`\ell`.
VERBOSE = False


CONSTANT_C = 1.0  #: The constant :math:`C` used in Corollary 1 of paper [["EXP3 with Drift Detection for the Switching Bandit Problem", Robin Allesiardo & Raphael Feraud]](https://www.researchgate.net/profile/Allesiardo_Robin/publication/281028960_EXP3_with_Drift_Detection_for_the_Switching_Bandit_Problem/links/55d1927808aee19936fdac8e.pdf).


class DriftDetection_IndexPolicy(CD_IndexPolicy):
    r""" The Drift-Detection generic policy for non-stationary bandits, using a custom Drift-Detection test, for 1-dimensional exponential families.

    - From [["EXP3 with Drift Detection for the Switching Bandit Problem", Robin Allesiardo & Raphael Feraud]](https://www.researchgate.net/profile/Allesiardo_Robin/publication/281028960_EXP3_with_Drift_Detection_for_the_Switching_Bandit_Problem/links/55d1927808aee19936fdac8e.pdf).
    """
    def __init__(self, nbArms,
            H=None, delta=None, C=CONSTANT_C,
            horizon=None, policy=Exp3,
            *args, **kwargs
        ):
        super(DriftDetection_IndexPolicy, self).__init__(nbArms, epsilon=1, policy=policy, *args, **kwargs)
        self.startGame()
        # New parameters
        self.horizon = horizon

        if H is None:
            H = int(np.ceil(C * np.sqrt(horizon * np.log(horizon))))
        assert H >= nbArms, "Error: for the Drift-Detection algorithm, the parameter H should be >= K = {}, but H = {}".format(nbArms, H)  # DEBUG
        self.H = H  #: Parameter :math:`H` for the Drift-Detection algorithm. Default value is :math:`\lceil C \sqrt{T \log(T)} \rceil`, for some constant :math:`C=` ``C`` (= :data:`CONSTANT_C` by default).

        if delta is None:
            delta = np.sqrt(np.log(horizon) / (nbArms * horizon))
        self.delta = delta  #: Parameter :math:`\delta` for the Drift-Detection algorithm. Default value is :math:`\sqrt{\frac{\log(T)}{K T}}` for :math:`K` arms and horizon :math:`T`.

        if 'gamma' not in kwargs:
            gamma = np.sqrt((nbArms * np.log(nbArms) * np.log(horizon)) / horizon)
            try:
                self.policy.gamma = gamma
            except AttributeError:
                print("Warning: the policy {} tried to use default value of gamma = {} but could not set attribute self.policy.gamma to gamma (maybe it's using an Exp3 with a non-constant value of gamma).".format(self, gamma))  # DEBUG

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def proba_random_exploration(self):
        r"""Parameter :math:`\gamma` for the Exp3 algorithm."""
        return self.policy.gamma

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def threshold_h(self):
        r"""Parameter :math:`\varepsilon` for the Drift-Detection algorithm.

        .. math:: \varepsilon = \sqrt{\frac{K \log(\frac{1}{\delta})}{2 \gamma H}}.
        """
        return 2 * sqrt((self.nbArms * log(1.0 / self.delta)) / (2 * self.proba_random_exploration * self.H))

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def min_number_of_pulls_to_test_change(self):
        r"""Compute :math:`\Gamma_{\min}(I) := \frac{\gamma H}{K}`, the minimum number of samples we should have for all arms before testing for a change."""
        Gamma_min = self.proba_random_exploration * self.H / self.nbArms
        return int(np.ceil(Gamma_min))

    def __str__(self):
        return r"DriftDetection-{}($T={}$, $c={:.3g}$, $\alpha={:.3g}$)".format(self._policy.__name__, self.horizon, self.threshold_h, self.proba_random_exploration)

    def detect_change(self, arm, verbose=VERBOSE):
        r""" Detect a change in the current arm, using a Drift-Detection test (DD).

        .. math::

            k_{\max} &:= \arg\max_k \tilde{\rho}_k(t),\\
            DD_t(k) &= \hat{\mu}_k(I) - \hat{\mu}_{k_{\max}}(I).

        - The change is detected if there is an arm :math:`k` such that :math:`DD_t(k) \geq 2 * \varepsilon = h`, where :attr:`threshold_h` is the threshold of the test, and :math:`I` is the (number of the) current interval since the last (global) restart,
        - where :math:`\tilde{\rho}_k(t)` is the trust probability of arm :math:`k` from the Exp3 algorithm,
        - and where :math:`\hat{\mu}_k(I)` is the empirical mean of arm :math:`k` from the data in the current interval.

        .. warning::

            FIXME I know this implementation is not (yet) correct...
            I should count differently the samples we obtained from the Gibbs distribution (when Exp3 uses the trust vector) and from the uniform distribution
            This :math:`\Gamma_{\min}(I)` is the minimum number of samples obtained from the uniform exploration (of probability :math:`\gamma`).
            It seems painful to code correctly, I will do it later.
        """
        # XXX Do we have enough samples?
        min_pulls = np.min(self.last_pulls)
        if min_pulls < self.min_number_of_pulls_to_test_change:  # no we don't
            return False, None
        # Yes we do have enough samples
        trusts = self.policy.trusts
        k_max = np.argmax(trusts)
        means = [np.mean(rewards) for rewards in self.all_rewards]
        meanOfTrustedArm = means[k_max]
        for otherArm in range(self.nbArms):
            difference_of_mean = means[otherArm] - meanOfTrustedArm
            if verbose: print("  - For the mean mu(k={}) = {} and mean of trusted arm mu(k_max={}) = {}, their difference is {}, compared to c = {}...".format(otherArm, means[otherArm], k_max, meanOfTrustedArm, difference_of_mean, self.threshold_h))
            if difference_of_mean >= self.threshold_h:
                return True, None
        return False, None


# --- Exp3R

class Exp3R(DriftDetection_IndexPolicy):
    r""" The Exp3.R policy for non-stationary bandits.
    """

    def __init__(self, nbArms, policy=Exp3, *args, **kwargs):
        super(Exp3R, self).__init__(nbArms, policy=policy, *args, **kwargs)

    def __str__(self):
        return r"Exp3R($T={}$, $c={:.3g}$, $\alpha={:.3g}$)".format(self.horizon, self.threshold_h, self.proba_random_exploration)


# --- Exp3R++

class Exp3RPlusPlus(DriftDetection_IndexPolicy):
    r""" The Exp3.R++ policy for non-stationary bandits.
    """

    def __init__(self, nbArms, policy=Exp3PlusPlus, *args, **kwargs):
        super(Exp3RPlusPlus, self).__init__(nbArms, policy=policy, *args, **kwargs)

    def __str__(self):
        return r"Exp3R++($T={}$, $c={:.3g}$, $\alpha={:.3g}$)".format(self.horizon, self.threshold_h, self.proba_random_exploration)
