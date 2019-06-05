# -*- coding: utf-8 -*-
""" Different variants of the Explore-Then-Commit policy.

- Reference: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
- And [Kaufmann & Moy, 2017, ICC](http://icc2017.ieee-icc.org/program/tutorials#TT01), E.Kaufmann's slides at IEEE ICC 2017
- See also: https://github.com/SMPyBandits/SMPyBandits/issues/62 and https://github.com/SMPyBandits/SMPyBandits/issues/102
- Also [On Explore-Then-Commit Strategies, by A.Garivier et al, NIPS, 2016](https://arxiv.org/pdf/1605.08988.pdf)

.. warning:: They sometimes do not work empirically as well as the theory predicted...

.. warning:: TODO I should factor all this code and write all of them in a more "unified" way...
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
from scipy.special import lambertw

# Local import
try:
    from .EpsilonGreedy import EpsilonGreedy
    from .BasePolicy import BasePolicy
    from .with_proba import with_proba
except ImportError:
    from EpsilonGreedy import EpsilonGreedy
    from BasePolicy import BasePolicy
    from with_proba import with_proba


#: Default value for the gap, :math:`\Delta = \min_{i\neq j} \mu_i - \mu_j`, :math:`\Delta = 0.1` as in many basic experiments.
GAP = 0.1


class ETC_KnownGap(EpsilonGreedy):
    r""" Variant of the Explore-Then-Commit policy, with known horizon :math:`T` and gap :math:`\Delta = \min_{i\neq j} \mu_i - \mu_j`.

    - Reference: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, horizon=None, gap=GAP,
                 lower=0., amplitude=1.):
        super(ETC_KnownGap, self).__init__(nbArms, epsilon=0.5, lower=lower, amplitude=amplitude)
        # Arguments
        assert horizon > 0, "Error: the 'horizon' parameter for ETC_KnownGap class has to be > 0, but was {}...".format(horizon)  # DEBUG
        self.horizon = int(horizon)  #: Parameter :math:`T` = known horizon of the experiment.
        assert 0 <= gap <= 1, "Error: the 'gap' parameter for ETC_KnownGap class has to be in [0, 1], but was {}.".format(gap)  # DEBUG
        self.gap = gap  #: Known gap parameter for the stopping rule.
        # Compute the time m
        m = max(0, int(np.floor(((4. / gap**2) * np.log(horizon * gap**2 / 4.)))))
        self.max_t = self.nbArms * m  #: Time until pure exploitation, ``m_`` steps in each arm.

    def __str__(self):
        return r"ETC_KnownGap($T={}$, $\Delta={:.3g}$, $T_0={}$)".format(self.horizon, self.gap, self.max_t)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def epsilon(self):
        r""" 1 while :math:`t \leq T_0`, 0 after, where :math:`T_0` is defined by:

        .. math:: T_0 = \lfloor \frac{4}{\Delta^2} \log(\frac{T \Delta^2}{4}) \rfloor.
        """
        if self.t <= self.max_t:
            # First phase: randomly explore!
            return 1
        else:
            # Second phase: just exploit!
            return 0


#: Default value for parameter :math:`\alpha` for :class:`ETC_RandomStop`
ALPHA = 4

class ETC_RandomStop(EpsilonGreedy):
    r""" Variant of the Explore-Then-Commit policy, with known horizon :math:`T` and random stopping time. Uniform exploration until the stopping time.

    - Reference: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, horizon=None, alpha=ALPHA,
                 lower=0., amplitude=1.):
        super(ETC_RandomStop, self).__init__(nbArms, epsilon=0.5, lower=lower, amplitude=amplitude)
        # Arguments
        assert horizon > 0, "Error: the 'horizon' parameter for ETC_RandomStop class has to be > 0."
        self.horizon = int(horizon)  #: Parameter :math:`T` = known horizon of the experiment.
        self.alpha = alpha  #: Parameter :math:`\alpha` in the formula (4 by default).
        self.stillRandom = True  #: Still randomly exploring?

    def __str__(self):
        return r"ETC_RandomStop($T={}$)".format(self.horizon)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def epsilon(self):
        r""" 1 while :math:`t \leq \tau`, 0 after, where :math:`\tau` is a random stopping time, defined by:

        .. math:: \tau = \inf\{ t \in\mathbb{N},\; \max_{i \neq j} \| \widehat{X_i}(t) - \widehat{X_j}(t) \| > \sqrt{\frac{4 \log(T/t)}{t}} \}.
        """
        if np.min(self.pulls) > 0:
            means = self.rewards / self.pulls
            largestDiffMean = max([abs(mi - mj) for mi in means for mj in means if mi != mj])
            if largestDiffMean > np.sqrt((self.alpha * np.log(self.horizon / self.t)) / self.t):
                self.stillRandom = False
        # Done
        if self.stillRandom:
            # First phase: randomly explore!
            return 1
        else:
            # Second phase: just exploit!
            return 0


# --- Other Explore-then-Commit, smarter ones

class ETC_FixedBudget(EpsilonGreedy):
    r""" The Fixed-Budget variant of the Explore-Then-Commit policy, with known horizon :math:`T` and gap :math:`\Delta = \min_{i\neq j} \mu_i - \mu_j`. Sequential exploration until the stopping time.

    - Reference: [On Explore-Then-Commit Strategies, by A.Garivier et al, NIPS, 2016](https://arxiv.org/pdf/1605.08988.pdf), Algorithm 1.
    """

    def __init__(self, nbArms, horizon=None, gap=GAP,
                 lower=0., amplitude=1.):
        super(ETC_FixedBudget, self).__init__(nbArms, epsilon=0.5, lower=lower, amplitude=amplitude)
        # Arguments
        assert horizon > 0, "Error: the 'horizon' parameter for ETC_KnownGap class has to be > 0, but was {}...".format(horizon)  # DEBUG
        self.horizon = int(horizon)  #: Parameter :math:`T` = known horizon of the experiment.
        assert 0 <= gap <= 1, "Error: the 'gap' parameter for ETC_KnownGap class has to be in [0, 1], but was {}.".format(gap)  # DEBUG
        self.gap = gap  #: Known gap parameter for the stopping rule.
        # Compute the time n
        n = np.ceil(2 * abs(lambertw(horizon**2 * gap**4 / (32 * np.pi))) / gap**2)
        self.max_t = nbArms * n  #: Time until pure exploitation.
        self.round_robin_index = -1   #: Internal index to keep the Round-Robin phase
        self.best_identified_arm = None  #: Arm on which we commit, not defined in the beginning.

    def __str__(self):
        return r"ETC_FixedBudget($T={}$, $\Delta={:.3g}$, $T_0={}$)".format(self.horizon, self.gap, self.max_t)

    def choice(self):
        r""" For n rounds, choose each arm sequentially in a Round-Robin phase, then commit to the arm with highest empirical average.

        .. math:: n = \lfloor \frac{2}{\Delta^2} \mathcal{W}(\frac{T^2 \Delta^4}{32 \pi}) \rfloor.

        - Where :math:`\mathcal{W}` is the Lambert W function, defined implicitly by :math:`W(y) \exp(W(y)) = y` for any :math:`y > 0` (and computed with :func:`scipy.special.lambertw`).
        """
        if self.t <= self.max_t:
            self.round_robin_index = (self.round_robin_index + 1) % self.nbArms
            return self.round_robin_index
        else:
            # Commit to the best arm
            if self.best_identified_arm is None:
                means = self.rewards / self.pulls
                self.best_identified_arm = np.random.choice(np.nonzero(means == np.max(means))[0])
            return self.best_identified_arm

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def epsilon(self):
        r""" 1 while :math:`t \leq n`, 0 after. """
        if self.t <= self.max_t:
            # First phase: sequentially explore!
            return 1
        else:
            # Second phase: just exploit!
            return 0


# --- Explore-then-Commit with Round-Robin and Stopping Criteria

class _ETC_RoundRobin_WithStoppingCriteria(EpsilonGreedy):
    r""" Base class for variants of the Explore-Then-Commit policy, with known horizon :math:`T` and gap :math:`\Delta = \min_{i\neq j} \mu_i - \mu_j`. Sequential exploration until the stopping time.

    - Reference: [On Explore-Then-Commit Strategies, by A.Garivier et al, NIPS, 2016](https://arxiv.org/pdf/1605.08988.pdf), Algorithm 2 and 3.
    """

    def __init__(self, nbArms, horizon, gap=GAP,
                 lower=0., amplitude=1.):
        super(_ETC_RoundRobin_WithStoppingCriteria, self).__init__(nbArms, epsilon=0.5, lower=lower, amplitude=amplitude)
        # Arguments
        assert horizon > 0, "Error: the 'horizon' parameter for ETC_KnownGap class has to be > 0, but was {}...".format(horizon)  # DEBUG
        self.horizon = int(horizon)  #: Parameter :math:`T` = known horizon of the experiment.
        assert 0 <= gap <= 1, "Error: the 'gap' parameter for ETC_KnownGap class has to be in [0, 1], but was {}.".format(gap)  # DEBUG
        self.gap = gap  #: Known gap parameter for the stopping rule.
        self.round_robin_index = -1   #: Internal index to keep the Round-Robin phase
        self.best_identified_arm = None  #: Arm on which we commit, not defined in the beginning.

    def __str__(self):
        return r"{}($T={}$, $\Delta={:.3g}$)".format(self.__class__.__name__, self.horizon, self.gap)

    def choice(self):
        r""" Choose each arm sequentially in a Round-Robin phase, as long as the following criteria is not satisfied, then commit to the arm with highest empirical average.

        .. math:: (t/2) \max_{i \neq j} |\hat{\mu_i} - \hat{\mu_j}| < \log(T \Delta^2).
        """
        # not yet committed to the best arm
        if self.best_identified_arm is None:
            self.round_robin_index = (self.round_robin_index + 1) % self.nbArms
            # check if criteria is now false
            if self.round_robin_index == 0:  # only check at the end of a Round-Robin phase
                means = self.rewards / self.pulls
                if self.stopping_criteria():
                    self.best_identified_arm = np.random.choice(np.nonzero(means == np.max(means))[0])
            return self.round_robin_index
        return self.best_identified_arm

    def stopping_criteria(self):
        """ Test if we should stop the Round-Robin phase."""
        raise NotImplementedError

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def epsilon(self):
        r""" 1 while not fixed, 0 after. """
        if self.best_identified_arm is None:
            # First phase: sequentially explore!
            return 1
        else:
            # Second phase: just exploit!
            return 0


class ETC_SPRT(_ETC_RoundRobin_WithStoppingCriteria):
    r""" The Sequential Probability Ratio Test variant of the Explore-Then-Commit policy, with known horizon :math:`T` and gap :math:`\Delta = \min_{i\neq j} \mu_i - \mu_j`.

    - Very similar to :class:`ETC_RandomStop`, but with a sequential exploration until the stopping time.
    - Reference: [On Explore-Then-Commit Strategies, by A.Garivier et al, NIPS, 2016](https://arxiv.org/pdf/1605.08988.pdf), Algorithm 2.
    """

    def stopping_criteria(self):
        """ Test if we should stop the Round-Robin phase."""
        means = self.rewards / self.pulls
        return (self.t / 2) * (np.max(means) - np.min(means)) >= np.log(self.horizon * self.gap**2)



class ETC_BAI(_ETC_RoundRobin_WithStoppingCriteria):
    r""" The Best Arm Identification variant of the Explore-Then-Commit policy, with known horizon :math:`T`.

    - Very similar to :class:`ETC_RandomStop`, but with a sequential exploration until the stopping time.
    - Reference: [On Explore-Then-Commit Strategies, by A.Garivier et al, NIPS, 2016](https://arxiv.org/pdf/1605.08988.pdf), Algorithm 3.
    """

    def __init__(self, nbArms, horizon=None, alpha=ALPHA,
                 lower=0., amplitude=1.):
        super(ETC_BAI, self).__init__(nbArms, horizon=horizon, lower=lower, amplitude=amplitude)
        self.alpha = alpha  #: Parameter :math:`\alpha` in the formula (4 by default).

    def stopping_criteria(self):
        """ Test if we should stop the Round-Robin phase."""
        if self.t < self.nbArms:
            return False
        means = self.rewards / self.pulls
        return (np.max(means) - np.min(means)) >= np.sqrt(self.alpha * np.log(self.horizon / self.t) / self.t)


class DeltaUCB(BasePolicy):
    r""" The DeltaUCB policy, with known horizon :math:`T` and gap :math:`\Delta = \min_{i\neq j} \mu_i - \mu_j`.

    - Reference: [On Explore-Then-Commit Strategies, by A.Garivier et al, NIPS, 2016](https://arxiv.org/pdf/1605.08988.pdf), Algorithm 4.
    """

    def __init__(self, nbArms, horizon, gap=GAP, alpha=ALPHA,
                 lower=0., amplitude=1.):
        super(DeltaUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # Arguments
        assert horizon > 0, "Error: the 'horizon' parameter for ETC_KnownGap class has to be > 0, but was {}...".format(horizon)  # DEBUG
        self.horizon = int(horizon)  #: Parameter :math:`T` = known horizon of the experiment.
        assert 0 <= gap <= 1, "Error: the 'gap' parameter for ETC_KnownGap class has to be in [0, 1], but was {}.".format(gap)  # DEBUG
        self.gap = gap  #: Known gap parameter for the stopping rule.
        self.alpha = alpha  #: Parameter :math:`\alpha` in the formula (4 by default).
        #: Parameter :math:`\varepsilon_T = \Delta (\log(\mathrm{e} + T \Delta^2))^{-1/8}`.
        self.epsilon_T = gap * (np.log(np.exp(1) + horizon * gap**2))**(-1/8.0)

    def __str__(self):
        return r"DeltaUCB($T={}$, $\Delta={:.3g}$, $alpha={:.3g}$)".format(self.horizon, self.gap, self.alpha)

    def choice(self):
        r""" Chose between the most chosen and the least chosen arm, based on the following criteria:

        .. math::

            A_{t,\min} &= \arg\min_k N_k(t),\\
            A_{t,\max} &= \arg\max_k N_k(t).

        .. math::

            UCB_{\min} &= \hat{\mu}_{A_{t,\min}}(t-1) + \sqrt{\alpha \frac{\log(\frac{T}{N_{A_{t,\min}}})}{N_{A_{t,\min}}}} \\
            UCB_{\max} &= \hat{\mu}_{A_{t,\max}}(t-1) + \Delta - \alpha \varepsilon_T

        .. math::
            A(t) = \begin{cases}\\
                A(t) = A_{t,\min} & \text{if  } UCB_{\min} \geq UCB_{\max},\\
                A(t) = A_{t,\max} & \text{else}.
            \end{cases}
        """
        if self.t < self.nbArms:  # force initial exploration of each arm
            return self.t
        # 1. stats on the least chosen arm
        nb_least_chosen = np.min(self.pulls)
        least_chosen = np.random.choice(np.nonzero(self.pulls == nb_least_chosen)[0])
        mean_min = self.rewards[least_chosen] / self.pulls[least_chosen]
        ucb_min = mean_min + np.sqrt(self.alpha * np.log(self.horizon / nb_least_chosen) / nb_least_chosen)
        # 2. stats on the most chosen arm
        most_chosen = np.random.choice(np.nonzero(self.pulls == np.max(self.pulls))[0])
        mean_max = self.rewards[most_chosen] / self.pulls[most_chosen]
        ucb_max = mean_max + self.gap - self.alpha * self.epsilon_T
        # now check the two ucb
        if ucb_min >= ucb_max:
            return least_chosen
        else:
            return most_chosen
