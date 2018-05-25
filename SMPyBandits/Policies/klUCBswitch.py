# -*- coding: utf-8 -*-
r""" The kl-UCB-switch policy, for bounded distributions.

- Reference: [Garivier et al, 2018](https://arxiv.org/abs/1805.05071)
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from math import log, sqrt
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .kullback import klucbBern
    from .klUCB import klUCB, c
except ImportError:
    from kullback import klucbBern
    from klUCB import klUCB, c


#: Default value for the tolerance for computing numerical approximations of the kl-UCB indexes.
TOLERANCE = 1e-4


# --- different threshold functions

def threshold_switch_bestchoice(T, K, gamma=1.0/5):
    r""" The threshold function :math:`f(T, K)`, to know when to switch from using :math:`I^{KL}_k(t)` (kl-UCB index) to using :math:`I^{MOSS}_k(t)` (MOSS index).

    .. math:: f(T, K) := \lfloor (T / K)^{\gamma} \rfloor, \gamma = 1/5.
    """
    return np.floor((T / float(K)) ** gamma)


def threshold_switch_delayed(T, K, gamma=8.0/9):
    r""" Another threshold function :math:`f(T, K)`, to know when to switch from using :math:`I^{KL}_k(t)` (kl-UCB index) to using :math:`I^{MOSS}_k(t)` (MOSS index).

    .. math:: f(T, K) := \lfloor (T / K)^{\gamma} \rfloor, \gamma = 8/9.
    """
    return np.floor((T / float(K)) ** gamma)



threshold_switch_default = threshold_switch_bestchoice


# --- Numerical functions required for the indexes for kl-UCB-switch


def klucbplus_index(reward, pull, horizon, nbArms, klucb=klucbBern, c=c, tolerance=TOLERANCE):
    r""" One kl-UCB+ index, from [Cappé et al. 13](https://arxiv.org/pdf/1210.1136.pdf):

    .. math::

        \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
        I^{KL+}_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c \log(T / (K * N_k(t)))}{N_k(t)} \right\}.
    """
    return klucb(reward / pull, c * log(horizon / (nbArms * pull)) / pull, tolerance)


# def klucbplus_indexes(rewards, pulls, horizon, nbArms, klucb=klucbBern, c=c, tolerance=TOLERANCE):
#     r""" The kl-UCB+ indexes, from [Cappé et al. 13](https://arxiv.org/pdf/1210.1136.pdf):

#     .. math::

#         \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
#         I^{KL+}_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c \log(T / (K * N_k(t)))}{N_k(t)} \right\}.
#     """
#     return klucb(rewards / pulls, c * np.log(horizon / (nbArms * pulls)) / pulls, tolerance)


def mossplus_index(reward, pull, horizon, nbArms):
    r""" One MOSS+ index, from [Audibert & Bubeck, 2010](http://www.jmlr.org/papers/volume11/audibert10a/audibert10a.pdf):

    .. math::

        I^{MOSS+}_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\max\left(0, \frac{\log\left(\frac{T}{K N_k(t)}\right)}{N_k(t)}\right)}.
    """
    return (reward / pull) + sqrt(max(0, log(horizon / (nbArms * pull))) / (2 * pull))


# def mossplus_indexes(rewards, pulls, horizon, nbArms):
#     r""" The MOSS+ indexes, from [Audibert & Bubeck, 2010](http://www.jmlr.org/papers/volume11/audibert10a/audibert10a.pdf):

#     .. math::

#         I^{MOSS+}_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\max\left(0, \frac{\log\left(\frac{T}{K N_k(t)}\right)}{N_k(t)}\right)}.
#     """
#     return (rewards / pulls) + np.sqrt(np.maximum(0, np.log(horizon / (nbArms * pulls))) / (2 * pulls))


# --- Classes


class klUCBswitch(klUCB):
    """ The kl-UCB-switch policy, for bounded distributions.

    - Reference: [Garivier et al, 2018](https://arxiv.org/abs/1805.05071)
    """

    def __init__(self, nbArms, horizon=None,
            threshold="best",
            tolerance=TOLERANCE, klucb=klucbBern, c=c,
            lower=0., amplitude=1.
        ):
        super(klUCBswitch, self).__init__(nbArms, tolerance=tolerance, klucb=klucb, c=c, lower=lower, amplitude=amplitude)
        assert horizon is not None, "Error: the klUCBswitch policy require knowledge of the horizon T. Use klUCBswitchAnytime if you need an anytime variant."  # DEBUG
        assert horizon >= 1, "Error: the horizon T should be >= 1."  # DEBUG
        self.horizon = horizon  #: Parameter :math:`T` = known horizon of the experiment.

        # A function, like :func:`threshold_switch`, of T and K, to decide when to switch from kl-UCB indexes to MOSS indexes (for each arm).
        self._threshold_switch_name = "?"
        if isinstance(threshold, str):
            self._threshold_switch_name = ""
            if "best" in threshold:
                threshold_switch = threshold_switch_bestchoice
            elif "delayed" in threshold:
                threshold_switch = threshold_switch_delayed
                self._threshold_switch_name = "delayed f"
            else:
                threshold_switch = threshold_switch_default
        else:
            threshold_switch = threshold
            self._threshold_switch_name = threshold.__name__
        #: For klUCBswitch (not the anytime variant), we can precompute the threshold as it is constant, :math:`= f(T, K)`.
        self.constant_threshold_switch = threshold_switch(self.horizon, self.nbArms)

        #: Initialize internal memory: at first, every arm uses the kl-UCB index, then some will switch to MOSS. (Array of K bool).
        self.use_MOSS_index = np.zeros(nbArms, dtype=bool)

    def __str__(self):
        name = "" if self.klucb.__name__[5:] == "Bern" else self.klucb.__name__[5:] + ", "
        complement = "$T={}${}{}{}".format(self.horizon, name, "" if self.c == 1 else r", $c={:.3g}$".format(self.c), "" if self._threshold_switch_name == "" else ", {}".format(self._threshold_switch_name))
        return r"kl-UCB-switch({})".format(complement)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            U_k(t) = \begin{cases}
                U^{KL+}_k(t) & \text{if } N_k(t) \leq f(T, K), \\
                U^{MOSS+}_k(t) & \text{if } N_k(t) > f(T, K).
            \end{cases}.

        - It starts by using :func:`klucbplus_index`, then it calls :func:`threshold_switch` to know when to stop and start using :func:`mossplus_index`.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        elif self.use_MOSS_index[arm]:
            # no need to compute the threshold, we already use the MOSS index
            return mossplus_index(self.rewards[arm], self.pulls[arm], self.horizon, self.nbArms)
        else:
            if self.pulls[arm] > self.constant_threshold_switch:
                self.self.use_MOSS_index[arm] = True
                return mossplus_index(self.rewards[arm], self.pulls[arm], self.horizon, self.nbArms)
            else:  # default is to use kl-UCB index
                return klucbplus_index(self.rewards[arm], self.pulls[arm], self.horizon, self.nbArms, klucb=self.klucb, c=self.c, tolerance=self.tolerance)

    # def computeAllIndex(self):
    #     """ Compute the current indexes for all arms, in a vectorized manner."""
    #     # XXX I don't think I could hack numpy operations to be faster than a loop for this algorithm
    #     indexes = FIXME
    #     indexes[self.pulls < 1] = float('+inf')
    #     self.index[:] = indexes


# --- Numerical functions required for the indexes for anytime variant kl-UCB-switch


def logplus(x):
    r""" The :math:`\log_+` function.

    .. math:: \log_+(x) := \max(0, \log(x)).
    """
    return max(0, log(x))


# def logplus_vect(x):
#     r""" The :math:`\log_+` function.

#     .. math:: \log_+(x) := \max(0, \log(x)).
#     """
#     return np.maximum(0, np.log(x))


def phi(x):
    r""" The :math:`\phi(x)` function defined in equation (6) in their paper.

    .. math:: \phi(x) := \log_+(x (1 + (\log_+(x))^2)).
    """
    return logplus(x * (1 + (logplus(x))**2))


# def phi_vect(x):
#     r""" The :math:`\phi(x)` function defined in equation (6) in their paper.

#     .. math:: \phi(x) := \log_+(x (1 + (\log_+(x))^2)).
#     """
#     return logplus_vect(x * (1 + (logplus_vect(x))**2))


def klucb_index(reward, pull, t, nbArms, klucb=klucbBern, c=c, tolerance=TOLERANCE):
    r""" One kl-UCB index, from [Garivier & Cappé - COLT, 2011](https://arxiv.org/pdf/1102.2490.pdf):

    .. math::

        \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
        I^{KL}_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c \log(t / N_k(t))}{N_k(t)} \right\}.
    """
    return klucb(reward / pull, c * phi(t / (nbArms * pull)) / pull, tolerance)


# def klucb_indexes(rewards, pulls, t, nbArms, klucb=klucbBern, c=c, tolerance=TOLERANCE):
#     r""" The kl-UCB indexes, from [Garivier & Cappé - COLT, 2011](https://arxiv.org/pdf/1102.2490.pdf):

#     .. math::

#         \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
#         I^{KL}_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c \log(t / N_k(t))}{N_k(t)} \right\}.
#     """
#     return klucb(rewards / pulls, c * phi_vect(t / (nbArms * pulls)) / pulls, tolerance)


def moss_index(reward, pull, t, nbArms):
    r""" One MOSS index, from [Audibert & Bubeck, 2010](http://www.jmlr.org/papers/volume11/audibert10a/audibert10a.pdf):

    .. math::

        I^{MOSS}_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\max\left(0, \frac{\log\left(\frac{t}{K N_k(t)}\right)}{N_k(t)}\right)}.
    """
    return (reward / pull) + sqrt(phi(log(t / (nbArms * pull))) / (2 * pull))


# def moss_indexes(rewards, pulls, t, nbArms):
#     r""" The MOSS indexes, from [Audibert & Bubeck, 2010](http://www.jmlr.org/papers/volume11/audibert10a/audibert10a.pdf):

#     .. math::

#         I^{MOSS}_k(t) &= \frac{X_k(t)}{N_k(t)} + \sqrt{\max\left(0, \frac{\log\left(\frac{t}{K N_k(t)}\right)}{N_k(t)}\right)}.
#     """
#     return (rewards / pulls) + np.sqrt(phi_vect(np.log(t / (nbArms * pulls))) / (2 * pulls))



# --- Anytime variant


class klUCBswitchAnytime(klUCBswitch):
    r""" The anytime variant of the kl-UCB-switch policy, for bounded distributions.

    - It does not use a doubling trick, but an augmented exploration function (replaces the :math:`\log_+` by :math:`\phi` in both :func:`klucb_index` and :func:`moss_index` from :func:`klucbplus_index` and :func:`mossplus_index`).
    - Reference: [Garivier et al, 2018](https://arxiv.org/abs/1805.05071)
    """

    def __init__(self, nbArms,
            threshold="delayed",
            tolerance=TOLERANCE, klucb=klucbBern, c=c,
            lower=0., amplitude=1.
        ):
        super(klUCBswitchAnytime, self).__init__(nbArms, horizon=float('+inf'), threshold=threshold, tolerance=tolerance, klucb=klucb, c=c, lower=lower, amplitude=amplitude)

        self._threshold_switch_name = "?"
        if isinstance(threshold, str):
            self._threshold_switch_name = ""
            if "best" in threshold:
                threshold_switch = threshold_switch_bestchoice
            elif "delayed" in threshold:
                threshold_switch = threshold_switch_delayed
                self._threshold_switch_name = "delayed f"
            else:
                threshold_switch = threshold_switch_default
        else:
            threshold_switch = threshold
            self._threshold_switch_name = threshold.__name__
        #: A function, like :func:`threshold_switch`, of T and K, to decide when to switch from kl-UCB indexes to MOSS indexes (for each arm).
        self.threshold_switch = threshold_switch

    def __str__(self):
        name = "" if self.klucb.__name__[5:] == "Bern" else self.klucb.__name__[5:] + ", "
        complement = "{}{}{}".format(name, "" if self.c == 1 else r", $c={:.3g}$".format(self.c), "" if self._threshold_switch_name == "" else ", {}".format(self._threshold_switch_name))
        if complement.startswith(", "): complement = complement.replace(", ", "", 1)
        complement = "({})".format(complement) if complement != "" else ""
        return r"kl-UCB-switch{}".format(complement)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math::

            U_k(t) = \begin{cases}
                U^{KL}_k(t) & \text{if } N_k(t) \leq f(t, K), \\
                U^{MOSS}_k(t) & \text{if } N_k(t) > f(t, K).
            \end{cases}.

        - It starts by using :func:`klucb_index`, then it calls :func:`threshold_switch` to know when to stop and start using :func:`moss_index`.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        elif self.use_MOSS_index[arm]:
            # no need to compute the threshold, we already use the MOSS index
            return moss_index(self.rewards[arm], self.pulls[arm], self.t, self.nbArms)
        else:
            if self.pulls[arm] > self.threshold_switch(self.t, self.nbArms):
                self.self.use_MOSS_index[arm] = True
                return moss_index(self.rewards[arm], self.pulls[arm], self.t, self.nbArms)
            else:  # default is to use kl-UCB index
                return klucb_index(self.rewards[arm], self.pulls[arm], self.t, self.nbArms, klucb=self.klucb, c=self.c, tolerance=self.tolerance)
