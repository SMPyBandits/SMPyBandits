# -*- coding: utf-8 -*-
"""
author: Julien Seznec

Rotting Adaptive Window Upper Confidence Bounds for rotting bandits.

Reference : [Seznec et al.,  2019b]
A single algorithm for both rested and restless rotting bandits (WIP)
Julien Seznec, Pierre Ménard, Alessandro Lazaric, Michal Valko
"""


from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Julien Seznec"
__version__ = "0.1"

import numpy as np
import time
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .BasePolicy import BasePolicy
    from .IndexPolicy import IndexPolicy
    from .FEWA import FEWA, EFF_FEWA
    from .kullback import klucbBern
except ImportError:
    from BasePolicy import BasePolicy
    from IndexPolicy import IndexPolicy
    from FEWA import FEWA, EFF_FEWA
    from kullback import klucbBern



class EFF_RAWUCB(EFF_FEWA):
    """
    Efficient Rotting Adaptive Window Upper Confidence Bound (RAW-UCB) [Seznec et al.,  2019b, WIP]
    Efficient trick described in [Seznec et al.,  2019a, https://arxiv.org/abs/1811.11043] (m=2)
    and [Seznec et al.,  2019b, WIP] (m<=2)
    We use the confidence level :math:`\delta_t = \frac{1}{t^\alpha}`.
    """

    def choice(self):
        not_selected = np.where(self.pulls == 0)[0]
        if len(not_selected):
            return not_selected[0]
        self.ucb = self._compute_ucb()
        return np.nanmin(self.ucb, axis=1).argmax()

    def _compute_ucb(self):
        return (self.statistics[0, :, :] / self.windows + self.outlogconst * np.sqrt(np.log(self._inlog())))

    def _append_thresholds(self, w):
        # FEWA use two confidence bounds. Hence, the outlogconst is twice smaller for RAWUCB
        return np.sqrt(2 * self.alpha * self.subgaussian ** 2 / w)

    def __str__(self):
        return r"EFF_RAW-UCB($\alpha={:.3g}, \, m={:.3g}$)".format(self.alpha, self.grid)


class EFF_RAWklUCB(EFF_RAWUCB):
    """
    Use KL-confidence bound instead of close formula approximation.
    Experimental work : Much slower (!!) because we compute many UCB at each round per arm)
    """
    def __init__(self, nbArms, subgaussian=1, alpha=1, klucb=klucbBern, tol=1e-4, m=2):
        super(EFF_RAWklUCB, self).__init__(nbArms=nbArms, subgaussian=subgaussian, alpha=alpha, m=m)
        self.c = alpha
        self.klucb_vec = np.vectorize(klucb, excluded=['precision'])
        self.tolerance = tol

    def choice(self):
        not_selected = np.where(self.pulls == 0)[0]
        if len(not_selected):
            return not_selected[0]
        self.ucb = self.klucb_vec(self.statistics[0, :, :] / self.windows, self.c * np.log(self.t + 1) / self.windows,
                                  precision=self.tolerance)
        return np.argmax(np.nanmin(self.ucb, axis=1))

    def __str__(self):
        return r"EFF_RAW-klUCB($c={:.3g}, \, m={:.3g}$)".format(self.alpha, self.grid)


class RAWUCB(EFF_RAWUCB):
    """
    Rotting Adaptive Window Upper Confidence Bound (RAW-UCB) [Seznec et al.,  2019b, WIP]
    We use the confidence level :math:`\delta_t = \frac{1}{t^\alpha}`.
    """
    def __init__(self, nbArms, subgaussian=1, alpha=1):
        super(RAWUCB, self).__init__(nbArms=nbArms, subgaussian=subgaussian, alpha=alpha, m=1 + 1e-15)

    def __str__(self):
        return r"RAW-UCB($\alpha={:.3g}$)".format(self.alpha)


class EFF_RAWUCB_pp(EFF_RAWUCB):
    """
    Efficient Rotting Adaptive Window Upper Confidence Bound ++ (RAW-UCB++) [Seznec et al.,  2020, Thesis]
    We use the confidence level :math:`\delta_t = \frac{Kh}{t(1+log(t/Kh)^\Beta)}`.
    """

    def __init__(self, nbArms, subgaussian=1, beta=2, m =2):
        super(EFF_RAWUCB_pp, self).__init__(nbArms=nbArms, subgaussian=subgaussian, alpha=1, m=m)
        self.beta = beta

    def __str__(self):
        return r"EFF-RAW-UCB++($\beta={:.3g}, \, m={:.3g}$)".format(self.beta, self.grid)

    def _compute_ucb(self):
        return (self.statistics[0, :, :] / self.windows + self.outlogconst * np.sqrt(np.log(self._inlog(self.windows))))

    def _inlog(self, w):
        moss_confidence = self.t/(w * self.nbArms)
        moss_confidence[moss_confidence < 1] = 1
        inlog = moss_confidence * (1 + np.log(moss_confidence)) ** self.beta
        return inlog

class EFF_RAWUCB_pp2(EFF_RAWUCB):
    """
    Efficient Rotting Adaptive Window Upper Confidence Bound ++ (RAW-UCB++) [Seznec et al.,  2020, Thesis]
    We use the confidence level :math:`\delta_t = \left(\frac{Kh}{t}\right)^{\alpha}`.
    :math:`\Beta=2` corresponds to an asymptotic optimal tuning of UCB for stationnary bandits
    (Bandit Algorithms, Lattimore and Szepesvari,  Chapter 7, https://tor-lattimore.com/downloads/book/book.pdf)
    """

    def __str__(self):
        return r"EFF-RAW-UCB++($\alpha={:.3g}, \, m={:.3g}$)".format(self.alpha, self.grid)

    def _compute_ucb(self):
        return (self.statistics[0, :, :] / self.windows + self.outlogconst * np.sqrt(np.log(self._inlog(self.windows))))

    def _inlog(self, w):
        moss_confidence = self.t/(w * self.nbArms)
        moss_confidence[moss_confidence < 1] = 1
        return moss_confidence **self.alpha

class RAWUCB_pp(EFF_RAWUCB_pp):
    """
    Rotting Adaptive Window Upper Confidence Bound (RAW-UCB) [Seznec et al.,  2019b, WIP]
    We use the confidence level :math:`\delta_t = \frac{Kh}{t^\alpha}`.
    """
    def __init__(self, nbArms, subgaussian=1, beta=2):
        super(EFF_RAWUCB_pp, self).__init__(nbArms=nbArms, subgaussian=subgaussian, beta=beta, m=1 + 1e-15)

    def __str__(self):
        return r"RAW-UCB++($\beta={:.3g}$)".format(self.beta)

if __name__ == "__main__":
    # Code for debugging purposes.
    start = time.time()
    HORIZON = 10**4
    sigma = 1
    reward = {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8}
    policy = EFF_RAWUCB_pp(5, subgaussian=sigma, beta=2)
    for t in range(HORIZON):
        choice = policy.choice()
        policy.getReward(choice, reward[choice])
    print(time.time() - start)
    print(policy.windows[:10])
    print(policy.outlogconst[:10])
    print(policy.pulls)