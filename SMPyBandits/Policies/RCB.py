# -*- coding: utf-8 -*-
""" The RCB, Randomized Confidence Bound, policy for bounded bandits.

- Reference: [["On the Optimality of Perturbations in Stochastic and Adversarial Multi-armed Bandit Problems", by Baekjin Kim, Ambuj Tewari, arXiv:1902.00610]](https://arxiv.org/pdf/1902.00610.pdf)
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

try:
    from .RandomizedIndexPolicy import RandomizedIndexPolicy
    from .UCBalpha import UCBalpha
except ImportError:
    from RandomizedIndexPolicy import RandomizedIndexPolicy
    from UCBalpha import UCBalpha


class RCB(RandomizedIndexPolicy, UCBalpha):
    """ The RCB, Randomized Confidence Bound, policy for bounded bandits.

    - Reference: [["On the Optimality of Perturbations in Stochastic and Adversarial Multi-armed Bandit Problems", by Baekjin Kim, Ambuj Tewari, arXiv:1902.00610]](https://arxiv.org/pdf/1902.00610.pdf)
    """
    # FIXME I should implement these RandomizedIndexPolicy variants in a more generic way!
    pass
