# -*- coding: utf-8 -*-
""" MAB.MAB class to wrap the arms."""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.3"

import numpy as np


class MAB(object):
    """ Multi-armed Bandit environment.

    - configuration has to be a dict with 'arm_type' and 'params' keys.
    - 'arm_type' is a class from the Arms module
    - 'params' is a dict, used as a list/tuple/iterable of named parameters given to 'arm_type'.

    Example:

        configuration = {
            'arm_type': Bernoulli,
            'params':   [0.1, 0.5, 0.9]
        }

    It will create three Bernoulli arms, of parameters (means) 0.1, 0.5 and 0.9.
    """

    def __init__(self, configuration):
        print("Creating a new MAB problem ...")  # DEBUG
        if isinstance(configuration, dict):
            print("  Reading arms of this MAB problem from a dictionnary 'configuration' = {} ...".format(configuration))  # DEBUG
            arm_type = configuration["arm_type"]
            print(" - with 'arm_type' =", arm_type)  # DEBUG
            params = configuration["params"]
            print(" - with 'params' =", params)  # DEBUG
            # Each 'param' could be one value (eg. 'mean' = probability for a Bernoulli) or a tuple (eg. '(mu, sigma)' for a Gaussian)
            self.arms = []
            for param in params:
                try:
                    self.arms.append(arm_type(*param))
                except TypeError:
                    self.arms.append(arm_type(param))
        else:
            print("  Taking arms of this MAB problem from a list of arms 'configuration' = {} ...".format(configuration))  # DEBUG
            self.arms = []
            for arm in configuration:
                self.arms.append(arm)
        print(" - with 'arms' =", self.arms)  # DEBUG
        self.nbArms = len(self.arms)
        print(" - with 'nbArms' =", self.nbArms)  # DEBUG
        self.maxArm = np.max(self.means())
        print(" - with 'maxArm' =", self.maxArm)  # DEBUG

    def __repr__(self):
        return '<' + self.__class__.__name__ + repr(self.__dict__) + '>'

    def means(self):
        return np.array([arm.mean() for arm in self.arms])

    def reprarms(self, nbPlayers=None, openTag='', endTag='^*'):
        """ Return a str representation of the list of the arms (repr(self.arms))

        - If nbPlayers > 0, it surrounds the representation of the best arms by openTag, endTag (for plot titles, in a multi-player setting).

        - Example: openTag = '', endTag = '^*' for LaTeX tags to put a star exponent.
        - Example: openTag = '<red>', endTag = '</red>' for HTML-like tags.
        - Example: openTag = r'\textcolor{red}{', endTag = '}' for LaTeX tags.
        """
        if nbPlayers is None:
            return repr(self.arms)
        else:
            assert nbPlayers > 0, "Error, the 'nbPlayers' argument for reprarms method of a MAB object has to be a positive integer."
            means = self.means()
            bestArms = np.argsort(means)[-min(nbPlayers, self.nbArms):]
            return '[{}]'.format(', '.join(
                openTag + repr(arm) + endTag if armId in bestArms else repr(arm)
                for armId, arm in enumerate(self.arms))
            )

    def lowerbound(self):
        """ Compute the [Lai & Robbins] lower bound for this MAB problem (complexity), using functions from kullback.py or kullback.so. """
        means = self.means()
        bestMean = max(means)
        oneLR = self.arms[0].oneLR
        return sum(oneLR(bestMean, mean) for mean in means if mean != bestMean)

    def lowerbound_multiplayers(self, nbPlayers):
        """ Compute our multi-players lower bound for this MAB problem (complexity), using functions from kullback.py or kullback.so. """
        sortedMeans = sorted(self.means())
        assert nbPlayers <= len(sortedMeans), "Error: this lowerbound_multiplayers() for a MAB problem is only valid when there is less users than arms. Here M = {} > K = {} ...".format(nbPlayers, len(sortedMeans))
        bestMeans = sortedMeans[-nbPlayers:]
        worstMeans = sortedMeans[:-nbPlayers]
        worstOfBestMean = bestMeans[0]

        # Our lower bound is this:
        oneLR = self.arms[0].oneLR
        print("    Using oneLR =", oneLR)  # DEBUG
        our_lowerbound = nbPlayers * sum(oneLR(worstOfBestMean, oneOfWorstMean) for oneOfWorstMean in worstMeans)
        print("  - Our lower bound gave = {} ...".format(our_lowerbound))  # DEBUG

        # The initial lower bound in Theorem 6 from [Anandkumar et al., 2010]
        kl = self.arms[0].kl
        print("    Using kl =", kl)  # DEBUG
        anandkumar_lowerbound = sum(sum((worstOfBestMean - oneOfWorstMean) / kl(oneOfWorstMean, oneOfBestMean) for oneOfWorstMean in worstMeans) for oneOfBestMean in bestMeans)
        print("  - The initial lower bound in Theorem 6 from [Anandkumar et al., 2010] gave = {} ...".format(anandkumar_lowerbound))  # DEBUG

        # Check that our bound is better (ie bigger)
        assert anandkumar_lowerbound <= our_lowerbound, "Error, our lower bound is worse than the one in Theorem 6 from [Anandkumar et al., 2010], but it should always be better..."
        return our_lowerbound, anandkumar_lowerbound
