# -*- coding: utf-8 -*-
""" MAB.MAB class to wrap the arms."""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.5"

import numpy as np
import matplotlib.pyplot as plt

# Local imports
from .plotsettings import DPI, signature, maximizeWindow, wraptext, wraplatex, palette, legend, show_and_save


class MAB(object):
    """ Multi-armed Bandit environment.

    - configuration has to be a dict with 'arm_type' and 'params' keys.
    - 'arm_type' is a class from the Arms module
    - 'params' is a dict, used as a list/tuple/iterable of named parameters given to 'arm_type'.

    Example::

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
            # Each 'param' could be one value (eg. 'mean' = probability for a Bernoulli) or a tuple (eg. '(mu, sigma)' for a Gaussian) or a dictionnary
            # XXX Maybe that's not a good idea...
            # if isinstance(params, list):   # Sort the means
            #     params = sorted(params)
            self.arms = []
            for param in params:
                self.arms.append(arm_type(*param) if isinstance(param, (dict, tuple, list)) else arm_type(param))
        else:
            print("  Taking arms of this MAB problem from a list of arms 'configuration' = {} ...".format(configuration))  # DEBUG
            self.arms = []
            for arm in configuration:
                self.arms.append(arm)
        # Compute the means
        self.means = np.array([arm.mean for arm in self.arms])
        print(" - with 'arms' =", self.arms)  # DEBUG
        self.nbArms = len(self.arms)
        print(" - with 'nbArms' =", self.nbArms)  # DEBUG
        self.maxArm = np.max(self.means)
        print(" - with 'maxArm' =", self.maxArm)  # DEBUG
        self.minArm = np.min(self.means)
        print(" - with 'minArm' =", self.minArm)  # DEBUG
        # Print lower bound and HOI factor
        print("\nThis MAB problem has: \n - a [Lai & Robbins] complexity constant C(mu) = {:.3g} ... \n - a Optimal Arm Identification factor H_OI(mu) = {:.2%} ...".format(self.lowerbound(), self.hoifactor()))  # DEBUG

    def __repr__(self):
        return "{}(nbArms: {}, arms: {}, minArm: {:.3g}, maxArm: {:.3g})".format(self.__class__.__name__, self.nbArms, self.arms, self.minArm, self.maxArm)

    # # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    # @property
    # def means(self):
    #     """Return list of means."""
    #     return np.array([arm.mean for arm in self.arms])

    def reprarms(self, nbPlayers=None, openTag='', endTag='^*', latex=True):
        """ Return a str representation of the list of the arms (repr(self.arms))

        - If nbPlayers > 0, it surrounds the representation of the best arms by openTag, endTag (for plot titles, in a multi-player setting).

        - Example: openTag = '', endTag = '^*' for LaTeX tags to put a star exponent.
        - Example: openTag = '<red>', endTag = '</red>' for HTML-like tags.
        - Example: openTag = r'\textcolor{red}{', endTag = '}' for LaTeX tags.
        """
        if nbPlayers is None:
            text = repr(self.arms)
        else:
            assert nbPlayers > 0, "Error, the 'nbPlayers' argument for reprarms method of a MAB object has to be a positive integer."
            means = self.means
            bestArms = np.argsort(means)[-min(nbPlayers, self.nbArms):]
            text = '[{}]'.format(', '.join(
                openTag + repr(arm) + endTag if armId in bestArms else repr(arm)
                for armId, arm in enumerate(self.arms))
            )
        if latex:
            return wraplatex(text)
        else:
            return wraptext(text)

    #
    # --- Compute lower bounds

    def lowerbound(self):
        """ Compute the constant C(mu), for [Lai & Robbins] lower-bound for this MAB problem (complexity), using functions from kullback.py or kullback.so. """
        return sum(a.oneLR(self.maxArm, a.mean) for a in self.arms if a.mean != self.maxArm)

    def hoifactor(self):
        """ Compute the HOI factor H_OI(mu), the Optimal Arm Identification (OI) factor, for this MAB problem (complexity). Cf. (3.3) in Navikkumar MODI's thesis, "Machine Learning and Statistical Decision Making for Green Radio" (2017)."""
        return sum(a.oneHOI(self.maxArm, a.mean) for a in self.arms if a.mean != self.maxArm) / float(self.nbArms)

    def lowerbound_multiplayers(self, nbPlayers=1):
        """ Compute our multi-players lower bound for this MAB problem (complexity), using functions from kullback.py or kullback.so. """
        sortedMeans = sorted(self.means)
        assert nbPlayers <= len(sortedMeans), "Error: this lowerbound_multiplayers() for a MAB problem is only valid when there is less users than arms. Here M = {} > K = {} ...".format(nbPlayers, len(sortedMeans))
        # FIXME it is highly suboptimal to have a lowerbound = 0 if nbPlayers == nbArms
        bestMeans = sortedMeans[-nbPlayers:]
        worstMeans = sortedMeans[:-nbPlayers]
        worstOfBestMean = bestMeans[0]

        # Our lower bound is this:
        oneLR = self.arms[0].oneLR
        our_lowerbound = nbPlayers * sum(oneLR(worstOfBestMean, oneOfWorstMean) for oneOfWorstMean in worstMeans)
        print("  - For {} players, our lower bound gave = {:.3g} ...".format(nbPlayers, our_lowerbound))  # DEBUG

        # The initial lower bound in Theorem 6 from [Anandkumar et al., 2010]
        kl = self.arms[0].kl
        anandkumar_lowerbound = sum(sum((worstOfBestMean - oneOfWorstMean) / kl(oneOfWorstMean, oneOfBestMean) for oneOfWorstMean in worstMeans) for oneOfBestMean in bestMeans)
        print("  - For {} players, the initial lower bound in Theorem 6 from [Anandkumar et al., 2010] gave = {:.3g} ...".format(nbPlayers, anandkumar_lowerbound))  # DEBUG

        # Check that our bound is better (ie bigger)
        if anandkumar_lowerbound > our_lowerbound:
            print("Error, our lower bound is worse than the one in Theorem 6 from [Anandkumar et al., 2010], but it should always be better...")
        return our_lowerbound, anandkumar_lowerbound

    def upperbound_collisions(self, nbPlayers, times):
        """ Compute Anandkumar et al. multi-players upper bound for this MAB problem (complexity), using functions from kullback.py or kullback.so. """
        sortedMeans = sorted(self.means)
        assert nbPlayers <= len(sortedMeans), "Error: this lowerbound_multiplayers() for a MAB problem is only valid when there is less users than arms. Here M = {} > K = {} ...".format(nbPlayers, len(sortedMeans))
        bestMeans = sortedMeans[-nbPlayers:]

        def worstMeans_of_a(a):
            return sortedMeans[:-a]

        # First, the bound in Lemma 2 from [Anandkumar et al., 2010] uses this Upsilon(U, U)
        Upsilon = binomialCoefficient(nbPlayers, 2 * nbPlayers - 1)
        print("  - For {} players, Upsilon(M,M) = (2M-1 choose M) = {} ...".format(nbPlayers, Upsilon))

        # Then, Lemma 3 from [Anandkumar et al., 2010] bounds the excepted number of steps before all players have learned a correct ranking of the arms, for UCB1 only

        # First, the constant term
        from math import pi
        boundOnExpectedTprime_cstTerm = nbPlayers * sum(
            sum(
                (1 + pi**2 / 3.)
                for (b, mu_star_b) in enumerate(worstMeans_of_a(a))
            )
            for (a, mu_star_a) in enumerate(bestMeans)
        )
        print("  - For {} players, the bound with (1 + pi^2 / 3) = {:.3g} ...".format(nbPlayers, boundOnExpectedTprime_cstTerm))

        # And the term to multiply with log(t)
        boundOnExpectedTprime_logT = nbPlayers * sum(
            sum(
                8. / (mu_star_b - mu_star_a)**2
                for (b, mu_star_b) in enumerate(worstMeans_of_a(a))
            )
            for (a, mu_star_a) in enumerate(bestMeans)
        )
        print("  - For {} players, the bound with (8 / (mu_b^* - mu_a^*)^2) = {:.3g} ...".format(nbPlayers, boundOnExpectedTprime_logT))

        # Add them up
        boundOnExpectedTprime = boundOnExpectedTprime_cstTerm + boundOnExpectedTprime_logT * np.log(2 + times)

        # The upper bound in Theorem 3 from [Anandkumar et al., 2010]
        upperbound = nbPlayers * (Upsilon + 1) * boundOnExpectedTprime
        print("  - For {} players, Anandkumar et al. upper bound for the non-cumulated number of collisions is {:.3g} * log(t) here ...".format(nbPlayers, upperbound))  # DEBUG

        return upperbound

    # --- Plot methods

    def plotComparison_our_anandkumar(self, savefig=None):
        """Plot a comparison of our lowerbound and their lowerbound."""
        nbPlayers = self.nbArms
        lowerbounds = np.zeros((2, nbPlayers))
        for i in range(nbPlayers):
            lowerbounds[:, i] = self.lowerbound_multiplayers(i + 1)
        plt.figure()
        X = np.arange(1, 1 + nbPlayers)
        plt.plot(X, lowerbounds[0, :], 'ro-', label="Kaufmann & Besson lowerbound")
        plt.plot(X, lowerbounds[1, :], 'bd-', label="Anandkumar et al. lowerbound")
        legend()
        plt.xlabel("Number of players in the multi-players game.{}".format(signature))
        plt.ylabel("Lowerbound on the centralized cumulative normalized regret.")
        plt.title("Comparison of our lowerbound and the one from [Anandkumar et al., 2010].\n{} arms: ${}$".format(self.nbArms, self.reprarms()))
        show_and_save(showplot=True, savefig=savefig)

    def plotHistogram(self, horizon=10000, savefig=None):
        """Plot a horizon=10000 draws of each arms."""
        arms = self.arms
        rewards = np.zeros((len(arms), horizon))
        colors = palette(len(arms))
        for armId, arm in enumerate(arms):
            if hasattr(arm, 'draw_nparray'):  # XXX Use this method to speed up computation
                rewards[armId] = arm.draw_nparray((horizon,))
            else:  # Slower
                for t in range(horizon):
                    rewards[armId, t] = arm.draw(t)
        # Now plot
        plt.figure()
        for armId, arm in enumerate(arms):
            plt.hist(rewards[armId, :], bins=200, normed=True, color=colors[armId], label='$%s$' % repr(arm), alpha=0.7)
        legend()
        plt.xlabel("Rewards")
        plt.ylabel("Mass repartition of the rewards")
        plt.title("{} draws of rewards from these arms.\n{} arms: ${}$".format(horizon, self.nbArms, self.reprarms()))
        show_and_save(showplot=True, savefig=savefig)


# --- Utility functions

def binomialCoefficient(k, n):
    r""" Compute n factorial by a direct multiplicative method:  (:math:`C^n_k = {k \choose n}`).

    - Exact, using integers, not like https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.binom.html#scipy.special.binom which uses float numbers.
    - Complexity: O(1) in memory, O(n) in time.
    - From https://en.wikipedia.org/wiki/Binomial_coefficient#Binomial_coefficient_in_programming_languages
    - From: http://userpages.umbc.edu/~rcampbel/Computers/Python/probstat.html#ProbStat-Combin-Combinations
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # take advantage of symmetry
    c = 1
    for i in range(k):
        c *= (n - i) // (i + 1)
    return c
