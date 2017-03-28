# -*- coding: utf-8 -*-
""" MAB.MAB class to wrap the arms."""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.5"

import numpy as np
import matplotlib.pyplot as plt

# Local imports
from .plotsettings import signature, wraptext, wraplatex, palette, legend, show_and_save


class MAB(object):
    """ Multi-armed Bandit environment.

    - configuration can be a dict with 'arm_type' and 'params' keys. 'arm_type' is a class from the Arms module, and 'params' is a dict, used as a list/tuple/iterable of named parameters given to 'arm_type'. Example::

        configuration = {
            'arm_type': Bernoulli,
            'params':   [0.1, 0.5, 0.9]
        }

    - But it can also accept a list of already created arms::

        configuration = [
            Bernoulli(0.1),
            Bernoulli(0.5),
            Bernoulli(0.9),
        ]

    - Both will create three Bernoulli arms, of parameters (means) 0.1, 0.5 and 0.9.
    """

    def __init__(self, configuration):
        print("Creating a new MAB problem ...")  # DEBUG
        self.static = True  #: Flag to know if the problem is static or not.
        # Previous thing
        if isinstance(configuration, dict):
            print("  Reading arms of this MAB problem from a dictionnary 'configuration' = {} ...".format(configuration))  # DEBUG
            arm_type = configuration["arm_type"]
            print(" - with 'arm_type' =", arm_type)  # DEBUG
            params = configuration["params"]
            print(" - with 'params' =", params)  # DEBUG
            # Each 'param' could be one value (eg. 'mean' = probability for a Bernoulli) or a tuple (eg. '(mu, sigma)' for a Gaussian) or a dictionnary
            self.arms = []
            for param in params:
                self.arms.append(arm_type(*param) if isinstance(param, (dict, tuple, list)) else arm_type(param))
        else:
            print("  Taking arms of this MAB problem from a list of arms 'configuration' = {} ...".format(configuration))  # DEBUG
            self.arms = []
            for arm in configuration:
                self.arms.append(arm)

        # Compute the means and stats
        if self.static:
            print(" - with 'arms' =", self.arms)  # DEBUG
            self.means = np.array([arm.mean for arm in self.arms])
            print(" - with 'means' =", self.means)  # DEBUG
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
        centralized_lowerbound = sum(oneLR(worstOfBestMean, oneOfWorstMean) for oneOfWorstMean in worstMeans)
        print("  - For {} players, Anandtharam et al. centralized lower-bound gave = {:.3g} ...".format(nbPlayers, centralized_lowerbound))  # DEBUG

        our_lowerbound = nbPlayers * centralized_lowerbound
        print("  - For {} players, our lower bound gave = {:.3g} ...".format(nbPlayers, our_lowerbound))  # DEBUG

        # The initial lower bound in Theorem 6 from [Anandkumar et al., 2010]
        kl = self.arms[0].kl
        anandkumar_lowerbound = sum(sum((worstOfBestMean - oneOfWorstMean) / kl(oneOfWorstMean, oneOfBestMean) for oneOfWorstMean in worstMeans) for oneOfBestMean in bestMeans)
        print("  - For {} players, the initial lower bound in Theorem 6 from [Anandkumar et al., 2010] gave = {:.3g} ...".format(nbPlayers, anandkumar_lowerbound))  # DEBUG

        # Check that our bound is better (ie bigger)
        if anandkumar_lowerbound > our_lowerbound:
            print("Error, our lower bound is worse than the one in Theorem 6 from [Anandkumar et al., 2010], but it should always be better...")
        return our_lowerbound, anandkumar_lowerbound, centralized_lowerbound

    def upperbound_collisions(self, nbPlayers, times):
        """ Compute Anandkumar et al. multi-players upper bound for this MAB problem (complexity), for UCB only. Warning: it is HIGHLY asymptotic! """
        sortedMeans = sorted(self.means)
        assert nbPlayers <= len(sortedMeans), "Error: this lowerbound_multiplayers() for a MAB problem is only valid when there is less users than arms. Here M = {} > K = {} ...".format(nbPlayers, len(sortedMeans))
        bestMeans = sortedMeans[-nbPlayers:][::-1]

        def worstMeans_of_a(a):
            return sortedMeans[:-(a + 1)]

        # First, the bound in Lemma 2 from [Anandkumar et al., 2010] uses this Upsilon(U, U)
        Upsilon = binomialCoefficient(nbPlayers, 2 * nbPlayers - 1)
        print("  - For {} players, Upsilon(M,M) = (2M-1 choose M) = {} ...".format(nbPlayers, Upsilon))

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
        print("  - For {} players, Anandkumar et al. upper bound for the total cumulated number of collisions is {:.3g} here ...".format(nbPlayers, upperbound[-1]))  # DEBUG

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
        plt.title("{} draws of rewards from these arms.\n{} arms: ${}${}".format(horizon, self.nbArms, self.reprarms(), signature))
        show_and_save(showplot=True, savefig=savefig)


# FIXME experimental
class DynamicMAB(MAB):
    """Like a static MAB problem, but the arms are (randomly) regenerated everytime they are accessed.

    - Warning: this is still HIGHLY experimental!
    - It can be weird: M.arms is always different everytime it is accessed, but not nbArm, means, minArm, maxArm...
    """

    def __init__(self, configuration):
        self.static = False

        assert isinstance(configuration, dict) \
            and "arm_type" in configuration and "params" in configuration \
            and "function" in configuration["params"] and "args" in configuration["params"], \
            "Error: this DynamicMAB is not really a dynamic MAB, you should use a simple MAB instead!"

        print("  Special MAB problem, changing at every repetitions, read from a dictionnary 'configuration' = {} ...".format(configuration))  # DEBUG
        self.static = False
        self.arm_type = arm_type = configuration["arm_type"]
        print(" - with 'arm_type' =", arm_type)  # DEBUG
        params = configuration["params"]
        print(" - with 'params' =", params)  # DEBUG
        self.function = params["function"]
        print(" - with 'function' =", self.function)  # DEBUG
        self.args = params["args"]
        print(" - with 'args' =", self.args)  # DEBUG
        print("\n\n ==> Creating the dynamic arms ...")  # DEBUG
        self.newRandomArms()
        print("   - drawing a random set of arms")
        self.nbArms = len(self.arms)
        print("   - with 'nbArms' =", self.nbArms)  # DEBUG
        print("   - with 'arms' =", self.arms)  # DEBUG
        print(" - Example of initial draw of 'means' =", self.means)  # DEBUG
        print("   - with 'maxArm' =", self.maxArm)  # DEBUG
        print("   - with 'minArm' =", self.minArm)  # DEBUG

    def __repr__(self):
        if self._arms is not None:
            return "{}(nbArms: {}, arms: {}, minArm: {:.3g}, maxArm: {:.3g})".format(self.__class__.__name__, self.nbArms, self._arms, self.minArm, self.maxArm)
        else:
            return "{}(nbArms: {}, armType: {})".format(self.__class__.__name__, self.nbArms, self.arm_type)

    def reprarms(self, nbPlayers=None, openTag='', endTag='^*', latex=True):
        """Cannot represent the dynamic arms, so print the DynamicMAB object"""
        return repr(self)

    #
    # --- Dynamic arms and means

    def newRandomArms(self, verbose=True):
        """Generate a new list of arms, from arm_type(params['function](*params['args']))."""
        self._arms = [self.arm_type(mean) for mean in self.function(**self.args)]
        self.nbArms = len(self._arms)
        if verbose:
            print("\n  - Creating a new dynamic set of means for arms: DynamicMAB = {} ...".format(repr(self)))  # DEBUG
        return self._arms

    @property
    def arms(self):
        """Return the list of arms."""
        return self._arms

    @property
    def means(self):
        """Return the list of means."""
        return np.array([arm.mean for arm in self._arms])

    @property
    def minArm(self):
        """Return the smallest mean of the arms, for a dynamic MAB."""
        return np.min(self.means)

    @property
    def maxArm(self):
        """Return the largest mean of the arms, for a dynamic MAB."""
        return np.max(self.means)


# --- Utility functions

def binomialCoefficient(k, n):
    r""" Compute a binomial coefficient :math:`C^n_k` by a direct multiplicative method: :math:`C^n_k = {k \choose n}`.

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
