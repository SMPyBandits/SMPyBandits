# -*- coding: utf-8 -*-
""" MAB and DynamicMAB class to wrap the arms."""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
import matplotlib.pyplot as plt

try:
    from .pykov import Chain
except ImportError:
    print("Warning: 'pykov' module seems to not be available. Have you installed it from https://github.com/riccardoscalco/Pykov ?")

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
        """New MAB."""
        print("Creating a new MAB problem ...")  # DEBUG
        self.isDynamic   = False  #: Flag to know if the problem is static or not.
        self.isMarkovian = False  #: Flag to know if the problem is Markovian or not.
        self.arms = []  #: List of arms

        if isinstance(configuration, dict):
            print("  Reading arms of this MAB problem from a dictionnary 'configuration' = {} ...".format(configuration))  # DEBUG
            arm_type = configuration["arm_type"]
            print(" - with 'arm_type' =", arm_type)  # DEBUG
            params = configuration["params"]
            print(" - with 'params' =", params)  # DEBUG
            # Each 'param' could be one value (eg. 'mean' = probability for a Bernoulli) or a tuple (eg. '(mu, sigma)' for a Gaussian) or a dictionnary
            for param in params:
                self.arms.append(arm_type(*param) if isinstance(param, (dict, tuple, list)) else arm_type(param))
        else:
            print("  Taking arms of this MAB problem from a list of arms 'configuration' = {} ...".format(configuration))  # DEBUG
            for arm in configuration:
                self.arms.append(arm)

        # Compute the means and stats
        print(" - with 'arms' =", self.arms)  # DEBUG
        self.means = np.array([arm.mean for arm in self.arms])  #: Means of arms
        print(" - with 'means' =", self.means)  # DEBUG
        self.nbArms = len(self.arms)  #: Number of arms
        print(" - with 'nbArms' =", self.nbArms)  # DEBUG
        self.maxArm = np.max(self.means)  #: Max mean of arms
        print(" - with 'maxArm' =", self.maxArm)  # DEBUG
        self.minArm = np.min(self.means)  #: Min mean of arms
        print(" - with 'minArm' =", self.minArm)  # DEBUG
        # Print lower bound and HOI factor
        print("\nThis MAB problem has: \n - a [Lai & Robbins] complexity constant C(mu) = {:.3g} ... \n - a Optimal Arm Identification factor H_OI(mu) = {:.2%} ...".format(self.lowerbound(), self.hoifactor()))  # DEBUG
        print(" - with 'arms' represented as:", self.reprarms(1, latex=True))  # DEBUG

    def __repr__(self):
        return "{}(nbArms: {}, arms: {}, minArm: {:.3g}, maxArm: {:.3g})".format(self.__class__.__name__, self.nbArms, self.arms, self.minArm, self.maxArm)

    def reprarms(self, nbPlayers=None, openTag='', endTag='^*', latex=True):
        """ Return a str representation of the list of the arms (like `repr(self.arms)` but better)

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

    # --- Draw samples

    def draw(self, armId, t):
        """ Return a random sample from the armId-th arm, at time t. Usually t is not used."""
        return self.arms[armId].draw(t)

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
        print(" -  For {} players, Anandtharam et al. centralized lower-bound gave = {:.3g} ...".format(nbPlayers, centralized_lowerbound))  # DEBUG

        our_lowerbound = nbPlayers * centralized_lowerbound
        print(" -  For {} players, our lower bound gave = {:.3g} ...".format(nbPlayers, our_lowerbound))  # DEBUG

        # The initial lower bound in Theorem 6 from [Anandkumar et al., 2010]
        kl = self.arms[0].kl
        anandkumar_lowerbound = sum(sum((worstOfBestMean - oneOfWorstMean) / kl(oneOfWorstMean, oneOfBestMean) for oneOfWorstMean in worstMeans) for oneOfBestMean in bestMeans)
        print(" -  For {} players, the initial lower bound in Theorem 6 from [Anandkumar et al., 2010] gave = {:.3g} ...".format(nbPlayers, anandkumar_lowerbound))  # DEBUG

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
        print(" -  For {} players, Upsilon(M,M) = (2M-1 choose M) = {} ...".format(nbPlayers, Upsilon))

        # First, the constant term
        from math import pi
        boundOnExpectedTprime_cstTerm = nbPlayers * sum(
            sum(
                (1 + pi**2 / 3.)
                for (b, mu_star_b) in enumerate(worstMeans_of_a(a))
            )
            for (a, mu_star_a) in enumerate(bestMeans)
        )
        print(" -  For {} players, the bound with (1 + pi^2 / 3) = {:.3g} ...".format(nbPlayers, boundOnExpectedTprime_cstTerm))

        # And the term to multiply with log(t)
        boundOnExpectedTprime_logT = nbPlayers * sum(
            sum(
                8. / (mu_star_b - mu_star_a)**2
                for (b, mu_star_b) in enumerate(worstMeans_of_a(a))
            )
            for (a, mu_star_a) in enumerate(bestMeans)
        )
        print(" -  For {} players, the bound with (8 / (mu_b^* - mu_a^*)^2) = {:.3g} ...".format(nbPlayers, boundOnExpectedTprime_logT))

        # Add them up
        boundOnExpectedTprime = boundOnExpectedTprime_cstTerm + boundOnExpectedTprime_logT * np.log(2 + times)

        # The upper bound in Theorem 3 from [Anandkumar et al., 2010]
        upperbound = nbPlayers * (Upsilon + 1) * boundOnExpectedTprime
        print(" -  For {} players, Anandkumar et al. upper bound for the total cumulated number of collisions is {:.3g} here ...".format(nbPlayers, upperbound[-1]))  # DEBUG

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


RESTED = True  #: Default is rested Markovian.


def dict_of_transition_matrix(mat):
    """Convert a transition matrix (list of list or numpy array) to a dictionary mapping (state, state) to probabilities (as used by :class:`pykov.Chain`)."""
    if isinstance(mat, list):
        return {(i, j): mat[i][j] for i in range(len(mat)) for j in range(len(mat[i]))}
    else:
        return {(i, j): mat[i, j] for i in range(len(mat)) for j in range(len(mat[i]))}


def transition_matrix_of_dict(dic):
    """Convert a dictionary mapping (state, state) to probabilities (as used by :class:`pykov.Chain`) to a transition matrix (numpy array)."""
    keys = list(dic.keys())
    xkeys = sorted(list({i for i, _ in keys}))
    ykeys = sorted(list({j for _, j in keys}))
    return np.array([[dic[(i, j)] for i in xkeys] for j in ykeys])


# FIXME experimental, it works, but the regret plots in Evaluator* object has no meaning!
class MarkovianMAB(MAB):
    """ Classic MAB problem but the rewards are drawn from a rested/restless Markov chain.

    - configuration is a dict with 'rested' and 'transitions' keys.
    - 'rested' is a Boolean,
    - 'transitions' is list of K transition matrix, one for each arm.

    Example::

        configuration = {
            "arm_type": "Markovian",
            "params": {
                "rested": True,  # or False
                # Example from [Kalathil et al., 2012](https://arxiv.org/abs/1206.3582) Table 1
                "transitions": [
                    # 1st arm, Either a dictionary
                    {   # Mean = 0.375
                        (0, 0): 0.7, (0, 1): 0.3,
                        (1, 0): 0.5, (1, 1): 0.5,
                    },
                    # 2nd arm, Or a right transition matrix
                    [[0.2, 0.8], [0.6, 0.4]],  # Mean = 0.571
                ],
                # FIXME make this by default! include it in MAB.py and not in the configuration!
                "steadyArm": Bernoulli
            }
        }
    """

    def __init__(self, configuration):
        """New MarkovianMAB."""
        print("Creating a new MarkovianMAB problem ...")  # DEBUG
        self.isDynamic   = False  #: Flag to know if the problem is static or not.
        self.isMarkovian = True  #: Flag to know if the problem is Markovian or not.

        assert isinstance(configuration, dict), "Error: 'configuration' for a MarkovianMAB must be a dictionary."
        assert "params" in configuration and \
               isinstance(configuration["params"], dict) and \
               "transitions" in configuration["params"], \
            "Error: 'configuration.params' for a MarkovianMAB must be a dictionary with keys 'transition' and 'rested'."
        # Use input configuration
        transitions = configuration["params"]["transitions"]
        dict_transitions = []
        matrix_transitions = []
        for t in transitions:
            if isinstance(t, dict):
                dict_transitions.append(t)
                matrix_transitions.append(transition_matrix_of_dict(t))
            else:
                dict_transitions.append(dict_of_transition_matrix(t))
                matrix_transitions.append(np.asarray(t))

        self.matrix_transitions = matrix_transitions
        print(" - Using these transition matrices:", matrix_transitions)  # DEBUG
        self.dict_transitions = dict_transitions
        print(" - Using these transition dictionaries:", dict_transitions)  # DEBUG

        self.chains = [Chain(d) for d in dict_transitions]
        print(" - For these Markov chains:", self.chains)  # DEBUG

        self.rested = configuration["params"].get("rested", RESTED)  #: Rested or not Markovian model?
        print(" - Rested:", self.rested)  # DEBUG

        self.nbArms = len(self.matrix_transitions)  #: Number of arms
        print(" - with 'nbArms' =", self.nbArms)  # DEBUG

        # # Make every transition matrix a right stochastic transition matrix
        # for c in self.chains:
        #     c.stochastic()
        # Means of arms = steady distribution
        states = [np.array(list(c.states())) for c in self.chains]
        print(" - and states:", states)  # DEBUG
        try:
            steadys = [np.array(list(c.steady().values())) for c in self.chains]
        except ValueError:
            if len(c.steady()) == 0:
                print("[ERROR] the steady state of the Markov chain {} was not-found because it is non-ergodic...".format(c))
                raise ValueError("The Markov chain {} is non-ergodic, and so does not have a steady state distribution... Please choose another transition matrix that as to be irreducible, aperiodic, and reversible.".format(c))
        print(" - and steady state distributions:", steadys)  # DEBUG
        self.means = np.array([np.dot(s, p) for s, p in zip(states, steadys)])  #: Means of each arms, from their steady distributions.
        print(" - so it gives arms of means:", self.means)  # DEBUG

        self.arms = [configuration["params"]["steadyArm"](mean) for mean in self.means]
        print(" - so arms asymptotically equivalent to:", self.arms)  # DEBUG
        print(" - represented as:", self.reprarms(1, latex=True))  # DEBUG

        self.maxArm = np.max(self.means)  #: Max mean of arms
        print(" - with 'maxArm' =", self.maxArm)  # DEBUG
        self.minArm = np.min(self.means)  #: Min mean of arms
        print(" - with 'minArm' =", self.minArm)  # DEBUG

        #: States of each arm, initially they are all busy
        self.states = np.zeros(self.nbArms)
        print("DONE for creating this MarkovianMAB problem...")  # DEBUG

    def __repr__(self):
        return "{}(nbArms: {}, chains: {}, arms: {})".format(self.__class__.__name__, self.nbArms, self.matrix_transitions, self.arms)

    def reprarms(self, nbPlayers=None, openTag='', endTag='^*', latex=True):
        """ Return a str representation of the list of the arms (like `repr(self.arms)` but better).

        - For Markovian MAB, the chain and the steady Bernoulli arm is represented.
        - If nbPlayers > 0, it surrounds the representation of the best arms by openTag, endTag (for plot titles, in a multi-player setting).

        - Example: openTag = '', endTag = '^*' for LaTeX tags to put a star exponent.
        - Example: openTag = '<red>', endTag = '</red>' for HTML-like tags.
        - Example: openTag = r'\textcolor{red}{', endTag = '}' for LaTeX tags.
        """
        if nbPlayers is None:
            text = repr(self.matrix_transitions)
        else:
            assert nbPlayers > 0, "Error, the 'nbPlayers' argument for reprarms method of a MAB object has to be a positive integer."
            means = self.means
            bestArms = np.argsort(means)[-min(nbPlayers, self.nbArms):]
            dollar = '$' if latex else ''
            text = '{} Markovian rewards, {}[{}]{}'.format(
                "Rested" if self.rested else "Restless",
                dollar, ', '.join(
                    "{}{} : {}{}".format(openTag, np.asarray(mat).tolist(), repr(arm), endTag) if armId in bestArms
                    else "{} : {}".format(np.asarray(mat).tolist(), repr(arm))
                    for armId, (arm, mat) in enumerate(zip(self.arms, self.matrix_transitions))
                ), dollar
            )
        return wraptext(text)

    def draw(self, armId, t):
        """Move on the Markov chain and return its state as a reward (0 or 1, or else)."""
        # 1. Get current state for that arm, and its Markov chain
        state = self.states[armId]
        chain = self.chains[armId]
        # 2. Sample from that Markov chain
        nextState = chain.move(state)
        # 3. Update the state
        self.states[armId] = nextState
        # print("- For the arm #{}, previously in the state {}, the Markov chain moved to state {} ...".format(armId, state, nextState))  # DEBUG

        if not self.rested:
            # print("- Non-rested Markovian model, every other arm is also moving...")  # DEBUG
            for armId2 in range(self.nbArms):
                # For each other arm, they evolve
                if armId2 != armId:
                    state, chain = self.states[armId2], self.chains[armId2]
                    nextState = chain.move(state)
                    # print("    - For the arm #{}, previously in the state {}, the Markov chain moved to state {} ...".format(armId, state, nextState))  # DEBUG
                    self.states[armId2] = nextState

        return float(nextState)



# FIXME experimental, it works, but the regret plots in Evaluator* object has no meaning!
class DynamicMAB(MAB):
    """Like a static MAB problem, but the arms are (randomly) regenerated everytime they are accessed.

    - Warning: this is still HIGHLY experimental!
    - It can be weird: M.arms is always different everytime it is accessed, but not nbArm, means, minArm, maxArm...
    """

    def __init__(self, configuration):
        """New dynamic MAB."""
        self.isDynamic = True  #: Flag to know if the problem is static or not.

        assert isinstance(configuration, dict) \
            and "arm_type" in configuration and "params" in configuration \
            and "function" in configuration["params"] and "args" in configuration["params"], \
            "Error: this DynamicMAB is not really a dynamic MAB, you should use a simple MAB instead!"

        print("  Special MAB problem, changing at every repetitions, read from a dictionnary 'configuration' = {} ...".format(configuration))  # DEBUG

        self.arm_type = arm_type = configuration["arm_type"]  #: Kind of arm (DynamicMAB are homogeneous)
        print(" - with 'arm_type' =", arm_type)  # DEBUG
        params = configuration["params"]
        print(" - with 'params' =", params)  # DEBUG
        self.function = params["function"]  #: Function to generate the means
        print(" - with 'function' =", self.function)  # DEBUG
        self.args = params["args"]  #: Args to give to function
        print(" - with 'args' =", self.args)  # DEBUG
        print("\n\n ==> Creating the dynamic arms ...")  # DEBUG
        self.newRandomArms()
        print("   - drawing a random set of arms")
        self.nbArms = len(self.arms)  #: Means of arms
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
        return r"\mathrm{%s}(K=%i$, %s on $[%.3g, %.3g], \delta_{\min}=%.3g)" % (self.__class__.__name__, self.nbArms, str(self._arms[0]), self.args["lower"], self.args["lower"] + self.args["amplitude"], self.args["mingap"])

    #
    # --- Dynamic arms and means

    def newRandomArms(self, verbose=True):
        """Generate a new list of arms, from ``arm_type(params['function](*params['args']))``."""
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
