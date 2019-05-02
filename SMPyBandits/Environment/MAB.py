# -*- coding: utf-8 -*-
""" :class:`MAB`, :class:`MarkovianMAB`, :class:`ChangingAtEachRepMAB`, :class:`IncreasingMAB`, :class:`PieceWiseStationaryMAB` and :class:`NonStationaryMAB` classes to wrap the arms of some Multi-Armed Bandit problems.

Such class has to have *at least* these methods:

- ``draw(armId, t)`` to draw *one* sample from that ``armId`` at time ``t``,
- and ``reprarms()`` to pretty print the arms (for titles of a plot),
- and more, see below.

.. warning:: FIXME it is still a work in progress, I need to add continuously varying environments. See https://github.com/SMPyBandits/SMPyBandits/issues/71
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
import matplotlib.pyplot as plt

try:
    from .pykov import Chain
except ImportError as e:
    try:
        from pykov import Chain
    except ImportError:
        print("Warning: 'pykov' module seems to not be available. But it is shipped with SMPyBandits. Weird.")
        print("Dou you want to try to install it from https://github.com/riccardoscalco/Pykov ?")
        print("Warning: the 'MarkovianMAB' class will not work...")

# Local imports
try:
    from .plotsettings import signature, wraptext, wraplatex, palette, makemarkers, legend, show_and_save
except ImportError:
    from plotsettings import signature, wraptext, wraplatex, palette, makemarkers, legend, show_and_save


class MAB(object):
    """ Basic Multi-Armed Bandit problem, for stochastic and i.i.d. arms.

    - configuration can be a dict with 'arm_type' and 'params' keys. 'arm_type' is a class from the Arms module, and 'params' is a dict, used as a list/tuple/iterable of named parameters given to 'arm_type'. Example::

        configuration = {
            'arm_type': Bernoulli,
            'params':   [0.1, 0.5, 0.9]
        }

        configuration = {  # for fixed variance Gaussian
            'arm_type': Gaussian,
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
        print("\n\nCreating a new MAB problem ...")  # DEBUG
        self.isChangingAtEachRepetition   = False  #: Flag to know if the problem is changing at each repetition or not.
        self.isDynamic   = False  #: Flag to know if the problem is static or not.
        self.isMarkovian = False  #: Flag to know if the problem is Markovian or not.
        self.arms = []  #: List of arms
        self._sparsity = None

        if isinstance(configuration, dict):
            print("  Reading arms of this MAB problem from a dictionnary 'configuration' = {} ...".format(configuration))  # DEBUG
            arm_type = configuration["arm_type"]
            print(" - with 'arm_type' =", arm_type)  # DEBUG
            params = configuration["params"]
            print(" - with 'params' =", params)  # DEBUG
            # Each 'param' could be one value (eg. 'mean' = probability for a Bernoulli) or a tuple (eg. '(mu, sigma)' for a Gaussian) or a dictionnary
            for param in params:
                self.arms.append(arm_type(*param) if isinstance(param, (dict, tuple, list)) else arm_type(param))
            # XXX try to read sparsity
            self._sparsity = configuration["sparsity"] if "sparsity" in configuration else None
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
        if self._sparsity is not None:
            print(" - with 'sparsity' =", self._sparsity)  # DEBUG
        self.maxArm = np.max(self.means)  #: Max mean of arms
        print(" - with 'maxArm' =", self.maxArm)  # DEBUG
        self.minArm = np.min(self.means)  #: Min mean of arms
        print(" - with 'minArm' =", self.minArm)  # DEBUG
        # Print lower bound and HOI factor
        print("\nThis MAB problem has: \n - a [Lai & Robbins] complexity constant C(mu) = {:.3g} ... \n - a Optimal Arm Identification factor H_OI(mu) = {:.2%} ...".format(self.lowerbound(), self.hoifactor()))  # DEBUG
        print(" - with 'arms' represented as:", self.reprarms(1, latex=True))  # DEBUG

    def new_order_of_arm(self, arms):
        """ Feed a new order of the arms to the environment.

        - Updates :attr:`means` correctly.
        - Return the new position(s) of the best arm (to count and plot ``BestArmPulls`` correctly).

        .. warning:: This is a very limited support of non-stationary environment: only permutations of the arms are allowed, see :class:`NonStationaryMAB` for more.
        """
        assert sorted([arm.mean for arm in self.arms]) == sorted([arm.mean for arm in arms]), "Error: the new list of arms = {} does not have the same means as the previous ones."  # DEBUG
        assert set(self.arms) == set(arms), "Error: the new list of arms = {} does not have the same means as the previous ones."  # DEBUG
        self.arms = arms
        self.means = np.array([arm.mean for arm in self.arms])
        self.maxArm = np.max(self.means)
        self.minArm = np.min(self.means)
        return np.nonzero(np.isclose(self.means, self.maxArm))[0]

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
            assert nbPlayers >= 0, "Error, the 'nbPlayers' argument for reprarms method of a MAB object has to be a non-negative integer."  # DEBUG
            append_to_repr = ""

            means = self.means
            bestmean = np.max(means)
            bestArms = np.argsort(means)[-min(nbPlayers, self.nbArms):]
            repr_arms = [repr(arm) for arm in self.arms]

            # WARNING improve display for Gaussian arms that all have same variance
            if all("Gaussian" in str(type(arm)) for arm in self.arms) and len({arm.sigma for arm in self.arms}) == 1:
                sigma = self.arms[0].sigma
                repr_arms = [s.replace(', {:.3g}'.format(sigma), '') for s in repr_arms]
                append_to_repr = r", \sigma^2={:.3g}".format(sigma) if latex else ", sigma2={:.3g}".format(sigma)

            if nbPlayers == 0: bestArms = []
            text = '[{}]'.format(', '.join(
                openTag + repr_arms[armId] + endTag
                if (nbPlayers > 0 and (armId in bestArms or np.isclose(arm.mean, bestmean)))
                else repr_arms[armId]
                for armId, arm in enumerate(self.arms))
            )
            text += append_to_repr
        return wraplatex('$' + text + '$') if latex else wraptext(text)

    # --- Draw samples

    def draw(self, armId, t=1):
        """ Return a random sample from the armId-th arm, at time t. Usually t is not used."""
        return self.arms[armId].draw(t)

    def draw_nparray(self, armId, shape=(1,)):
        """ Return a numpy array of random sample from the armId-th arm, of a certain shape."""
        return self.arms[armId].draw_nparray(shape)

    def draw_each(self, t=1):
        """ Return a random sample from each arm, at time t. Usually t is not used."""
        return np.array([self.draw(armId, t) for armId in range(self.nbArms)])

    def draw_each_nparray(self, shape=(1,)):
        """ Return a numpy array of random sample from each arm, of a certain shape."""
        return np.array([self.draw_nparray(armId, shape) for armId in range(self.nbArms)])

    #
    # --- Helper to compute sets Mbest and Mworst

    def Mbest(self, M=1):
        """ Set of M best means."""
        sortedMeans = np.sort(self.means)
        return sortedMeans[-M:]

    def Mworst(self, M=1):
        """ Set of M worst means."""
        sortedMeans = np.sort(self.means)
        return sortedMeans[:-M]

    def sumBestMeans(self, M=1):
        """ Sum of the M best means."""
        return np.sum(self.Mbest(M=M))

    #
    # --- Helper to compute vector of min arms, max arms, all arms

    def get_minArm(self, horizon=None):
        """Return the vector of min mean of the arms.

        - It is a vector of length horizon.
        """
        return np.full(horizon, self.minArm)
        # return self.minArm  # XXX Nope, it's not a constant!

    def get_maxArm(self, horizon=None):
        """Return the vector of max mean of the arms.

        - It is a vector of length horizon.
        """
        return np.full(horizon, self.maxArm)
        # return self.maxArm  # XXX Nope, it's not a constant!

    def get_maxArms(self, M=1, horizon=None):
        """Return the vector of sum of the M-best means of the arms.

        - It is a vector of length horizon.
        """
        return np.full(horizon, self.sumBestMeans(M))

    def get_allMeans(self, horizon=None):
        """Return the vector of means of the arms.

        - It is a numpy array of shape (nbArms, horizon).
        """
        # allMeans = np.tile(self.means, (horizon, 1)).T
        allMeans = np.zeros((self.nbArms, horizon))
        for t in range(horizon):
            allMeans[:, t] = self.means
        return allMeans

    #
    # --- Estimate sparsity

    @property
    def sparsity(self):
        """ Estimate the sparsity of the problem, i.e., the number of arms with positive means."""
        if self._sparsity is not None:
            return self._sparsity
        else:
            return np.count_nonzero(self.means > 0)

    def str_sparsity(self):
        """ Empty string if ``sparsity = nbArms``, or a small string ', $s={}$' if the sparsity is strictly less than the number of arm."""
        s, K = self.sparsity, self.nbArms
        assert 0 <= s <= K, "Error: sparsity s = {} has to be 0 <= s <= K = {}...".format(s, K)
        # WARNING
        # disable this feature when not working on sparse simulations
        # return ""
        # or bring back this feature when working on sparse simulations
        return "" if s == K else ", $s={}$".format(s)

    #
    # --- Compute lower bounds

    def lowerbound(self):
        r""" Compute the constant :math:`C(\mu)`, for the [Lai & Robbins] lower-bound for this MAB problem (complexity), using functions from ``kullback.py`` or ``kullback.so`` (see :mod:`Arms.kullback`). """
        return sum(a.oneLR(self.maxArm, a.mean) for a in self.arms if a.mean != self.maxArm)

    def lowerbound_sparse(self, sparsity=None):
        """ Compute the constant :math:`C(\mu)`, for [Kwon et al, 2017] lower-bound for sparse bandits for this MAB problem (complexity)

        - I recomputed suboptimal solution to the optimization problem, and found the same as in [["Sparse Stochastic Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)].
        """
        if hasattr(self, "sparsity") and sparsity is None:
            sparsity = self._sparsity
        if sparsity is None:
            sparsity = self.nbArms

        try:
            try:
                from Policies.OSSB import solve_optimization_problem__sparse_bandits
            except ImportError:  # WARNING ModuleNotFoundError is only Python 3.6+
                from SMPyBandits.Policies.OSSB import solve_optimization_problem__sparse_bandits
            ci = solve_optimization_problem__sparse_bandits(self.means, sparsity=sparsity, only_strong_or_weak=False)
            # now we use these ci to compute the lower-bound
            gaps = [self.maxArm - a.mean for a in self.arms]
            lowerbound = sum( delta * c for (delta, c) in zip(gaps, ci) )
        except (ImportError, ValueError, AssertionError):  # WARNING this is durty!
            lowerbound = np.nan
        return lowerbound

    def hoifactor(self):
        """ Compute the HOI factor H_OI(mu), the Optimal Arm Identification (OI) factor, for this MAB problem (complexity). Cf. (3.3) in Navikkumar MODI's thesis, "Machine Learning and Statistical Decision Making for Green Radio" (2017)."""
        return sum(a.oneHOI(self.maxArm, a.mean) for a in self.arms if a.mean != self.maxArm) / float(self.nbArms)

    def lowerbound_multiplayers(self, nbPlayers=1):
        """ Compute our multi-players lower bound for this MAB problem (complexity), using functions from :mod:`kullback`. """
        sortedMeans = sorted(self.means)
        assert nbPlayers <= len(sortedMeans), "Error: this lowerbound_multiplayers() for a MAB problem is only valid when there is less users than arms. Here M = {} > K = {} ...".format(nbPlayers, len(sortedMeans))  # DEBUG
        # FIXME it is highly suboptimal to have a lowerbound = 0 if nbPlayers == nbArms ! We have to finish the theoretical analysis!
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
        assert nbPlayers <= len(sortedMeans), "Error: this lowerbound_multiplayers() for a MAB problem is only valid when there is less users than arms. Here M = {} > K = {} ...".format(nbPlayers, len(sortedMeans))  # DEBUG
        bestMeans = sortedMeans[-nbPlayers:][::-1]

        def worstMeans_of_a(a):
            """ Give the worst min if their is a arms."""
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
        lowerbounds = np.zeros((3, nbPlayers))
        for i in range(nbPlayers):
            lowerbounds[:, i] = self.lowerbound_multiplayers(i + 1)
        fig = plt.figure()
        X = np.arange(1, 1 + nbPlayers)
        plt.plot(X, lowerbounds[0, :], 'ro-', label="Besson & Kaufmann lowerbound")
        plt.plot(X, lowerbounds[1, :], 'bd-', label="Anandkumar et al. lowerbound")
        legend()
        plt.xlabel("Number $M$ of players in the multi-players game{}".format(signature))
        plt.ylabel("Lowerbound on the centralized cumulative normalized regret")
        plt.title("Comparison of our lowerbound and the one from [Anandkumar et al., 2010].\n{} arms: {}".format(self.nbArms, self.reprarms(0, latex=True)))
        show_and_save(showplot=True, savefig=savefig, fig=fig, pickleit=False)
        return fig

    def plotHistogram(self, horizon=10000, savefig=None, bins=50, alpha=0.9, density=None):
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
        fig = plt.figure()
        for armId, arm in enumerate(arms):
            plt.hist(rewards[armId, :], bins=bins, density=density, color=colors[armId], label='$%s$' % repr(arm), alpha=alpha)
        legend()
        plt.xlabel("Rewards")
        if density:
            plt.ylabel("Empirical density of the rewards")
        else:
            plt.ylabel("Empirical count of observations of the rewards")
        plt.title("{} draws of rewards from these arms.\n{} arms: {}{}".format(horizon, self.nbArms, self.reprarms(latex=True), signature))
        show_and_save(showplot=True, savefig=savefig, fig=fig, pickleit=False)
        return fig


# --- MarkovianMAB

RESTED = True  #: Default is rested Markovian.


def dict_of_transition_matrix(mat):
    """ Convert a transition matrix (list of list or numpy array) to a dictionary mapping (state, state) to probabilities (as used by :class:`pykov.Chain`)."""
    if isinstance(mat, list):
        return {(i, j): mat[i][j] for i in range(len(mat)) for j in range(len(mat[i]))}
    else:
        return {(i, j): mat[i, j] for i in range(len(mat)) for j in range(len(mat[i]))}


def transition_matrix_of_dict(dic):
    """ Convert a dictionary mapping (state, state) to probabilities (as used by :class:`pykov.Chain`) to a transition matrix (numpy array)."""
    keys = list(dic.keys())
    xkeys = sorted(list({i for i, _ in keys}))
    ykeys = sorted(list({j for _, j in keys}))
    return np.array([[dic[(i, j)] for i in xkeys] for j in ykeys])


class MarkovianMAB(MAB):
    """ Classic MAB problem but the rewards are drawn from a rested/restless Markov chain.

    - configuration is a dict with ``rested`` and ``transitions`` keys.
    - ``rested`` is a Boolean. See [Kalathil et al., 2012](https://arxiv.org/abs/1206.3582) page 2 for a description.
    - ``transitions`` is list of K transition matrices *or* dictionary (to specify non-integer states), one for each arm.

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

    - This class requires the [pykov](https://github.com/riccardoscalco/Pykov) module to represent and use Markov chain.
    """

    def __init__(self, configuration):
        """New MarkovianMAB."""
        print("\n\nCreating a new MarkovianMAB problem ...")  # DEBUG
        self.isChangingAtEachRepetition   = False  #: The problem is not changing at each repetition.
        self.isDynamic   = False  #: The problem is static.
        self.isMarkovian = True  #: The problem is Markovian.
        self._sparsity = None

        assert isinstance(configuration, dict), "Error: 'configuration' for a MarkovianMAB must be a dictionary."  # DEBUG
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

        # FIXED this will fail harshly if Pykov is not installed/present
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
        self.states = [np.array(list(c.states())) for c in self.chains]
        print(" - and states:", self.states)  # DEBUG
        try:
            self.steadys = [np.array(list(c.steady().values())) for c in self.chains]
        except ValueError:
            for c in self.chains:
                if len(c.steady()) == 0:
                    print("[ERROR] the steady state of the Markov chain {} was not-found because it is non-ergodic...".format(c))
                    raise ValueError("The Markov chain {} is non-ergodic, and so does not have a steady state distribution... Please choose another transition matrix that as to be irreducible, aperiodic, and reversible.".format(c))
        # If the steady state exist, go on
        print(" - and steady state distributions:", self.steadys)  # DEBUG
        self.means = np.array([np.dot(s, p) for s, p in zip(self.states, self.steadys)])  #: Means of each arms, from their steady distributions.
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
            assert nbPlayers >= 0, "Error, the 'nbPlayers' argument for reprarms method of a MAB object has to be a non-negative integer."  # DEBUG
            means = self.means
            bestArms = np.argsort(means)[-min(nbPlayers, self.nbArms):]
            if nbPlayers == 0: bestArms = []
            dollar = '$' if latex else ''
            text = r'{} Markovian rewards, {}[{}]{}'.format(
                "Rested" if self.rested else "Restless",
                dollar, ', '.join(
                    r"{}P: {}, \pi: {} ∼ {}{}".format(
                        openTag if armId in bestArms else "",
                        np.asarray(mat).tolist(), st, repr(arm),
                        endTag if armId in bestArms else ""
                    )
                    for armId, (arm, mat, st) in enumerate(zip(self.arms, self.matrix_transitions, self.steadys))
                ), dollar
            )
        return wraplatex(text) if latex else wraptext(text)

    def draw(self, armId, t=1):
        """ Move on the Markov chain and return its state as a reward (0 or 1, or else).

        - If *rested* Markovian, only the state of the Markov chain of arm `armId` changes. It is the simpler model, and the default model.
        - But if *restless* (non rested) Markovian, the states of all the Markov chain of all arms change (not only `armId`).
        """
        # 1. Get current state for that arm, and its Markov chain
        state, chain = self.states[armId], self.chains[armId]
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


# --- ChangingAtEachRepMAB

VERBOSE = True
VERBOSE = False  #: Whether to be verbose when generating new arms for Dynamic MAB

class ChangingAtEachRepMAB(MAB):
    """Like a stationary MAB problem, but the arms are (randomly) regenerated for each repetition, with the :meth:`newRandomArms` method.

    - ``M.arms`` and ``M.means`` is changed after each call to :meth:`newRandomArms`, but not ``nbArm``. All the other methods are carefully written to still make sense (``Mbest``, ``Mworst``, ``minArm``, ``maxArm``).

    .. warning:: It works perfectly fine, but it is still experimental, be careful when using this feature.

    .. note:: Testing bandit algorithms against randomly generated problems at each repetitions is usually referred to as *"Bayesian problems"* in the literature: a prior is set on problems (eg. uniform on :math:`[0,1]^K` or less obvious for instance if a ``mingap`` is set), and the performance is assessed against this prior. It differs from the *frequentist* point of view of having one fixed problem and doing eg. ``n=1000`` repetitions on the same problem.
    """

    def __init__(self, configuration, verbose=VERBOSE):
        """New ChangingAtEachRepMAB."""
        self.isChangingAtEachRepetition   = True  #: The problem is changing at each repetition or not.
        self.isDynamic   = False  #: The problem is static.
        self.isMarkovian = False  #: The problem is not Markovian.
        self._sparsity = None

        assert isinstance(configuration, dict) \
            and "arm_type" in configuration and "params" in configuration \
            and "newMeans" in configuration["params"] and "args" in configuration["params"], \
            "Error: this ChangingAtEachRepMAB is not really a dynamic MAB, you should use a simple MAB instead!"  # DEBUG
        self._verbose = verbose

        print("  Special MAB problem, changing at every repetitions, read from a dictionnary 'configuration' = {} ...".format(configuration))  # DEBUG

        self.arm_type = arm_type = configuration["arm_type"]  #: Kind of arm (ChangingAtEachRepMAB are homogeneous)
        print(" - with 'arm_type' =", arm_type)  # DEBUG
        params = configuration["params"]
        print(" - with 'params' =", params)  # DEBUG
        self.newMeans = params["newMeans"]  #: Function to generate the means
        print(" - with 'newMeans' =", self.newMeans)  # DEBUG
        self.args = params["args"]  #: Args to give to function
        print(" - with 'args' =", self.args)  # DEBUG
        # XXX try to read sparsity
        self._sparsity = configuration["sparsity"] if "sparsity" in configuration else None
        print("\n\n ==> Creating the dynamic arms ...")  # DEBUG
        # Keep track of the successive mean vectors
        self._historyOfMeans = []  # Historic of the means vectors
        self._t = 0  # nb of calls to the function for generating new arms
        # Generate a first mean vector
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
        """Cannot represent the dynamic arms, so print the ChangingAtEachRepMAB object"""
        # print("reprarms of a ChangingAtEachRepMAB object...")  # DEBUG
        # print("  It has self._historyOfMeans =\n{}".format(self._historyOfMeans))  # DEBUG
        # print("  It has self.means =\n{}".format(self.means))  # DEBUG
        text = "{text}, {K} with uniform means on [{dollar}{lower:.3g}, {upper:.3g}{dollar}]{mingap}{sparsity}".format(
            text="Bayesian MAB",
            K=str(self._arms[0]),
            lower=self.args["lower"],
            upper=self.args["lower"] + self.args["amplitude"],
            mingap="" if self.args["mingap"] is None or self.args["mingap"] == 0 else r", min gap=$%.3g$" % self.args["mingap"],
            sparsity="" if self._sparsity is None else ", sparsity = {dollar}{s}{dollar}".format(s=self._sparsity, dollar="$" if latex else ""),
            dollar="$" if latex else "",
        )
        return wraptext(text)

    #
    # --- Dynamic arms and means

    def newRandomArms(self, t=None, verbose=VERBOSE):
        """Generate a new list of arms, from ``arm_type(params['newMeans'](*params['args']))``."""
        one_draw_of_means = self.newMeans(**self.args)
        self._arms = [self.arm_type(mean) for mean in one_draw_of_means]
        self.nbArms = len(self._arms)  # useless
        self._t += 1  # new draw!
        self._historyOfMeans.append(one_draw_of_means)
        if verbose or self._verbose:
            print("\n  - Creating a new dynamic list of means = {} for arms: ChangingAtEachRepMAB = {} ...".format(np.array(one_draw_of_means), repr(self)))  # DEBUG
            # print("Currently self._t = {} and self._historyOfMeans = {} ...".format(self._t, self._historyOfMeans))  # DEBUG
        return one_draw_of_means

    # All these properties arms, means, minArm, maxArm cannot be attributes, as the means of arms change at every experiments

    @property
    def arms(self):
        """Return the *current* list of arms."""
        return self._arms

    @property
    def means(self):
        """ Return the list of means of arms for this ChangingAtEachRepMAB: after :math:`x` calls to :meth:`newRandomArms`, the return mean of arm :math:`k` is the mean of the :math:`x` means of that arm.

        .. warning:: Highly experimental!
        """
        return np.mean(np.array(self._historyOfMeans), axis=0)

    #
    # --- Helper to compute sets Mbest and Mworst

    def Mbest(self, M=1):
        """ Set of M best means (averaged on all the draws of new means)."""
        sortedMeans = np.mean(np.sort(np.array(self._historyOfMeans), axis=1), axis=0)
        return sortedMeans[-M:]

    def Mworst(self, M=1):
        """ Set of M worst means (averaged on all the draws of new means)."""
        sortedMeans = np.mean(np.sort(np.array(self._historyOfMeans), axis=1), axis=0)
        return sortedMeans[:-M]

    @property
    def minArm(self):
        """Return the smallest mean of the arms, for a dynamic MAB (averaged on all the draws of new means)."""
        return np.mean(np.min(np.array(self._historyOfMeans)))

    @property
    def maxArm(self):
        """Return the largest mean of the arms, for a dynamic MAB (averaged on all the draws of new means)."""
        return np.mean(np.max(np.array(self._historyOfMeans)))

    #
    # --- Compute lower bounds

    def lowerbound(self):
        """ Compute the constant C(mu), for [Lai & Robbins] lower-bound for this MAB problem (complexity), using functions from :mod:`kullback` (averaged on all the draws of new means)."""
        oneLR = self.arms[0].oneLR
        return np.mean([
                        sum(
                            oneLR(np.max(means), m)
                            for m in means
                            if m != np.max(means)
                        )
                        for means in self._historyOfMeans
                        ])

    def hoifactor(self):
        """ Compute the HOI factor H_OI(mu), the Optimal Arm Identification (OI) factor, for this MAB problem (complexity). Cf. (3.3) in Navikkumar MODI's thesis, "Machine Learning and Statistical Decision Making for Green Radio" (2017) (averaged on all the draws of new means)."""
        oneHOI = self.arms[0].oneHOI
        return np.mean([
                        sum(
                            oneHOI(np.max(means), m)
                            for m in means
                            if m != np.max(means)
                        ) / float(len(means))
                        for means in self._historyOfMeans
                        ])

    def lowerbound_multiplayers(self, nbPlayers=1):
        """ Compute our multi-players lower bound for this MAB problem (complexity), using functions from :mod:`kullback`. """
        oneLR = self.arms[0].oneLR
        kl = self.arms[0].kl

        avg_our_lowerbound, avg_anandkumar_lowerbound, avg_centralized_lowerbound = 0.0, 0.0, 0.0

        for means in self._historyOfMeans:
            sortedMeans = sorted(self.means)
            assert nbPlayers <= len(sortedMeans), "Error: this lowerbound_multiplayers() for a MAB problem is only valid when there is less users than arms. Here M = {} > K = {} ...".format(nbPlayers, len(sortedMeans))  # DEBUG
            # FIXME it is highly suboptimal to have a lowerbound = 0 if nbPlayers == nbArms ! We have to finish the theoretical analysis!
            bestMeans = sortedMeans[-nbPlayers:]
            worstMeans = sortedMeans[:-nbPlayers]
            worstOfBestMean = bestMeans[0]

            # Our lower bound is this:
            centralized_lowerbound = sum(oneLR(worstOfBestMean, oneOfWorstMean) for oneOfWorstMean in worstMeans)

            our_lowerbound = nbPlayers * centralized_lowerbound

            # The initial lower bound in Theorem 6 from [Anandkumar et al., 2010]
            anandkumar_lowerbound = sum(sum((worstOfBestMean - oneOfWorstMean) / kl(oneOfWorstMean, oneOfBestMean) for oneOfWorstMean in worstMeans) for oneOfBestMean in bestMeans)

            # Store them
            avg_our_lowerbound += our_lowerbound
            avg_anandkumar_lowerbound += anandkumar_lowerbound
            avg_centralized_lowerbound += centralized_lowerbound

        # Done, compute the averages of the lower-bounds
        avg_our_lowerbound /= float(len(self._historyOfMeans))
        avg_anandkumar_lowerbound /= float(len(self._historyOfMeans))
        avg_centralized_lowerbound /= float(len(self._historyOfMeans))

        print(" -  For {} players, Anandtharam et al. centralized lower-bound gave = {:.3g} ...".format(nbPlayers, avg_centralized_lowerbound))  # DEBUG
        print(" -  For {} players, our lower bound gave = {:.3g} ...".format(nbPlayers, avg_our_lowerbound))  # DEBUG
        print(" -  For {} players, the initial lower bound in Theorem 6 from [Anandkumar et al., 2010] gave = {:.3g} ...".format(nbPlayers, avg_anandkumar_lowerbound))  # DEBUG

        # Check that our bound is better (ie bigger)
        if avg_anandkumar_lowerbound > avg_our_lowerbound:
            print("Error, our lower bound is worse than the one in Theorem 6 from [Anandkumar et al., 2010], but it should always be better...")

        return avg_our_lowerbound, avg_anandkumar_lowerbound, avg_centralized_lowerbound


# --- PieceWiseStationaryMAB

class PieceWiseStationaryMAB(MAB):
    r"""Like a stationary MAB problem, but piece-wise stationary.

    - Give it a list of vector of means, and a list of change-point locations.

    - You can use :meth:`plotHistoryOfMeans` to see a nice plot of the history of means.

    .. note:: This is a generic class to implement one "easy" kind of non-stationary bandits, abruptly changing non-stationary bandits, if changepoints are fixed and decided in advanced.

    .. warning:: It works fine, but it is still experimental, be careful when using this feature.

    .. warning:: The number of arms is fixed, see https://github.com/SMPyBandits/SMPyBandits/issues/123 if you are curious about bandit problems with a varying number of arms (or sleeping bandits where some arms can be enabled or disabled at each time).
    """

    def __init__(self, configuration, verbose=VERBOSE):
        """New PieceWiseStationaryMAB."""
        self.isChangingAtEachRepetition   = False  #: The problem is not changing at each repetition.
        self.isDynamic   = True  #: The problem is dynamic.
        self.isMarkovian = False  #: The problem is not Markovian.
        self._sparsity = None

        assert isinstance(configuration, dict) \
            and "arm_type" in configuration and "params" in configuration \
            and "listOfMeans" in configuration["params"] \
            and "changePoints" in configuration["params"], \
            "Error: this PieceWiseStationaryMAB is not really a non-stationary MAB, you should use a simple MAB instead!"  # DEBUG
        self._verbose = verbose

        print("  Special MAB problem, with arm (possibly) changing at every time step, read from a dictionnary 'configuration' = {} ...".format(configuration))  # DEBUG

        self.arm_type = arm_type = configuration["arm_type"]  #: Kind of arm (PieceWiseStationaryMAB are homogeneous)
        print(" - with 'arm_type' =", arm_type)  # DEBUG
        params = configuration["params"]
        print(" - with 'params' =", params)  # DEBUG

        self.listOfMeans = np.array(params["listOfMeans"])  #: The list of means
        self.nbArms = len(self.listOfMeans[0])  #: Number of arms
        assert all(len(arms) == self.nbArms for arms in self.listOfMeans), "Error: the number of arms cannot be different between change-points."  # DEBUG
        print(" - with 'listOfMeans' =", self.listOfMeans)  # DEBUG

        self.changePoints = params["changePoints"]  #: List of the change points
        print(" - with 'changePoints' =", self.changePoints)  # DEBUG
        # XXX Maybe we need to add 0 in the list of changePoints
        if 0 not in self.changePoints and len(self.listOfMeans) == len(self.changePoints) - 1:
            self.changePoints = [0] + self.changePoints
        assert len(self.listOfMeans) == len(self.changePoints), "Error: the list of means {} does not has the same length as the list of change points {}...".format(self.listOfMeans, self.changePoints)  # DEBUG

        # XXX try to read sparsity
        self._sparsity = configuration["sparsity"] if "sparsity" in configuration else None

        print("\n\n ==> Creating the dynamic arms ...")  # DEBUG

        self.listOfArms = [
            [self.arm_type(mean) for mean in means]
            for means in self.listOfMeans
        ]

        self.currentInterval = 0  # current number of the interval we are in

        print("   - with 'nbArms' =", self.nbArms)  # DEBUG
        print("   - with 'arms' =", self.arms)  # DEBUG
        print(" - Initial draw of 'means' =", self.means)  # DEBUG

    def __repr__(self):
        if len(self.listOfArms) > 0:
            return "{}(nbArms: {}, arms: {})".format(self.__class__.__name__, self.nbArms, self.arms)
        else:
            return "{}(nbArms: {}, armType: {})".format(self.__class__.__name__, self.nbArms, self.arm_type)

    def reprarms(self, nbPlayers=None, openTag='', endTag='^*', latex=True):
        """Cannot represent the dynamic arms, so print the PieceWiseStationaryMAB object"""
        text = r"{text}, {arm} with $\Upsilon={M}$ break-points".format(
            text="Non-Stationary MAB",
            arm=str(self.arms[0]),
            M=len([tau for tau in self.changePoints if tau > 0]),
            # we do not count 0 and horizon
        )
        return wraptext(text)

    def newRandomArms(self, t=None, onlyOneArm=None, verbose=VERBOSE):
        """Fake function, there is nothing random here, it is just to tell the piece-wise stationary MAB problem to maybe use the next interval.
        """
        if t > 0 and t in self.changePoints:
            if verbose: print("  - BREAKPOINT For a PieceWiseStationaryMAB object, the function newRandomArms was called, with t = {}, and current interval was {}, so means was = {} and will be = {}...".format(t, self.currentInterval, self.listOfMeans[self.currentInterval], self.listOfMeans[self.currentInterval + 1]))  # DEBUG
            self.currentInterval += 1  # next interval!
        else:
            if verbose: print("  - For a PieceWiseStationaryMAB object, the function newRandomArms was called, with t = {}, and current interval is {}, so means is = {}...".format(t, self.currentInterval, self.listOfMeans[self.currentInterval]))  # DEBUG
        # return the latest generate means
        return self.listOfMeans[self.currentInterval]

    # --- Plot utility

    def plotHistoryOfMeans(self, horizon=None, savefig=None, forceTo01=False, showplot=True, pickleit=False):
        """Plot the history of means, as a plot with x axis being the time, y axis the mean rewards, and K curves one for each arm."""
        if horizon is None:
            horizon = max(self.changePoints)
        allMeans = self.get_allMeans(horizon=horizon)
        colors = palette(self.nbArms)
        markers = makemarkers(self.nbArms)
        # Now plot
        fig = plt.figure()
        for armId in range(self.nbArms):
            meanOfThisArm = allMeans[armId, :]
            plt.plot(meanOfThisArm, color=colors[armId], marker=markers[armId], markevery=(armId / 50., 0.1), label='Arm #{}'.format(armId), lw=4, alpha=0.9)
        legend()
        ymin, ymax = plt.ylim()
        if forceTo01:
            ymin, ymax = min(0, ymin), max(1, ymax)
            plt.ylim(ymin, ymax)
        if len(self.changePoints) > 20:
            print("WARNING: Adding vlines for the change points with more than 20 change points will be ugly on the plots...")  # DEBUG
        if len(self.changePoints) < 30:  # add the vlines only if not too many change points
            for tau in self.changePoints:
                if tau > 0 and tau < horizon:
                    plt.vlines(tau, ymin, ymax, linestyles='dotted', alpha=0.7)
        plt.xlabel(r"Time steps $t = 1...T$, horizon $T = {}${}".format(horizon, signature))
        plt.ylabel(r"Successive means of the $K = {}$ arms".format(self.nbArms))
        plt.title("History of means for {}".format(self.reprarms(latex=True)))
        show_and_save(showplot=showplot, savefig=savefig, fig=fig, pickleit=pickleit)
        return fig

    # All these properties arms, means, minArm, maxArm cannot be attributes, as the means of arms change at every experiments

    @property
    def arms(self):
        """Return the *current* list of arms. at time :math:`t` , the return mean of arm :math:`k` is the mean during the time interval containing :math:`t`."""
        return self.listOfArms[self.currentInterval]

    @property
    def means(self):
        """ Return the list of means of arms for this PieceWiseStationaryMAB: at time :math:`t` , the return mean of arm :math:`k` is the mean during the time interval containing :math:`t`.
        """
        return self.listOfMeans[self.currentInterval]

    #
    # --- Helper to compute values minArm and maxArm

    @property
    def minArm(self):
        """Return the smallest mean of the arms, for the current vector of means."""
        return np.min(self.means)

    @property
    def maxArm(self):
        """Return the largest mean of the arms, for the current vector of means."""
        return np.max(self.means)

    #
    # --- Helper to compute vector of min arms, max arms, all arms

    def get_minArm(self, horizon=None):
        """Return the smallest mean of the arms, for a piece-wise stationary MAB

        - It is a vector of length horizon.
        """
        if horizon is None:
            horizon = np.max(self.changePoints)
        mapOfMinArms = [np.min(means) for means in self.listOfMeans]
        meansOfMinArms = np.zeros(horizon)
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(self.changePoints) - 1 and t >= self.changePoints[nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfMinArms[t] = mapOfMinArms[nbChangePoint]
        return meansOfMinArms

    def get_minArms(self, M=1, horizon=None):
        """Return the vector of sum of the M-worst means of the arms, for a piece-wise stationary MAB.

        - It is a vector of length horizon.
        """
        if horizon is None:
            horizon = np.max(self.changePoints)
        def Mworst(unsorted_list):
            sorted_list = np.sort(unsorted_list)
            return np.sum(sorted_list[:-M])
        mapOfMworstMaxArms = [Mworst(means) for means in self.listOfMeans]
        meansOfMworstMaxArms = np.ones(horizon)
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(self.changePoints) - 1 and t >= self.changePoints[nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfMworstMaxArms[t] = mapOfMworstMaxArms[nbChangePoint]
        return meansOfMworstMaxArms

    def get_maxArm(self, horizon=None):
        """Return the vector of max mean of the arms, for a piece-wise stationary MAB.

        - It is a vector of length horizon.
        """
        if horizon is None:
            horizon = np.max(self.changePoints)
        mapOfMaxArms = [np.max(means) for means in self.listOfMeans]
        meansOfMaxArms = np.ones(horizon)
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(self.changePoints) - 1 and t >= self.changePoints[nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfMaxArms[t] = mapOfMaxArms[nbChangePoint]
        return meansOfMaxArms

    def get_maxArms(self, M=1, horizon=None):
        """Return the vector of sum of the M-best means of the arms, for a piece-wise stationary MAB.

        - It is a vector of length horizon.
        """
        if horizon is None:
            horizon = np.max(self.changePoints)
        def Mbest(unsorted_list):
            sorted_list = np.sort(unsorted_list)
            return np.sum(sorted_list[-M:])
        mapOfMBestMaxArms = [Mbest(means) for means in self.listOfMeans]
        meansOfMBestMaxArms = np.ones(horizon)
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(self.changePoints) - 1 and t >= self.changePoints[nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfMBestMaxArms[t] = mapOfMBestMaxArms[nbChangePoint]
        return meansOfMBestMaxArms

    def get_allMeans(self, horizon=None):
        """Return the vector of mean of the arms, for a piece-wise stationary MAB.

        - It is a numpy array of shape (nbArms, horizon).
        """
        if horizon is None:
            horizon = np.max(self.changePoints)
        meansOfArms = np.ones((self.nbArms, horizon))
        for armId in range(self.nbArms):
            nbChangePoint = 0
            for t in range(horizon):
                if nbChangePoint < len(self.changePoints) - 1 and t >= self.changePoints[nbChangePoint + 1]:
                    nbChangePoint += 1
                meansOfArms[armId][t] = self.listOfMeans[nbChangePoint][armId]
        return meansOfArms

    #
    # --- Compute lower bounds
    # TODO include knowledge of piece-wise stationarity in the lower-bounds

    # def lowerbound(self):
    #     """ Compute the constant C(mu), for [Lai & Robbins] lower-bound for this MAB problem (complexity), using functions from :mod:`kullback` (averaged on all the draws of new means)."""
    #     raise NotImplementedError

    # def hoifactor(self):
    #     """ Compute the HOI factor H_OI(mu), the Optimal Arm Identification (OI) factor, for this MAB problem (complexity). Cf. (3.3) in Navikkumar MODI's thesis, "Machine Learning and Statistical Decision Making for Green Radio" (2017) (averaged on all the draws of new means)."""
    #     raise NotImplementedError

    # def lowerbound_multiplayers(self, nbPlayers=1):
    #     """ Compute our multi-players lower bound for this MAB problem (complexity), using functions from :mod:`kullback`. """
    #     raise NotImplementedError


# --- PieceWiseStationaryMAB

class NonStationaryMAB(PieceWiseStationaryMAB):
    r"""Like a stationary MAB problem, but the arms *can* be modified *at each time step*, with the :meth:`newRandomArms` method.

    - ``M.arms`` and ``M.means`` is changed after each call to :meth:`newRandomArms`, but not ``nbArm``. All the other methods are carefully written to still make sense (``Mbest``, ``Mworst``, ``minArm``, ``maxArm``).

    .. note:: This is a generic class to implement different kinds of non-stationary bandits:

        - Abruptly changing non-stationary bandits, in different variants: changepoints are randomly drawn (once for all ``n`` repetitions or at different location fo each repetition).
        - Slowly varying non-stationary bandits, where the underlying mean of each arm is slowing randomly modified and a bound on the speed of change (e.g., Lipschitz constant of :math:`t \mapsto \mu_i(t)`) is known.

    .. warning:: It works fine, but it is still experimental, be careful when using this feature.

    .. warning:: The number of arms is fixed, see https://github.com/SMPyBandits/SMPyBandits/issues/123 if you are curious about bandit problems with a varying number of arms (or sleeping bandits where some arms can be enabled or disabled at each time).
    """

    def __init__(self, configuration, verbose=VERBOSE):
        """New NonStationaryMAB."""
        self.isChangingAtEachRepetition   = False  #: The problem is not changing at each repetition.
        self.isDynamic   = True  #: The problem is dynamic.
        self.isMarkovian = False  #: The problem is not Markovian.
        self._sparsity = None

        assert isinstance(configuration, dict) \
            and "arm_type" in configuration and "params" in configuration \
            and "newMeans" in configuration["params"] \
            and "changePoints" in configuration["params"] \
            and "args" in configuration["params"], \
            "Error: this NonStationaryMAB is not really a non-stationary MAB, you should use a simple MAB instead!"  # DEBUG
        self._verbose = verbose

        print("  NonStationary MAB problem, with arm (possibly) changing at every time step, read from a dictionnary 'configuration' = {} ...".format(configuration))  # DEBUG

        self.arm_type = arm_type = configuration["arm_type"]  #: Kind of arm (NonStationaryMAB are homogeneous)
        print(" - with 'arm_type' =", arm_type)  # DEBUG
        params = configuration["params"]
        print(" - with 'params' =", params)  # DEBUG
        self.newMeans = params["newMeans"]  #: Function to generate the means
        print(" - with 'newMeans' =", self.newMeans)  # DEBUG
        self.changePoints = params["changePoints"]  #: List of the change points
        print(" - with 'changePoints' =", self.changePoints)  # DEBUG
        self.onlyOneArm = params.get("onlyOneArm", None)  #: None by default, but can be "uniform" to only change *one* arm at each change point.
        print(" - with 'onlyOneArm' =", self.onlyOneArm)  # DEBUG
        self.args = params["args"]  #: Args to give to function
        print(" - with 'args' =", self.args)  # DEBUG
        # XXX try to read sparsity
        self._sparsity = configuration["sparsity"] if "sparsity" in configuration else None
        print("\n\n ==> Creating the dynamic arms ...")  # DEBUG
        # Keep track of the successive mean vectors
        self._historyOfMeans = dict()  # Historic of the means vectors, storing time of {changepoint: newMeans}
        self._historyOfChangePoints = []  # Historic of the change points
        self._t = 0  # nb of calls to the function for generating new arms
        # Generate a first mean vector
        self.newRandomArms(0)
        print("   - drawing a random set of arms")
        self.nbArms = len(self.arms)  #: Means of arms
        print("   - with 'nbArms' =", self.nbArms)  # DEBUG
        print("   - with 'arms' =", self.arms)  # DEBUG
        print(" - Example of initial draw of 'means' =", self.means)  # DEBUG

    def reprarms(self, nbPlayers=None, openTag='', endTag='^*', latex=True):
        """Cannot represent the dynamic arms, so print the NonStationaryMAB object"""
        # print("reprarms of a NonStationaryMAB object...")  # DEBUG
        # print("  It has self._historyOfMeans =\n{}".format(self._historyOfMeans))  # DEBUG
        # print("  It has self.means =\n{}".format(self.means))  # DEBUG
        text = "{text}, {arm} with uniform means on [{dollar}{lower:.3g}, {upper:.3g}{dollar}]{mingap}{sparsity}".format(
            text="Non-Stationary MAB",
            arm=str(self._arms[0]),
            lower=self.args["lower"],
            upper=self.args["lower"] + self.args["amplitude"],
            mingap="" if self.args["mingap"] is None or self.args["mingap"] == 0 else r", min gap=$%.3g$" % self.args["mingap"],
            sparsity="" if self._sparsity is None else ", sparsity = {dollar}{s}{dollar}".format(s=self._sparsity, dollar="$" if latex else ""),
            dollar="$" if latex else "",
        )
        return wraptext(text)

    #
    # --- Dynamic arms and means

    def newRandomArms(self, t=None, onlyOneArm=None, verbose=VERBOSE):
        """Generate a new list of arms, from ``arm_type(params['newMeans'](t, **params['args']))``.

        - If ``onlyOneArm`` is given and is an integer, the change of mean only occurs for this arm and the others stay the same.
        - If ``onlyOneArm="uniform"``, the change of mean only occurs for one arm and the others stay the same, and the changing arm is chosen uniformly at random.

        .. note:: Only the *means* of the arms change (and so, their order), not their family.

        .. warning:: TODO? So far the only change points we consider is when the means of arms change, but the family of distributions stay the same. I could implement a more generic way, for instance to be able to test algorithms that detect change between different families of distribution (e.g., from a Gaussian of variance=1 to a Gaussian of variance=2, with different or not means).
        """
        if ((t > 0 and t not in self.changePoints) or (t in self._historyOfChangePoints)):
            # return the latest generate means
            return self._historyOfMeans[self._historyOfChangePoints[-1]]
        self._historyOfChangePoints.append(t)
        one_draw_of_means = self.newMeans(**self.args)
        self._t += 1  # new draw!
        if onlyOneArm is not None and len(self._historyOfMeans) > 0:
            if onlyOneArm == "uniform":  # - Handling the option to change only one arm
                onlyOneArm = np.random.randint(self.nbArms)
            elif isinstance(onlyOneArm, int):  # - Or a set of arms
                onlyOneArm = np.random.choice(self.nbArms, min(onlyOneArm, self.nbArms), False)
            if np.ndim(onlyOneArm) == 0:
                onlyOneArm = [onlyOneArm]
            elif np.ndim(onlyOneArm) == 1 and np.size(onlyOneArm) == 1:
                onlyOneArm = [onlyOneArm[0]]  # force to extract the list then wrap it back
            # - If only one arm, and not the first random means, change only one
            # print("onlyOneArm =", onlyOneArm)  # DEBUG
            for arm in range(self.nbArms):
                if arm not in onlyOneArm:
                    one_draw_of_means[arm] = self._historyOfMeans[self._historyOfChangePoints[-2]][arm]
        self._historyOfMeans[t] = one_draw_of_means
        self._arms = [self.arm_type(mean) for mean in one_draw_of_means]
        self.nbArms = len(self._arms)  # useless
        if verbose or self._verbose:
            print("\n  - Creating a new dynamic list of means = {} for arms: NonStationaryMAB = {} ...".format(np.array(one_draw_of_means), repr(self)))  # DEBUG
            # print("Currently self._t = {} and self._historyOfMeans = {} ...".format(self._t, self._historyOfMeans))  # DEBUG
        return one_draw_of_means

    def get_minArm(self, horizon=None):
        """Return the smallest mean of the arms, for a non-stationary MAB

        - It is a vector of length horizon.
        """
        if horizon is None:
            horizon = np.max(self._historyOfChangePoints)
        mapOfMinArms = [np.min (self._historyOfMeans[tau]) for tau in sorted(self._historyOfChangePoints)]
        meansOfMinArms = np.zeros(horizon)
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(self._historyOfChangePoints) - 1 and t >= self._historyOfChangePoints[nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfMinArms[t] = mapOfMinArms[nbChangePoint]
        return meansOfMinArms

    def get_maxArm(self, horizon=None):
        """Return the vector of max mean of the arms, for a non-stationary MAB.

        - It is a vector of length horizon.
        """
        if horizon is None:
            horizon = np.max(self._historyOfChangePoints)
        mapOfMaxArms = [np.max(self._historyOfMeans[tau]) for tau in sorted(self._historyOfChangePoints)]
        meansOfMaxArms = np.ones(horizon)
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(self._historyOfChangePoints) - 1 and t >= self._historyOfChangePoints[nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfMaxArms[t] = mapOfMaxArms[nbChangePoint]
        return meansOfMaxArms

    def get_allMeans(self, horizon=None):
        """Return the vector of mean of the arms, for a non-stationary MAB.

        - It is a numpy array of shape (nbArms, horizon).
        """
        if horizon is None:
            horizon = np.max(self._historyOfChangePoints)
        mapOfArms = [self._historyOfMeans[tau] for tau in sorted(self._historyOfChangePoints)]
        meansOfArms = np.ones((self.nbArms, horizon))
        for armId in range(self.nbArms):
            nbChangePoint = 0
            for t in range(horizon):
                if nbChangePoint < len(self._historyOfChangePoints) - 1 and t >= self._historyOfChangePoints[nbChangePoint + 1]:
                    nbChangePoint += 1
                meansOfArms[armId][t] = mapOfArms[nbChangePoint][armId]
        return meansOfArms


# --- IncreasingMAB


def static_change_lower_amplitude(t, l_t, a_t):
    r"""A function called by :class:`IncreasingMAB` *at every time t*, to compute the (possibly) knew values for :math:`l_t` and :math:`a_t`.

    - First argument is a boolean, `True` if a change occurred, `False` otherwise.
    """
    return False, l_t, a_t


#: Default value for the :func:`doubling_change_lower_amplitude` function.
L0, A0, DELTA, T0, DELTA_T, ZOOM = None, None, 0, 100, 500, 1.1
L0, A0, DELTA, T0, DELTA_T, ZOOM = None, None, 0, 100, 500, 1.05
L0, A0, DELTA, T0, DELTA_T, ZOOM = None, None, 1, 2500, 5000, 2
L0, A0, DELTA, T0, DELTA_T, ZOOM = None, None, 0, -1, 1000, 2
L0, A0, DELTA, T0, DELTA_T, ZOOM = -1, 1, 0, -1, 1000, 2
L0, A0, DELTA, T0, DELTA_T, ZOOM = -1, 2, 0, -1, -1, 2


def doubling_change_lower_amplitude(t, l_t, a_t, l0=L0, a0=A0, delta=DELTA, T0=T0, deltaT=DELTA_T, zoom=ZOOM):
    r"""A function called by :class:`IncreasingMAB` *at every time t*, to compute the (possibly) knew values for :math:`l_t` and :math:`a_t`.

    - At time 0, it forces to use :math:`l_0, a_0` if they are given and not ``None``.
    - At step `T0`, it reduces :math:`l_t` by `delta` (typically from `0` to `-1`).
    - Every `deltaT` steps, it multiplies both  :math:`l_t` and :math:`a_t` by `zoom`.
    - First argument is a boolean, `True` if a change occurred, `False` otherwise.
    """
    if t == 0 and (l0 is not None or a0 is not None):
        different_starting = (l_t != l0) or (a_t != a0)
        if l0 is not None:
            l_t = l0
        if a0 is not None:
            a_t = a0
        return different_starting, l_t, a_t
    elif t > 0:
        if t == T0:
            return (delta != 0), l_t - delta, a_t
        elif deltaT > 0 and t % deltaT == 0:
            return (zoom != 1), zoom * l_t, zoom * a_t
    return False, l_t, a_t


default_change_lower_amplitude = doubling_change_lower_amplitude


class IncreasingMAB(MAB):
    """Like a stationary MAB problem, but the range of the rewards is increased from time to time, to test the :class:`Policy.WrapRange` policy.

    - M.arms and M.means is NOT changed after each call to ``newRandomArms()``, but not nbArm.

    .. warning:: It is purely experimental, be careful when using this feature.
    """

    def __init__(self, configuration):
        """New MAB."""
        super(IncreasingMAB, self).__init__(configuration)
        # XXX Expects a function of (t, lower, amplitude) that gives the new (lower, amplitude)
        self.isDynamic   = True  #: Flag to know if the problem is static or not.
        # WARNING the hash function used on configuration dictionary don't like to have non-hashable part in the dictionary keys, I need to fix that
        if isinstance(configuration, dict):
            self._change_lower_amplitude = configuration.get("change_lower_amplitude", default_change_lower_amplitude)
            if self._change_lower_amplitude is True:
                self._change_lower_amplitude = default_change_lower_amplitude
        else:
            self._change_lower_amplitude = default_change_lower_amplitude
        # Compute the first lower and amplitude values
        lowers, amplitudes = [], []
        for a in self.arms:
            l, a = a.lower_amplitude
            lowers.append(l)
            amplitudes.append(a)
        self._first_lowers = np.array(lowers)
        self._first_amplitudes = np.array(amplitudes)
        self._lowers = np.array(lowers)
        self._amplitudes = np.array(amplitudes)

    def draw(self, armId, t=1):
        """ Return a random sample from the armId-th arm, at time t. Usually t is not used."""
        l_t, a_t = self._lowers[armId], self._amplitudes[armId]
        haschanged, l_tp1, a_tp1 = self._change_lower_amplitude(t, l_t, a_t)
        reward = self.arms[armId].draw(t)
        if haschanged:
            print("Warning: for {}, current l_t, a_t values for arm {} have changed, from {}, {} to {}, {}...".format(self, self.arms[armId], l_t, a_t, l_tp1, a_tp1))  # DEBUG
            self._lowers[armId], self._amplitudes[armId] = l_tp1, a_tp1
        l_of_a, a_of_a = self._first_lowers[armId], self._first_amplitudes[armId]
        # scale it to [0, 1]?
        reward = (reward - l_of_a) / a_of_a
        # now unscale it in the new interval
        reward = l_tp1 + reward * a_tp1
        # finally, be done and return it
        assert l_tp1 <= reward <= l_tp1 + a_tp1, "Error: the new rescaled reward {:.3g} is not in [{:.3g}, {:.3g}]... that shouldn't be possible!".format(reward, l_tp1, l_tp1 + a_tp1)  # DEBUG
        return reward


# --- Utility functions

def binomialCoefficient(k, n):
    r""" Compute a binomial coefficient :math:`C^n_k` by a direct multiplicative method: :math:`C^n_k = {k \choose n}`.

    - Exact, using integers, not like https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.binom.html#scipy.special.binom which uses float numbers.
    - Complexity: :math`\mathcal{O}(1)` in memory, :math`\mathcal{O}(n)` in time.
    - From https://en.wikipedia.org/wiki/Binomial_coefficient#Binomial_coefficient_in_programming_languages
    - From: http://userpages.umbc.edu/~rcampbel/Computers/Python/probstat.html#ProbStat-Combin-Combinations

    - Examples:

    >>> binomialCoefficient(-3, 10)
    0
    >>> binomialCoefficient(1, -10)
    0
    >>> binomialCoefficient(1, 10)
    10
    >>> binomialCoefficient(5, 10)
    80
    >>> binomialCoefficient(5, 20)
    12960
    >>> binomialCoefficient(10, 30)
    10886400
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


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
