# -*- coding: utf-8 -*-
""" Evaluator class to wrap and run the simulations."""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.2"

# Generic imports
from copy import deepcopy
import random
# Scientific imports
import numpy as np
import matplotlib.pyplot as plt
try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    print("joblib not found. Install it from pypi ('pip install joblib') or conda.")
    USE_JOBLIB = False
# Local imports
from .plotsettings import DPI, signature, maximizeWindow, palette, makemarkers
from .Result import Result
from .MAB import MAB


# Parameters for the random events
random_shuffle = False
random_invert = False
nb_random_events = 5


class Evaluator(object):
    """ Evaluator class to run the simulations."""

    def __init__(self, configuration,
                 finalRanksOnAverage=True, averageOn=5e-3,
                 useJoblibForPolicies=False):
        # Configuration
        self.cfg = configuration
        # Attributes
        self.nbPolicies = len(self.cfg['policies'])
        print("Number of policies in this comparaison:", self.nbPolicies)
        self.horizon = self.cfg['horizon']
        print("Time horizon:", self.horizon)
        self.repetitions = self.cfg['repetitions']
        print("Number of repetitions:", self.repetitions)
        self.delta_t_save = self.cfg['delta_t_save']
        print("Sampling rate DELTA_T_SAVE:", self.delta_t_save)
        self.duration = int(self.horizon / self.delta_t_save)
        # Parameters for the random events
        self.random_shuffle = self.cfg.get('random_shuffle', random_shuffle)
        self.random_invert = self.cfg.get('random_invert', random_invert)
        self.nb_random_events = self.cfg.get('nb_random_events', nb_random_events)
        # Flags
        self.finalRanksOnAverage = finalRanksOnAverage
        self.averageOn = averageOn
        self.useJoblibForPolicies = useJoblibForPolicies
        self.useJoblib = USE_JOBLIB and self.cfg['n_jobs'] != 1
        self.cache_rewards = self.cfg.get('cache_rewards', False)
        # Internal object memory
        self.envs = []
        self.policies = []
        self.__initEnvironments__()
        # Internal vectorial memory
        self.rewards = np.zeros((self.nbPolicies, len(self.envs), self.duration))
        self.rewardsSquared = np.zeros((self.nbPolicies, len(self.envs), self.duration))
        self.BestArmPulls = dict()
        self.pulls = dict()
        for env in range(len(self.envs)):
            self.BestArmPulls[env] = np.zeros((self.nbPolicies, self.duration))
            self.pulls[env] = np.zeros((self.nbPolicies, self.envs[env].nbArms))
        print("Number of environments to try:", len(self.envs))
        # To speed up plotting
        self.times = np.arange(1, 1 + self.horizon, self.delta_t_save)

    # --- Init methods

    def __initEnvironments__(self):
        for configuration_arms in self.cfg['environment']:
            self.envs.append(MAB(configuration_arms))

    def __initPolicies__(self, env):
        for policyId, policy in enumerate(self.cfg['policies']):
            print("- Adding policy #{} = {} ...".format(policyId + 1, policy))  # DEBUG
            if isinstance(policy, dict):
                print("  Creating this policy from a dictionnary 'self.cfg['policies'][{}]' = {} ...".format(policyId, policy))  # DEBUG
                self.policies.append(policy['archtype'](env.nbArms, **policy['params']))
            else:
                print("  Using this already created policy 'self.cfg['policies'][{}]' = {} ...".format(policyId, policy))  # DEBUG
                self.policies.append(policy)

    # --- Start computation

    def compute_cache_rewards(self, arms):
        """ Compute only once the rewards, then launch the experiments with the same matrix (r_{k,t})."""
        rewards = np.zeros((len(arms), self.horizon))
        print("\n===> Pre-computing the rewards ... Of shape {} ...\n    In order for all simulated algorithms to face the same random rewards (robust comparaison of A1,..,An vs Aggr(A1,..,An)) ...\n".format(np.shape(rewards)))  # DEBUG
        for armId, arm in enumerate(arms):
            if hasattr(arm, 'draw_nparray'):  # XXX Use this method to speed up computation
                rewards[armId] = arm.draw_nparray((self.horizon,))
            else:  # Slower
                for t in range(self.horizon):
                    rewards[armId, t] = arm.draw(t)
        self.allrewards = rewards

    def startAllEnv(self):
        for envId, env in enumerate(self.envs):
            self.startOneEnv(envId, env)

    # @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof
    def startOneEnv(self, envId, env):
        print("\nEvaluating environment:", repr(env))
        self.policies = []
        self.__initPolicies__(env)
        # Precompute rewards
        if self.cache_rewards:
            allrewards = self.compute_cache_rewards(env.arms)
        else:
            allrewards = None

        # Start for all policies
        for policyId, policy in enumerate(self.policies):
            print("\n- Evaluating policy #{}/{}: {} ...".format(policyId + 1, self.nbPolicies, policy))
            if self.useJoblib:
                seeds = np.random.randint(low=0, high=100 * self.repetitions, size=self.repetitions)
                results = joblib.Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
                    joblib.delayed(delayed_play)(env, policy, self.horizon,
                        random_shuffle=self.random_shuffle, random_invert=self.random_invert, nb_random_events=self.nb_random_events,
                        delta_t_save=self.delta_t_save, allrewards=allrewards, seed=seeds[i]
                    )
                    for i in range(self.repetitions)
                )
            else:
                results = []
                for _ in range(self.repetitions):
                    r = delayed_play(env, policy, self.horizon,
                        random_shuffle=self.random_shuffle, random_invert=self.random_invert, nb_random_events=self.nb_random_events,
                        delta_t_save=self.delta_t_save, allrewards=allrewards
                        )
                    results.append(r)
            # Get the position of the best arms
            means = np.array([arm.mean() for arm in env.arms])
            bestarm = np.max(means)
            index_bestarm = np.nonzero(np.isclose(means, bestarm))[0]
            # Store the results
            for r in results:
                self.rewards[policyId, envId, :] += r.rewards
                self.rewardsSquared[policyId, envId, :] += r.rewardsSquared
                self.BestArmPulls[envId][policyId, :] += np.cumsum(np.in1d(r.choices, index_bestarm))
                # FIXME this BestArmPulls is wrong in case of dynamic change of arm configurations
                self.pulls[envId][policyId, :] += r.pulls

    # --- Getter methods

    def getPulls(self, policyId, environmentId=0):
        return self.pulls[environmentId][policyId, :] / float(self.repetitions)

    def getBestArmPulls(self, policyId, environmentId=0):
        # We have to divide by a arange() = cumsum(ones) to get a frequency
        return self.BestArmPulls[environmentId][policyId, :] / (float(self.repetitions) * self.times)

    def getRewards(self, policyId, environmentId=0):
        return self.rewards[policyId, environmentId, :] / float(self.repetitions)

    def getMaxRewards(self, environmentId=0):
        return np.max(self.rewards[:, environmentId, :] / float(self.repetitions))

    def getCumulatedRegret(self, policyId, environmentId=0):
        # return self.times * self.envs[environmentId].maxArm - np.cumsum(self.getRewards(policyId, environmentId))
        return np.cumsum(self.envs[environmentId].maxArm - self.getRewards(policyId, environmentId))

    def getAverageRewards(self, policyId, environmentId=0):
        return np.cumsum(self.getRewards(policyId, environmentId)) / self.times

    def getRewardsSquared(self, policyId, environmentId=0):
        return self.rewardsSquared[policyId, environmentId, :] / float(self.repetitions)

    def getSTDRegret(self, policyId, environmentId=0, meanRegret=False):
        X = self.times
        YMAX = self.getMaxRewards(environmentId=environmentId)
        Y = self.getRewards(policyId, environmentId)
        Y2 = self.getRewardsSquared(policyId, environmentId)
        if meanRegret:  # Cumulated expectation on time
            Ycum2 = (np.cumsum(Y) / X)**2
            Y2cum = np.cumsum(Y2) / X
            assert np.all(Y2cum >= Ycum2), "Error: getSTDRegret found a nan value in the standard deviation (ie a point where Y2cum < Ycum2)."
            stdY = np.sqrt(Y2cum - Ycum2)
            YMAX *= 20  # XXX make it look smaller, for the plots
        else:  # Expectation on nb of repetitions
            # https://en.wikipedia.org/wiki/Algebraic_formula_for_the_variance#In_terms_of_raw_moments
            # std(Y) = sqrt( E[Y**2] - E[Y]**2 )
            stdY = np.cumsum(np.sqrt(Y2 - Y**2))
            YMAX *= np.log(2 + self.duration)  # Normalize the std variation
            YMAX *= 50  # XXX make it look larger, for the plots
        # Renormalize this standard deviation
        stdY /= YMAX
        return stdY

    # --- Plotting methods

    def plotRegrets(self, environmentId,
                    savefig=None, meanRegret=False, plotSTD=False, semilogx=False, normalizedRegret=False
                    ):
        plt.figure()
        ymin = 0
        colors = palette(self.nbPolicies)
        markers = makemarkers(self.nbPolicies)
        markers_on = int(self.duration / 10.0) * np.arange(0, 10)
        delta_marker = 1 + int(self.duration / 200.0)  # XXX put back 0 if needed
        X = self.times - 1
        for i, policy in enumerate(self.policies):
            if meanRegret:
                Y = self.getAverageRewards(i, environmentId)
            else:
                Y = self.getCumulatedRegret(i, environmentId)
                if normalizedRegret:
                    Y /= np.log(2 + X)   # XXX prevent /0
            ymin = min(ymin, np.min(Y))
            lw = 4 if str(policy)[:4] == 'Aggr' else 2
            if semilogx:
                plt.semilogx(X, Y, label=str(policy), color=colors[i], marker=markers[i], markevery=(delta_marker * (i % self.envs[environmentId].nbArms) + markers_on), lw=lw)
            else:
                plt.plot(X, Y, label=str(policy), color=colors[i], marker=markers[i], markevery=(delta_marker * (i % self.envs[environmentId].nbArms) + markers_on), lw=lw)
            # XXX plt.fill_between http://matplotlib.org/users/recipes.html#fill-between-and-alpha instead of plt.errorbar
            if plotSTD and self.repetitions > 1:
                stdY = self.getSTDRegret(i, environmentId, meanRegret=meanRegret)
                # stdY = 0.01 * np.max(np.abs(Y))  # DEBUG: 1% std to see it
                if normalizedRegret:
                    stdY /= np.log(2 + X)
                plt.fill_between(X, Y - stdY, Y + stdY, facecolor=colors[i], alpha=0.4)
                # plt.errorbar(X, Y, yerr=stdY, label=str(policy), color=colors[i], marker=markers[i], markevery=(delta_marker * (i % self.envs[environmentId].nbArms) + markers_on), alpha=0.9)
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}$".format(self.horizon))
        lowerbound = self.envs[environmentId].lowerbound()
        # ymax = max(plt.ylim()[1], 1)  # XXX Not smart, if maxmu = 0.1 we don't see anything
        ymax = plt.ylim()[1]
        plt.ylim(ymin, ymax)
        if meanRegret:
            # We plot a horizontal line ----- at the best arm mean
            plt.plot(X, self.envs[environmentId].maxArm * np.ones_like(X), 'k--', label="Mean of the best arm = ${:.3g}$".format(self.envs[environmentId].maxArm))
            plt.legend(loc='lower right', numpoints=1, fancybox=True, framealpha=0.7)  # http://matplotlib.org/users/recipes.html#transparent-fancy-legends
            plt.ylabel(r"Mean reward, average on time $\tilde{r}_t = \frac{1}{t} \sum_{s = 1}^{t} \mathbb{E}_{%d}[r_s]$" % (self.repetitions,))
            plt.title("Mean rewards for different bandit algorithms, averaged ${}$ times\n{} arms: ${}${}".format(self.repetitions, self.envs[environmentId].nbArms, repr(self.envs[environmentId].arms), signature))
        elif normalizedRegret:
            # We also plot the Lai & Robbins lower bound
            plt.plot(X, lowerbound * np.ones_like(X), 'k-', label="Lai & Robbins lower bound = ${:.3g}$".format(lowerbound), lw=3)
            plt.legend(loc='upper left', numpoints=1, fancybox=True, framealpha=0.7)  # http://matplotlib.org/users/recipes.html#transparent-fancy-legends
            plt.ylabel(r"Normalized cumulated regret $\frac{R_t}{\log t} = \frac{t}{\log t} \mu^* - \frac{1}{\log t}\sum_{s = 1}^{t} \mathbb{E}_{%d}[r_s]$" % (self.repetitions,))
            plt.title("Normalized cumulated regrets for different bandit algorithms, averaged ${}$ times\n{} arms: ${}${}".format(self.repetitions, self.envs[environmentId].nbArms, repr(self.envs[environmentId].arms), signature))
        else:
            # We also plot the Lai & Robbins lower bound
            plt.plot(X, lowerbound * np.log(1 + X), 'k-', label=r"Lai & Robbins lower bound = ${:.3g}\; \log(T)$".format(lowerbound), lw=3)
            plt.legend(loc='upper left', numpoints=1, fancybox=True, framealpha=0.7)  # http://matplotlib.org/users/recipes.html#transparent-fancy-legends
            plt.ylabel(r"Cumulated regret $R_t = t \mu^* - \sum_{s = 1}^{t} \mathbb{E}_{%d}[r_s]$" % (self.repetitions,))
            plt.title("Cumulated regrets for different bandit algorithms, averaged ${}$ times\n{} arms: ${}${}".format(self.repetitions, self.envs[environmentId].nbArms, repr(self.envs[environmentId].arms), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI, bbox_inches='tight')
        plt.show()

    def plotBestArmPulls(self, environmentId, savefig=None):
        plt.figure()
        colors = palette(self.nbPolicies)
        markers = makemarkers(self.nbPolicies)
        markers_on = int(self.duration / 10.0) * np.arange(0, 10)
        delta_marker = 1 + int(self.duration / 200.0)  # XXX put back 0 if needed
        X = self.times
        for i, policy in enumerate(self.policies):
            Y = self.getBestArmPulls(i, environmentId)
            lw = 5 if str(policy)[:4] == 'Aggr' else 3
            plt.plot(X, Y, label=str(policy), color=colors[i], marker=markers[i], markevery=(delta_marker * (i % self.envs[environmentId].nbArms) + markers_on), lw=lw)
        plt.legend(loc='best', numpoints=1, fancybox=True, framealpha=0.7)  # http://matplotlib.org/users/recipes.html#transparent-fancy-legends
        plt.xlabel(r"Time steps $t = 1 .. T$, horizon $T = {}$".format(self.horizon))
        plt.ylim(-0.03, 1.03)
        plt.ylabel(r"Frequency of pulls of the optimal arm")
        plt.title("Best arm pulls frequency for different bandit algorithms, averaged ${}$ times\n{} arms: ${}${}".format(self.repetitions, self.envs[environmentId].nbArms, repr(self.envs[environmentId].arms), signature))
        maximizeWindow()
        if savefig is not None:
            print("Saving to", savefig, "...")
            plt.savefig(savefig, dpi=DPI, bbox_inches='tight')
        plt.show()

    def printFinalRanking(self, environmentId=0):
        assert 0 < self.averageOn < 1, "Error, the parameter averageOn of a EvaluatorMultiPlayers classs has to be in (0, 1) strictly, but is = {} here ...".format(self.averageOn)
        print("\nFinal ranking for this environment #{} :".format(environmentId))
        nbPolicies = self.nbPolicies
        lastY = np.zeros(nbPolicies)
        for i, policy in enumerate(self.policies):
            Y = self.getCumulatedRegret(i, environmentId)
            if self.finalRanksOnAverage:
                lastY[i] = np.mean(Y[-int(self.averageOn * self.duration)])   # get average value during the last 0.5% of the iterations
            else:
                lastY[i] = Y[-1]  # get the last value
        # Sort lastY and give ranking
        index_of_sorting = np.argsort(lastY)
        for i, k in enumerate(index_of_sorting):
            policy = self.policies[k]
            print("- Policy '{}'\twas ranked\t{} / {} for this simulation (last regret = {:g}).".format(str(policy), i + 1, nbPolicies, lastY[k]))
        return lastY, index_of_sorting


# Helper function for the parallelization

# @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof
def delayed_play(env, policy, horizon, delta_t_save=1,
                 random_shuffle=random_shuffle, random_invert=random_invert, nb_random_events=nb_random_events,
                 seed=None, allrewards=None):
    # XXX Try to give a unique seed to random & numpy.random for each call of this function
    try:
        random.seed(seed)
        np.random.seed(seed)
    except SystemError:
        print("Warning: setting random.seed and np.random.seed seems to not be available. Are you using Windows?")  # XXX
    # We have to deepcopy because this function is Parallel-ized
    env = deepcopy(env)
    policy = deepcopy(policy)
    # Start game
    policy.startGame()
    result = Result(env.nbArms, horizon, delta_t_save=delta_t_save)  # One Result object, for every policy
    # XXX Experimental support for random events: shuffling or inverting the list of arms, at these time steps
    t_events = [i * int(horizon / float(nb_random_events)) for i in range(nb_random_events)]
    if nb_random_events is None or nb_random_events <= 0:
        random_shuffle = False
        random_invert = False

    for t in range(horizon):
        choice = policy.choice()

        # XXX do this quicker!
        if allrewards is None:
            reward = env.arms[choice].draw(t)
        else:
            reward = allrewards[choice, t]

        policy.getReward(choice, reward)

        # if t % delta_t_save == 0:  # XXX inefficient and does not work yet
        #     # if delta_t_save > 1: print("t =", t, "delta_t_save =", delta_t_save, " : saving ...")  # DEBUG
        #     result.store(t, choice, reward)
        result.store(t, choice, reward)

        # XXX Experimental : shuffle the arms at the middle of the simulation
        if random_shuffle:
            if t in t_events:  # XXX improve this: it is slow to test 'in <a list>', faster to compute a 't % ...'
                random.shuffle(env.arms)
                # print("Shuffling the arms ...")  # DEBUG
        # XXX Experimental : invert the order of the arms at the middle of the simulation
        if random_invert:
            if t in t_events:  # XXX improve this: it is slow to test 'in <a list>', faster to compute a 't % ...'
                env.arms = env.arms[::-1]
                # print("Inverting the order of the arms ...")  # DEBUG
    return result
