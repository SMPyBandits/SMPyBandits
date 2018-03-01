# -*- coding: utf-8 -*-
""" EvaluatorSparseMultiPlayers class to wrap and run the simulations, for the multi-players case with sparse activated players.
Lots of plotting methods, to have various visualizations. See documentation.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

# Generic imports
from copy import deepcopy
from re import search
import random
from random import random as uniform_in_zero_one
# Scientific imports
import numpy as np
import matplotlib.pyplot as plt
# import h5py
# Local imports, libraries
from .usejoblib import USE_JOBLIB, Parallel, delayed
from .usetqdm import USE_TQDM, tqdm
# Local imports, tools and config
from .plotsettings import BBOX_INCHES, signature, maximizeWindow, palette, makemarkers, add_percent_formatter, wraptext, wraplatex, legend, show_and_save, nrows_ncols, addTextForWorstCases
from .sortedDistance import weightedDistance, manhattan, kendalltau, spearmanr, gestalt, meanDistance, sortedDistance
from .fairnessMeasures import amplitude_fairness, std_fairness, rajjain_fairness, mean_fairness, fairnessMeasure, fairness_mapping
# Local imports, objects and functions
from .CollisionModels import onlyUniqUserGetsRewardSparse, full_lost_if_collision
from .MAB import MAB, MarkovianMAB, DynamicMAB
from .ResultMultiPlayers import ResultMultiPlayers
# Inheritance
from .EvaluatorMultiPlayers import EvaluatorMultiPlayers, _extract


REPETITIONS = 1  #: Default nb of repetitions
ACTIVATION = 1  #: Default probability of activation
DELTA_T_PLOT = 50  #: Default sampling rate for plotting

MORE_ACCURATE = False          #: Use the count of selections instead of rewards for a more accurate mean/std reward measure.
MORE_ACCURATE = True           #: Use the count of selections instead of rewards for a more accurate mean/std reward measure.

FINAL_RANKS_ON_AVERAGE = True  #: Default value for ``finalRanksOnAverage``
USE_JOBLIB_FOR_POLICIES = False  #: Default value for ``useJoblibForPolicies``. Does not speed up to use it (too much overhead in using too much threads); so it should really be disabled.
PICKLE_IT = True  #: Default value for ``pickleit`` for saving the figures. If True, then all ``plt.figure`` object are saved (in pickle format).

# --- Class EvaluatorSparseMultiPlayers

class EvaluatorSparseMultiPlayers(EvaluatorMultiPlayers):
    """ Evaluator class to run the simulations, for the multi-players case.
    """

    def __init__(self, configuration,
                 moreAccurate=MORE_ACCURATE):
        super(EvaluatorSparseMultiPlayers, self).__init__(configuration, moreAccurate=moreAccurate)
        self.activations = self.cfg.get('activations', ACTIVATION)  #: Probability of activations
        assert np.min(self.activations) > 0 and np.max(self.activations) <= 1, "Error: probability of activations = {} were not all in (0, 1] ...".format(self.activations)  # DEBUG
        self.collisionModel = self.cfg.get('collisionModel', onlyUniqUserGetsRewardSparse)  #: Which collision model should be used
        self.full_lost_if_collision = full_lost_if_collision.get(self.collisionModel.__name__, True)  #: Is there a full loss of rewards if collision ? To compute the correct decomposition of regret
        print("Using collision model {} (function {}).\nMore details:\n{}".format(self.collisionModel.__name__, self.collisionModel, self.collisionModel.__doc__))

    # --- Start computation

    def startOneEnv(self, envId, env):
        """Simulate that env."""
        print("\n\nEvaluating environment:", repr(env))  # DEBUG
        self.players = []
        self.__initPlayers__(env)
        # Get the position of the best arms
        means = env.means
        bestarm = env.maxArm
        indexes_bestarm = np.nonzero(np.isclose(means, bestarm))[0]

        def store(r, repeatId):
            """Store the result of the experiment r."""
            self.rewards[envId] += np.cumsum(r.rewards, axis=1)  # cumsum on time
            self.lastCumRewards[envId][repeatId] = np.sum(r.rewards)  # sum on time and sum on policies
            self.pulls[envId] += r.pulls
            self.lastPulls[envId][:, :, repeatId] = r.pulls
            self.allPulls[envId] += r.allPulls
            self.collisions[envId] += r.collisions
            self.lastCumCollisions[envId][:, repeatId] = np.sum(r.collisions, axis=1)  # sum on time
            for playerId in range(self.nbPlayers):
                self.nbSwitchs[envId][playerId, 1:] += (np.diff(r.choices[playerId, :]) != 0)
                self.bestArmPulls[envId][playerId, :] += np.cumsum(np.in1d(r.choices[playerId, :], indexes_bestarm))
                # FIXME there is probably a bug in this computation
                self.freeTransmissions[envId][playerId, :] += np.array([r.choices[playerId, t] not in r.collisions[:, t] for t in range(self.horizon)])

        # Start now
        if self.useJoblib:
            seeds = np.random.randint(low=0, high=100 * self.repetitions, size=self.repetitions)
            repeatIdout = 0
            historyOfMeans = []
            for r in Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
                delayed(delayed_play)(env, self.players, self.horizon, self.collisionModel, self.activations, seed=seeds[repeatId], repeatId=repeatId)
                for repeatId in tqdm(range(self.repetitions), desc="Repeat||")
            ):
                historyOfMeans.append(r._means)
                store(r, repeatIdout)
                repeatIdout += 1
            if env.isDynamic:
                env._t += self.repetitions  # new self.repetitions draw!
                env._historyOfMeans = historyOfMeans
        else:
            for repeatId in tqdm(range(self.repetitions), desc="Repeat"):
                r = delayed_play(env, self.players, self.horizon, self.collisionModel, self.activations, repeatId=repeatId)
                store(r, repeatId)

    # --- Getter methods

    def getCentralizedRegret_LessAccurate(self, envId=0):
        """Compute the empirical centralized regret: cumsum on time of the mean rewards of the M best arms - cumsum on time of the empirical rewards obtained by the players, based on accumulated rewards."""
        meansArms = np.sort(self.envs[envId].means)
        sumBestMeans = self.envs[envId].sumBestMeans(min(self.envs[envId].nbArms, self.nbPlayers))
        # FIXED how to count it when there is more players than arms ?
        # FIXME it depends on the collision model !
        if self.envs[envId].nbArms < self.nbPlayers:
            # sure to have collisions, then the best strategy is to put all the collisions in the worse arm
            worseArm = np.min(meansArms)
            sumBestMeans -= worseArm  # This count the collisions
        averageBestRewards = self._times * sumBestMeans
        # And for the actual rewards, the collisions are counted in the rewards logged in self.getRewards
        actualRewards = np.sum(self.rewards[envId][:, :], axis=0) / float(self.repetitions)
        return averageBestRewards - actualRewards

    # --- Three terms in the regret

    def getFirstRegretTerm(self, envId=0):
        """Extract and compute the first term :math:`(a)` in the centralized regret: losses due to pulling suboptimal arms."""
        means = self.envs[envId].means
        sortingIndex = np.argsort(means)
        means = np.sort(means)
        deltaMeansWorstArms = means[-min(self.envs[envId].nbArms, self.nbPlayers)] - means[:-min(self.envs[envId].nbArms, self.nbPlayers)]
        allPulls = self.allPulls[envId] / float(self.repetitions)  # Shape: (nbPlayers, nbArms, duration)
        allWorstPulls = allPulls[:, sortingIndex[:-min(self.envs[envId].nbArms, self.nbPlayers)], :]
        worstPulls = np.sum(allWorstPulls, axis=0)  # sum for all players
        losses = np.dot(deltaMeansWorstArms, worstPulls)  # Count and sum on k in Mworst
        firstRegretTerm = np.cumsum(losses)  # Accumulate losses
        return firstRegretTerm

    def getSecondRegretTerm(self, envId=0):
        """Extract and compute the second term :math:`(b)` in the centralized regret: losses due to not pulling optimal arms."""
        means = self.envs[envId].means
        sortingIndex = np.argsort(means)
        means = np.sort(means)
        deltaMeansBestArms = means[-min(self.envs[envId].nbArms, self.nbPlayers):] - means[-min(self.envs[envId].nbArms, self.nbPlayers)]
        allPulls = self.allPulls[envId] / float(self.repetitions)  # Shape: (nbPlayers, nbArms, duration)
        allBestPulls = allPulls[:, sortingIndex[-min(self.envs[envId].nbArms, self.nbPlayers):], :]
        bestMisses = 1 - np.sum(allBestPulls, axis=0)  # sum for all players
        losses = np.dot(deltaMeansBestArms, bestMisses)  # Count and sum on k in Mbest
        secondRegretTerm = np.cumsum(losses)  # Accumulate losses
        return secondRegretTerm

    def getThirdRegretTerm(self, envId=0):
        """Extract and compute the third term :math:`(c)` in the centralized regret: losses due to collisions."""
        means = self.envs[envId].means
        countCollisions = self.collisions[envId]   # Shape: (nbArms, duration)
        if not self.full_lost_if_collision:
            print("Warning: the collision model ({}) does *not* yield a loss in communication when colliding (one user can communicate, or in average one user can communicate), so countCollisions -= 1 for the 3rd regret term ...".format(self.collisionModel.__name__))  # DEBUG
            countCollisions = np.maximum(0, countCollisions - 1)
        losses = np.dot(means, countCollisions / float(self.repetitions))  # Count and sum on k in 1...K
        thirdRegretTerm = np.cumsum(losses)  # Accumulate losses
        return thirdRegretTerm

    def getCentralizedRegret_MoreAccurate(self, envId=0):
        """Compute the empirical centralized regret, based on counts of selections and not actual rewards."""
        return self.getFirstRegretTerm(envId=envId) + self.getSecondRegretTerm(envId=envId) + self.getThirdRegretTerm(envId=envId)

    def getCentralizedRegret(self, envId=0, moreAccurate=None):
        """Using either the more accurate or the less accurate regret count."""
        moreAccurate = moreAccurate if moreAccurate is not None else self.moreAccurate
        # print("Computing the vector of mean cumulated regret with '{}' accurate method...".format("more" if moreAccurate else "less"))  # DEBUG
        if moreAccurate:
            return self.getCentralizedRegret_MoreAccurate(envId=envId)
        else:
            return self.getCentralizedRegret_LessAccurate(envId=envId)

    # --- Last regrets

    def getLastRegrets_LessAccurate(self, envId=0):
        """Extract last regrets, based on accumulated rewards."""
        meansArms = np.sort(self.envs[envId].means)
        sumBestMeans = self.envs[envId].sumBestMeans(self.nbPlayers)
        # FIXED how to count it when there is more players than arms ?
        # FIXME it depends on the collision model !
        if self.envs[envId].nbArms < self.nbPlayers:
            # sure to have collisions, then the best strategy is to put all the collisions in the worse arm
            worseArm = np.min(meansArms)
            sumBestMeans -= worseArm  # This count the collisions
        return self.horizon * sumBestMeans - self.lastCumRewards[envId]

    def getAllLastWeightedSelections(self, envId=0):
        """Extract weighted count of selections."""
        all_last_weighted_selections = np.zeros(self.repetitions)
        lastCumCollisions = self.lastCumCollisions[envId]
        for armId, mean in enumerate(self.envs[envId].means):
            last_selections = np.sum(self.lastPulls[envId][:, armId, :], axis=0)  # sum on players
            all_last_weighted_selections += mean * (last_selections - lastCumCollisions[armId, :])
        return all_last_weighted_selections

    def getLastRegrets_MoreAccurate(self, envId=0):
        """Extract last regrets, based on counts of selections and not actual rewards."""
        meansArms = np.sort(self.envs[envId].means)
        sumBestMeans = self.envs[envId].sumBestMeans(self.nbPlayers)
        # FIXED how to count it when there is more players than arms ?
        # FIXME it depends on the collision model !
        if self.envs[envId].nbArms < self.nbPlayers:
            # sure to have collisions, then the best strategy is to put all the collisions in the worse arm
            worseArm = np.min(meansArms)
            sumBestMeans -= worseArm  # This count the collisions
        return self.horizon * sumBestMeans - self.getAllLastWeightedSelections(envId=envId)

    def getLastRegrets(self, envId=0, moreAccurate=None):
        """Using either the more accurate or the less accurate regret count."""
        moreAccurate = moreAccurate if moreAccurate is not None else self.moreAccurate
        # print("Computing the vector of last cumulated regrets (on repetitions) with '{}' accurate method...".format("more" if moreAccurate else "less"))  # DEBUG
        if moreAccurate:
            return self.getLastRegrets_MoreAccurate(envId=envId)
        else:
            return self.getLastRegrets_LessAccurate(envId=envId)

    def strPlayers(self, short=False, latex=True):
        """Get a string of the players and their activations probability for this environment."""
        listStrPlayersActivations = [("%s, $p=%s$" if latex else "%s, p=%s") % (_extract(str(player)), str(activation)) for (player, activation) in zip(self.players, self.activations)]
        if len(set(listStrPlayersActivations)) == 1:  # Unique user and unique activation
            if latex:
                text = r'${} \times$ {}'.format(self.nbPlayers, listStrPlayersActivations[0])
            else:
                text = r'{} x {}'.format(self.nbPlayers, listStrPlayersActivations[0])
        else:
            text = ', '.join(listStrPlayersActivations)
        text = wraptext(text)
        if not short:
            text = '{} players: {}'.format(self.nbPlayers, text)
        return text


def delayed_play(env, players, horizon, collisionModel, activations,
                 seed=None, repeatId=0):
    """Helper function for the parallelization."""
    # Give a unique seed to random & numpy.random for each call of this function
    try:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    except (ValueError, SystemError):
        print("Warning: setting random.seed and np.random.seed seems to not be available. Are you using Windows?")  # XXX
    means = env.means
    if env.isDynamic:
        means = env.newRandomArms()
    players = deepcopy(players)
    nbArms = env.nbArms
    nbPlayers = len(players)
    # Start game
    for player in players:
        player.startGame()
    # Store results
    result = ResultMultiPlayers(env.nbArms, horizon, nbPlayers, means=means)
    rewards = np.zeros(nbPlayers)
    choices = np.zeros(nbPlayers, dtype=int)
    pulls = np.zeros((nbPlayers, nbArms), dtype=int)
    collisions = np.zeros(nbArms, dtype=int)

    nbActivations = np.zeros(nbPlayers, dtype=int)

    prettyRange = tqdm(range(horizon), desc="Time t") if repeatId == 0 else range(horizon)
    for t in prettyRange:
        # Reset the array, faster than reallocating them!
        rewards.fill(0)
        choices.fill(-100000)
        pulls.fill(0)
        collisions.fill(0)
        # Decide who gets activated
        # # 1. pure iid Bernoulli activations, so sum(random_activations) == np.random.binomial(nbPlayers, activation) if activations are all the same
        # random_activations = np.random.random_sample(nbPlayers) <= activations
        # FIXME finish these experiments
        # 2. maybe first decide how many players from [0, nbArms] or [0, nbPlayers] are activated, then who
        # nb_activated_players = np.random.binomial(nbArms, np.mean(activations))
        nb_activated_players = np.random.binomial(nbPlayers, np.mean(activations))
        # who_is_activated = np.random.choice(nbPlayers, size=nb_activated_players, replace=False)
        who_is_activated = np.random.choice(nbPlayers, size=nb_activated_players, replace=False, p=np.asarray(activations)/np.sum(activations))
        random_activations = np.in1d(np.arange(nbPlayers), who_is_activated)
        # Every player decides which arm to pull
        for playerId, player in enumerate(players):
            # if with_proba(activations[playerId]):
            if random_activations[playerId]:
                nbActivations[playerId] += 1
                choices[playerId] = player.choice()
                # print(" Round t = \t{}, player \t#{:>2}/{} ({}) \tgot activated and chose : {} ...".format(t, playerId + 1, len(players), player, choices[playerId]))  # DEBUG
            # else:
            #     print(" Round t = \t{}, player \t#{:>2}/{} ({}) \tdid not get activated ...".format(t, playerId + 1, len(players), player))  # DEBUG

        # Then we decide if there is collisions and what to do why them
        # XXX It is here that the player may receive a reward, if there is no collisions
        collisionModel(t, env.arms, players, choices, rewards, pulls, collisions)

        # Finally we store the results
        result.store(t, choices, rewards, pulls, collisions)

    # Print the quality of estimation of arm ranking for this policy, just for 1st repetition
    if repeatId == 0:
        print("\nNumber of activations by players:")
        for playerId, player in enumerate(players):
            try:
                print("\nThe policy {} was activated {} times after {} steps...".format(player, nbActivations[playerId], horizon))
                order = player.estimatedOrder()
                print("Estimated order by the policy {} after {} steps: {} ...".format(player, horizon, order))
                print("  ==> Optimal arm identification: {:.2%} (relative success)...".format(weightedDistance(order, env.means, n=nbPlayers)))
                # print("  ==> Manhattan   distance from optimal ordering: {:.2%} (relative success)...".format(manhattan(order)))
                # print("  ==> Spearman    distance from optimal ordering: {:.2%} (relative success)...".format(spearmanr(order)))
                # print("  ==> Gestalt     distance from optimal ordering: {:.2%} (relative success)...".format(gestalt(order)))
                print("  ==> Mean distance from optimal ordering: {:.2%} (relative success)...".format(meanDistance(order)))
            except AttributeError:
                print("Unable to print the estimated ordering, no method estimatedOrder was found!")

    return result


def with_proba(proba):
    """`True` with probability = `proba`, `False` with probability = `1 - proba`.

    Examples:

    >>> tosses = [with_proba(0.6) for _ in range(10000)]; sum(tosses)
    6043
    >>> tosses = [with_proba(0.111) for _ in range(100000)]; sum(tosses)
    11162
    """
    return uniform_in_zero_one() <= proba
