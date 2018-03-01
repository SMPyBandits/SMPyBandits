# -*- coding: utf-8 -*-
""" Define the different collision models.

Collision models are generic functions, taking:

- the time: 't'
- the arms of the current environment: 'arms'
- the list of players: 'players'
- the numpy array of their choices: 'choices'
- the numpy array to store their rewards: 'rewards'
- the numpy array to store their pulls: 'pulls'
- the numpy array to store their collisions: 'collisions'

As far as now, there is 4 different collision models implemented:

- :func:`noCollision`: simple collision model where all players sample it and receive the reward.
- :func:`onlyUniqUserGetsReward`: simple collision model, where only the players alone on one arm sample it and receive the reward (default).
- :func:`rewardIsSharedUniformly`: in case of more than one player on one arm, only one player (uniform choice) can sample it and receive the reward.
- :func:`closerUserGetsReward`: in case of more than one player on one arm, only the closer player can sample it and receive the reward. It can take, or create if not given, a random distance of each player to the base station (random number in [0, 1]).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

try:
    from functools import lru_cache  # Only for Python 3.2+
except ImportError:
    print("ImportError: functools.lru_cache is not available, using a fake implementation.")

    def lru_cache(maxsize=128, typed=False):
        """ Fake implementation of functools.lru_cache, not available in Python 2."""

        def lru_cache_internal(f):
            """Fake wrapped f, in fact it's just f."""
            return f
        return lru_cache_internal

import numpy as np


def handleCollision_or_getZeroReward(player, arm, lower=0):
    """ If the player has a method handleCollision, it is called, otherwise a reward of lower is given to the player for that arm.
    """
    # player.handleCollision(arm) is called to inform the user that there were a collision
    if hasattr(player, 'handleCollision'):
        player.handleCollision(arm)
    else:
        # XXX Should player.getReward() be called with a reward = 0 when there is collisions (to change the internals memory of the player) ?
        player.getReward(arm, lower)  # XXX Strong assumption on the model


def onlyUniqUserGetsReward(t, arms, players, choices, rewards, pulls, collisions):
    """ Simple collision model where only the players alone on one arm samples it and receives the reward.

    - This is the default collision model, cf. [[Multi-Player Bandits Revisited, Lilian Besson and Emilie Kaufmann, 2017]](https://hal.inria.fr/hal-01629733).
    - The numpy array 'choices' is increased according to the number of users who collided (it is NOT binary).
    """
    # First, sense in all the arms
    sensing = [a.draw(t) for a in arms]
    # XXX Yes, I know, it's suboptimal to sample each arm even if no player chose it
    # But a quick benchmark showed it was quicker than
    # sensing = [a.draw(t) for i,a in enumerate(arms) if nbCollisions[i]>=0]

    nbCollisions = np.bincount(choices, minlength=len(arms)) - 1
    # print("onlyUniqUserGetsReward() at time t = {}, nbCollisions = {}.".format(t, nbCollisions))  # DEBUG

    # if np.max(nbCollisions) >= 1:  # DEBUG
    #     print("- onlyUniqUserGetsReward: some collisions on channels {} at time t = {} ...".format(np.nonzero(np.array(nbCollisions) >= 1)[0], t))  # DEBUG
    for i, player in enumerate(players):  # Loop is needed because player is needed
        # FIXED pulls counts the number of selection, not the number of successful selection!! HUGE BUG! See https://github.com/SMPyBandits/SMPyBandits/issues/33
        pulls[i, choices[i]] += 1
        if nbCollisions[choices[i]] < 1:  # No collision
            player.getReward(choices[i], sensing[choices[i]])  # Observing *sensing*
            rewards[i] = sensing[choices[i]]  # Storing actual rewards
        else:
            # print("  - 1 collision on channel {} : {} other users chose it at time t = {} ...".format(choices[i], nbCollisions[choices[i]], t))  # DEBUG
            collisions[choices[i]] += 1  # Should be counted here, onlyUniqUserGetsReward
            # handleCollision_or_getZeroReward(player, choices[i])  # NOPE
            player.handleCollision(choices[i], sensing[choices[i]])  # Observing *sensing* but collision
            # If learning is done on sensing, handleCollision uses this reward
            # But if learning is done on ACK, handleCollision does not use this reward


# Default collision model to use
defaultCollisionModel = onlyUniqUserGetsReward


def onlyUniqUserGetsRewardSparse(t, arms, players, choices, rewards, pulls, collisions):
    """ Simple collision model where only the players alone on one arm samples it and receives the reward.

    - This is the default collision model, cf. [[Multi-Player Bandits Revisited, Lilian Besson and Emilie Kaufmann, 2017]](https://hal.inria.fr/hal-01629733).
    - The numpy array 'choices' is increased according to the number of users who collided (it is NOT binary).
    - Support for player non activated, by choosing a negative index.
    """
    # First, sense in all the arms
    sensing = [a.draw(t) for a in arms]

    nbCollisions = np.bincount(choices[choices >= 0], minlength=len(arms)) - 1
    # print("onlyUniqUserGetsRewardSparse() at time t = {}, nbCollisions = {}.".format(t, nbCollisions))  # DEBUG

    # if np.max(nbCollisions) >= 1:  # DEBUG
    #     print("- onlyUniqUserGetsRewardSparse: some collisions on channels {} at time t = {} ...".format(np.nonzero(np.array(nbCollisions) >= 1)[0], t))  # DEBUG
    for i, player in enumerate(players):  # Loop is needed because player is needed
        # FIXED pulls counts the number of selection, not the number of successful selection!! HUGE BUG! See https://github.com/SMPyBandits/SMPyBandits/issues/33
        if choices[i] >= 0:
            pulls[i, choices[i]] += 1
            if nbCollisions[choices[i]] < 1:  # No collision
                player.getReward(choices[i], sensing[choices[i]])  # Observing *sensing*
                rewards[i] = sensing[choices[i]]  # Storing actual rewards
            else:
                # print("  - 1 collision on channel {} : {} other users chose it at time t = {} ...".format(choices[i], nbCollisions[choices[i]], t))  # DEBUG
                collisions[choices[i]] += 1  # Should be counted here, onlyUniqUserGetsRewardSparse
                # handleCollision_or_getZeroReward(player, choices[i])  # NOPE
                player.handleCollision(choices[i], sensing[choices[i]])  # Observing *sensing* but collision
                # If learning is done on sensing, handleCollision uses this reward
                # But if learning is done on ACK, handleCollision does not use this reward


def allGetRewardsAndUseCollision(t, arms, players, choices, rewards, pulls, collisions):
    """ A variant of the first simple collision model where all players sample their arm, receive their rewards, and are informed of the collisions.


    .. note:: it is NOT the one we consider, and so our lower-bound on centralized regret is wrong (users don't care about collisions for their internal rewards so regret does not take collisions into account!)

    - This is the NOT default collision model, cf. [Liu & Zhao, 2009](https://arxiv.org/abs/0910.2065v3) collision model 1.
    - The numpy array 'choices' is increased according to the number of users who collided (it is NOT binary).
    """
    nbCollisions = np.bincount(choices, minlength=len(arms)) - 1  # XXX this is faster!
    # print("allGetRewardsAndUseCollision() at time t = {}, nbCollisions = {}.".format(t, nbCollisions))  # DEBUG
    # if np.max(nbCollisions) >= 1:  # DEBUG
    #     print("- allGetRewardsAndUseCollision: some collisions on channels {} at time t = {} ...".format(np.nonzero(np.array(nbCollisions) >= 1)[0], t))  # DEBUG
    for i, player in enumerate(players):  # Loop is needed because player is needed
        # FIXED pulls counts the number of selection, not the number of successful selection!! HUGE BUG! See https://github.com/SMPyBandits/SMPyBandits/issues/33
        pulls[i, choices[i]] += 1

        rewards[i] = arms[choices[i]].draw(t)
        player.getReward(choices[i], rewards[i])

        if nbCollisions[choices[i]] >= 1:  # If collision
            # print("  - 1 collision on channel {} : {} other users chose it at time t = {} ...".format(choices[i], nbCollisions[choices[i]], t))  # DEBUG
            collisions[choices[i]] += 1  # Should be counted here, allGetRewardsAndUseCollision
            player.handleCollision(choices[i])  # FIXED


def noCollision(t, arms, players, choices, rewards, pulls, collisions):
    """ Simple collision model where all players sample it and receive the reward.

    - It corresponds to the single-player simulation: each player is a policy, compared without collision.
    - The numpy array 'collisions' is not modified.
    """
    for i, player in enumerate(players):
        rewards[i] = arms[choices[i]].draw(t)
        player.getReward(choices[i], rewards[i])
        pulls[i, choices[i]] += 1
        # collisions[choices[i]] += 0  # that's the idea, but useless to do it


def rewardIsSharedUniformly(t, arms, players, choices, rewards, pulls, collisions):
    """ Less simple collision model where:

    - The players alone on one arm sample it and receive the reward.
    - In case of more than one player on one arm, only one player (uniform choice) can sample it and receive the reward. It is chosen by the base station.


    .. Note:: it can also model a choice from the users point of view: in a time frame (eg. 1 second), when there is a collision, each colliding user chose (uniformly) a random small time offset (eg. 20 ms), and start sensing + emitting again after that time. The first one to sense is alone, it transmits, and the next ones find the channel used when sensing. So only one player is transmitting, and from the base station point of view, it is the same as if it was chosen uniformly among the colliding users.

    """
    # For each arm, explore who chose it
    for armId, arm in enumerate(arms):
        # If he is alone, sure to be chosen, otherwise only one get randomly chosen
        players_who_chose_it = np.nonzero(choices == armId)[0]
        # print("players_who_chose_it =", players_who_chose_it)  # DEBUG
        # print("np.shape(players_who_chose_it) =", np.shape(players_who_chose_it))  # DEBUG
        # if len(players_who_chose_it) > 1:  # DEBUG
        #     print("- rewardIsSharedUniformly: for arm {}, {} users won't have a reward at time t = {} ...".format(armId, len(players_who_chose_it) - 1, t))  # DEBUG
        if np.size(players_who_chose_it) > 0:
            collisions[armId] += np.size(players_who_chose_it) - 1   # Increase nb of collisions for nb of player who chose it, minus 1 (eg, if 1 then no collision, if 2 then one collision)
            i = np.random.choice(players_who_chose_it)
            rewards[i] = arm.draw(t)
            players[i].getReward(armId, rewards[i])
            pulls[i, armId] += 1
            for j in players_who_chose_it:
                if i != j:
                    handleCollision_or_getZeroReward(players[j], armId)


# XXX Using a cache to not regenerate a random vector of distances. Siooooux!
@lru_cache(maxsize=None, typed=False)  # XXX size is NOT bounded... bad!
def random_distances(nbPlayers):
    """ Get a random vector of distances."""
    distances = np.random.random_sample(nbPlayers)
    print("I just generated a new distances vector, for {} players : distances = {} ...".format(nbPlayers, distances))  # DEBUG
    return distances


def closerUserGetsReward(t, arms, players, choices, rewards, pulls, collisions, distances='uniform'):
    """ Simple collision model where:

    - The players alone on one arm sample it and receive the reward.
    - In case of more than one player on one arm, only the closer player can sample it and receive the reward. It can take, or create if not given, a distance of each player to the base station (numbers in [0, 1]).
    - If distances is not given, it is either generated randomly (random numbers in [0, 1]) or is a linspace of nbPlayers values in (0, 1), equally spacen (default).

    .. note:: This kind of effects is known in telecommunication as the Near-Far effect or the Capture effect [Roberts, 1975](https://dl.acm.org/citation.cfm?id=1024920)
    """
    if distances is None or (isinstance(distances, str) and distances == 'uniform'):  # Uniformly spacen distances, in (0, 1)
        distances = np.linspace(0, 1, len(players) + 1, endpoint=False)[1:]
    elif isinstance(distances, str) and distances == 'random':  # Or fully uniform
        distances = random_distances(len(players))
    # For each arm, explore who chose it
    for armId, arm in enumerate(arms):
        # If he is alone, sure to be chosen, otherwise only the closest one can sample
        players_who_chose_it = np.nonzero(choices == armId)[0]
        # print("players_who_chose_it =", players_who_chose_it)  # DEBUG
        # if np.size(players_who_chose_it) > 1:  # DEBUG
        #     print("- rewardIsSharedUniformly: for arm {}, {} users won't have a reward at time t = {} ...".format(armId, np.size(players_who_chose_it) - 1, t))  # DEBUG
        if np.size(players_who_chose_it) > 0:
            collisions[armId] += np.size(players_who_chose_it) - 1   # Increase nb of collisions for nb of player who chose it, minus 1 (eg, if 1 then no collision, if 2 then one collision as the closest gets it)
            distancesChosen = distances[players_who_chose_it]
            smaller_distance = np.min(distancesChosen)
            # print("Using distances to chose the user who can pull arm {} : only users at the minimal distance = {} can transmit ...".format(armId, smaller_distance))  # DEBUG
            if np.count_nonzero(distancesChosen == smaller_distance) == 1:
                i = players_who_chose_it[np.argmin(distancesChosen)]
                # print("Only one user is at minimal distance, of index i =", i)  # DEBUG
            else:   # XXX very low probability, if the distances are randomly chosen
                i = players_who_chose_it[np.random.choice(np.nonzero(distancesChosen == smaller_distance))]
                print("  Randomly choosing one user at minimal distance = {:.4g}, among {}... Index i = {} was chose !".format(smaller_distance, np.count_nonzero(distancesChosen == smaller_distance), i + 1))  # DEBUG
            # Player i can pull the armId
            rewards[i] = arm.draw(t)
            players[i].getReward(armId, rewards[i])
            pulls[i, armId] += 1
            for j in players_who_chose_it:
                # The other players cannot
                if i != j:
                    handleCollision_or_getZeroReward(players[j], armId)


#: List of possible collision models
collision_models = [
    onlyUniqUserGetsReward,
    onlyUniqUserGetsRewardSparse,
    allGetRewardsAndUseCollision,
    noCollision,
    rewardIsSharedUniformly,
    closerUserGetsReward,
]


#: Mapping of collision model names to True or False,
#: to know if a collision implies a lost communication or not in this model
full_lost_if_collision = {
    # Fake collision model
    "noCollision": False,
    # No lost communication in case of collision
    "closerUserGetsReward": False,
    # In average, no lost communication in case of collision
    "rewardIsSharedUniformly": False,
    # Lost communication in case of collision
    "onlyUniqUserGetsReward": True,
    "onlyUniqUserGetsRewardSparse": True,
    "allGetRewardsAndUseCollision": True,
}


#: Only export and expose the useful functions and constants defined here
__all__ = [
    "onlyUniqUserGetsReward",
    "onlyUniqUserGetsRewardSparse",
    "allGetRewardsAndUseCollision",
    "noCollision",
    "closerUserGetsReward",
    "rewardIsSharedUniformly",
    "defaultCollisionModel",
    "collision_models",
    "full_lost_if_collision"
]
