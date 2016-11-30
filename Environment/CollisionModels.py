# -*- coding: utf-8 -*-
""" Define some basic collision models.

Collision models are generic functions, taking:

 - the time: 't'
 - the arms of the current environment: 'arms'
 - the list of players: 'players'
 - the numpy array of their choices: 'choices'
 - the numpy array to store their rewards: 'rewards'
 - the numpy array to store their pulls: 'pulls'
 - the numpy array to store their collisions: 'collisions'

As far as now, there is 3 different collision models implemented:

 - noCollision: simple collision model where all players sample it and receive the reward.
 - onlyUniqUserGetsReward: simple collision model, where only the players alone on one arm sample it and receive the reward (default).
 - rewardIsSharedUniformly: in case of more than one player on one arm, only one player (uniform choice) can sample it and receive the reward.
 - closerUserGetsReward: in case of more than one player on one arm, only the closer player can sample it and receive the reward. It can take, or create if not given, a random distance of each player to the base station (random number in [0, 1]).
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np


def onlyUniqUserGetsReward(t, arms, players, choices, rewards, pulls, collisions):
    """ Simple collision model where only the players alone on one arm sample it and receive the reward.

    - This is the default collision model, cf. https://arxiv.org/abs/0910.2065v3 collision model 1.
    - The numpy array 'choices' is increased according to the number of users who collided (it is NOT binary).
    """
    nbCollisions = [np.sum(choices == arm) - 1 for arm in range(len(arms))]
    # print("nbCollisions =", nbCollisions)  # DEBUG
    # if np.max(nbCollisions) >= 1:  # DEBUG
    #     print("- onlyUniqUserGetsReward: some collisions on channels {} at time t = {} ...".format(np.nonzero(np.array(nbCollisions) >= 1)[0], t))  # DEBUG
    for i, player in enumerate(players):
        if nbCollisions[choices[i]] < 1:  # No collision
            rewards[i] = arms[choices[i]].draw(t)
            player.getReward(choices[i], rewards[i])
            pulls[i, choices[i]] += 1
        else:
            # print("  - 1 collision on channel {} : {} other users chosed it at time t = {} ...".format(choices[i], nbCollisions[choices[i]], t))  # DEBUG
            collisions[choices[i]] += 1  # Should be counted here, onlyUniqUserGetsReward
            # XXX player.handleCollision(t, choices[i]) should be called to inform the user that there were a collision
            if hasattr(player, 'handleCollision'):
                player.handleCollision(t, choices[i])
                # TODO had this to some multi-players policies
                # Example: ALOHA will not visit an arm for some time after seeing a collision!
            else:
                # XXX should player.getReward() be called with a reward = 0 when there is collisions (to change the internals memory of the player) ?
                player.getReward(choices[i], 0)  # FIXME Strong assumption on the model


# Default collision model to use
defaultCollisionModel = onlyUniqUserGetsReward


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
    """ Simple collision model where:

    - The players alone on one arm sample it and receive the reward.
    - In case of more than one player on one arm, only one player (uniform choice) can sample it and receive the reward.
    """
    # For each arm, explore who chosed it
    for arm in range(len(arms)):
        # If he is alone, sure to be chosen, otherwise only one get randomly chosen
        players_who_chosed_it = np.nonzero(choices == arm)[0]
        # print("players_who_chosed_it =", players_who_chosed_it)  # DEBUG
        # print("np.shape(players_who_chosed_it) =", np.shape(players_who_chosed_it))  # DEBUG
        # if len(players_who_chosed_it) > 1:  # DEBUG
        #     print("- rewardIsSharedUniformly: for arm {}, {} users won't have a reward at time t = {} ...".format(arm, len(players_who_chosed_it) - 1, t))  # DEBUG
        if np.size(players_who_chosed_it) > 0:
            collisions[arm] += np.size(players_who_chosed_it) - 1   # Increase nb of collisions for nb of player who chosed it, minus 1 (eg, if 1 then no collision, if 2 then one collision)
            i = np.random.choice(players_who_chosed_it)
            rewards[i] = arms[arm].draw(t)
            players[i].getReward(arm, rewards[i])
            pulls[i, arm] += 1
            for j in players_who_chosed_it:
                if i != j:
                    # XXX player.handleCollision(t, arm) should be called to inform the user that there were a collision
                    if hasattr(players[j], 'handleCollision'):
                        players[j].handleCollision(t, arm)
                        # TODO had this to some multi-players policies
                        # Example: ALOHA will not visit an arm for some time after seeing a collision!
                    else:
                        # XXX should players[j].getReward() be called with a reward = 0 when there is collisions (to change the internals memory of the player) ?
                        players[j].getReward(arm, 0)  # FIXME Strong assumption on the model


def closerUserGetsReward(t, arms, players, choices, rewards, pulls, collisions, distances=None):
    """ Simple collision model where:

    - The players alone on one arm sample it and receive the reward.
    - In case of more than one player on one arm, only the closer player can sample it and receive the reward. It can take, or create if not given, a distance of each player to the base station (numbers in [0, 1]).
    - If distances is not given, it is either generated randomly (random numbers in [0, 1]) or is a linspace of nbPlayers values in (0, 1), equally spacen (default).
    """
    if distances is None:  # Uniformly spacen distances, in (0, 1)
        # FIXME find a way to generate the distances only once, from the function side, and then use it
        distances = np.linspace(0, 1, len(players) + 1, endpoint=False)[1:]
    if distances == 'random':  # Or fully uniform
        distances = np.random.random_sample(len(players))
    # For each arm, explore who chosed it
    for arm in range(len(arms)):
        # If he is alone, sure to be chosen, otherwise only the closest one can sample
        players_who_chosed_it = np.nonzero(choices == arm)[0]
        # print("players_who_chosed_it =", players_who_chosed_it)  # DEBUG
        # if np.size(players_who_chosed_it) > 1:  # DEBUG
        #     print("- rewardIsSharedUniformly: for arm {}, {} users won't have a reward at time t = {} ...".format(arm, np.size(players_who_chosed_it) - 1, t))  # DEBUG
        if np.size(players_who_chosed_it) > 0:
            collisions[arm] += np.size(players_who_chosed_it) - 1   # Increase nb of collisions for nb of player who chosed it, minus 1 (eg, if 1 then no collision, if 2 then one collision as the closest gets it)
            distancesChosen = distances[players_who_chosed_it]
            smaller_distance = np.min(distancesChosen)
            # print("Using distances to chose the user who can pull arm {} : only users at the minimal distance = {} can transmit ...".format(arm, smaller_distance))  # DEBUG
            if np.count_nonzero(distancesChosen == smaller_distance) == 1:
                i = players_who_chosed_it[np.argmin(distancesChosen)]
                # print("Only one user is at minimal distance, of index i =", i)  # DEBUG
            else:   # XXX very low probability, if the distances are randomly chosen
                i = players_who_chosed_it[np.random.choice(np.argwhere(distancesChosen == smaller_distance))]
                print("  Randomly choosing one user at minimal distance = {:.4f}, among {}... Index i = {} was chosed !".format(smaller_distance, np.count_nonzero(distancesChosen == smaller_distance), i + 1))  # DEBUG
            # Player i can pull the arm
            rewards[i] = arms[arm].draw(t)
            players[i].getReward(arm, rewards[i])
            pulls[i, arm] += 1
            for j in players_who_chosed_it:
                # The other players cannot
                if i != j:
                    # XXX player.handleCollision(t, arm) should be called to inform the user that there were a collision
                    if hasattr(players[j], 'handleCollision'):
                        players[j].handleCollision(t, arm)
                        # TODO had this to some multi-players policies
                        # Example: ALOHA will not visit an arm for some time after seeing a collision!
                    else:
                        # XXX should players[j].getReward() be called with a reward = 0 when there is collisions (to change the internals memory of the player) ?
                        players[j].getReward(arm, 0)  # FIXME Strong assumption on the model


# List of possible collision models
collision_models = [onlyUniqUserGetsReward, noCollision, rewardIsSharedUniformly, closerUserGetsReward, ]
