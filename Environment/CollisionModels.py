# -*- coding: utf-8 -*-
""" Define some basic collision models.

Collision model are generic functions, taking:

 - the time: 't'
 - the arms of the current environment: 'arms'
 - the list of players: 'players'
 - the numpy array of their choices: 'choices'
 - the numpy array to store their rewards: 'rewards'
 - the numpy array to store their pulls: 'pulls'
 - the numpy array to store their collisions: 'collisions'
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

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
        if nbCollisions[choices[i]] < 1:
            rewards[i] = arms[choices[i]].draw(t)
            player.getReward(choices[i], rewards[i])
            pulls[i, choices[i]] += 1
        else:
            # print("  - 1 collision on channel {} : {} other users choosed it at time t = {} ...".format(choices[i], nbCollisions[choices[i]], t))  # DEBUG
            collisions[choices[i]] += 1
            # FIXME player[i].getReward() should be called with a reward = 0 when there is collisions (to change the internals memory of the player)
            player.getReward(choices[i], 0)


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


# Default collision model to use
# defaultCollisionModel = noCollision


def rewardIsSharedUniformly(t, arms, players, choices, rewards, pulls, collisions):
    """ Simple collision model where:

    - The players alone on one arm sample it and receive the reward.
    - In case of more than one player on one arm, only one player (uniform choice) can sample it and receive the reward.
    """
    for arm in range(len(arms)):
        # If he is alone, sure to be chosen, otherwise only one get randomly chosen
        players_who_chosed_it = np.nonzero(choices == arm)[0]
        # print("players_who_chosed_it =", players_who_chosed_it)  # DEBUG
        # print("np.shape(players_who_chosed_it) =", np.shape(players_who_chosed_it))  # DEBUG
        # if len(players_who_chosed_it) > 1:  # DEBUG
        #     print("- rewardIsSharedUniformly: for arm {}, {} users won't have a reward at time t = {} ...".format(arm, len(players_who_chosed_it) - 1, t))  # DEBUG
        if np.size(players_who_chosed_it) > 0:
            collisions[arm] += np.size(players_who_chosed_it)   # Increase nb of collisions for nb of player who collided ?
            i = np.random.choice(players_who_chosed_it)
            rewards[i] = arms[choices[i]].draw(t)
            players[i].getReward(choices[i], rewards[i])
            pulls[i, choices[i]] += 1
            for j in players_who_chosed_it:
                if i != j:
                    # FIXME player[j].getReward() should be called with a reward = 0 when there is collisions (to change the internals memory of the player)
                    players[j].getReward(choices[j], 0)


# Default collision model to use
# defaultCollisionModel = rewardIsSharedUniformly


# List of possible collision models
collision_models = [onlyUniqUserGetsReward, noCollision, rewardIsSharedUniformly, ]
