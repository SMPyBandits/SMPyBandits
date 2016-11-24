# -*- coding: utf-8 -*-
""" Define some basic collision models.

Collision model are generic functions, taking:

 - the time t
 - the environment
 - the list of players
 - the numpy array of their choices
 - the numpy array to store their rewards
 - the number of arms, nbArms
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np


def onlyUniqUserGetsReward(t, arms, players, choices, rewards, nbArms):
    """ Simple collision model where only the players alone on one an arm sample it and receive the reward.
    """
    nbCollisions = [np.sum(choices == arm) - 1 for arm in range(nbArms)]
    # print("nbCollisions =", nbCollisions)  # DEBUG
    if np.max(nbCollisions) > 1:  # DEBUG
        print("- onlyUniqUserGetsReward: some collisions on channels {} at time t = {} ...".format(nbCollisions >= 1, t))  # DEBUG
    for i, player in enumerate(players):
        if nbCollisions[choices[i]] < 1:
            rewards[i] = arms[choices[i]].draw(t)
            player.getReward(choices[i], rewards[i])
        # else:
        #     print("  - 1 collision on channel {} : {} other users choosed it at time t = {} ...".format(choices[i], nbCollisions[choices[i]], t))  # DEBUG


# Default collision model to use
defaultCollisionModel = onlyUniqUserGetsReward


def noCollision(t, arms, players, choices, rewards, nbArms):
    """ Simple collision model where all players sample it and receive the reward.
    It corresponds to the single-player simulation: each player is a policy, compared without collision.
    """
    for i, player in enumerate(players):
        rewards[i] = arms[choices[i]].draw(t)
        player.getReward(choices[i], rewards[i])


# Default collision model to use
# defaultCollisionModel = noCollision


def rewardIsSharedUniformly(t, arms, players, choices, rewards, nbArms):
    """ Simple collision model where:
    - The players alone on one an arm sample it and receive the reward.
    - In case of more than one player on one arm, only one player (uniform choice) can sample it and receive the reward.
    """
    for arm in range(nbArms):
        players_who_chosed_it = np.argwhere(choices == arm)
        # If he is alone, sure to be chosen, otherwise only one get randomly chosen
        if len(players_who_chosed_it) > 1:  # DEBUG
            print("- rewardIsSharedUniformly: for arm {}, {} users won't have a reward at time t = {} ...".format(arm, len(players_who_chosed_it) - 1, t))  # DEBUG
        i = np.random.choice(players_who_chosed_it)
        rewards[i] = arms[choices[i]].draw(t)
        players[i].getReward(choices[i], rewards[i])


# Default collision model to use
# defaultCollisionModel = rewardIsSharedUniformly


# List of possible collision models
collision_models = [onlyUniqUserGetsReward, noCollision, rewardIsSharedUniformly, ]
