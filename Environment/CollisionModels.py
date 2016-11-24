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
    nbCollisions = [np.sum(choices == arm) for arm in range(nbArms)]
    for i, player in enumerate(players):
        if nbCollisions[choices[i]] == 0:
            rewards[i] = arms[choices[i]].draw(t)
            player.getReward(choices[i], rewards[i])


# Default collision model to use
defaultCollisionModel = onlyUniqUserGetsReward


def noCollision(t, arms, players, choices, rewards, nbArms):
    """ Simple collision model where all players sample it and receive the reward.
    It corresponds to the single-player simulation: each player is a policy, compared without collision.
    """
    for i, player in enumerate(players):
        rewards[i] = arms[choices[i]].draw(t)
        player.getReward(choices[i], rewards[i])


def rewardIsSharedUniformly(t, arms, players, choices, rewards, nbArms):
    """ Simple collision model where:
    - The players alone on one an arm sample it and receive the reward.
    - In case of more than one player on one arm, only one player (uniform choice) can sample it and receive the reward.
    """
    nbCollisions = [np.sum(choices == arm) for arm in range(nbArms)]
    for i, player in enumerate(players):
        if nbCollisions[choices[i]] == 0:
            rewards[i] = arms[choices[i]].draw(t)
            player.getReward(choices[i], rewards[i])
    for arm in range(nbArms):
        players_who_chosed_it = np.argwhere(choices == arm)
        i = np.random.choice(players_who_chosed_it)
        rewards[i] = arms[choices[i]].draw(t)
        player.getReward(choices[i], rewards[i])


# List of possible collision models
collision_models = [onlyUniqUserGetsReward, noCollision, rewardIsSharedUniformly, ]
