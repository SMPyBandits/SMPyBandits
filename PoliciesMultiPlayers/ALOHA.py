# -*- coding: utf-8 -*-
""" ALOHA: generalized implementation of the single-player policy from [Concurrent bandits and cognitive radio network, O.Avner & S.Mannor, 2014](https://arxiv.org/abs/1404.5421).

This policy uses the collision avoidance mechanism that is inspired by the classical ALOHA protocol, and any single player policy.
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np
import numpy.random as rn

from .BaseMPPolicy import BaseMPPolicy
from .ChildPointer import ChildPointer


# --- Functions to define tnext intervals

def tnext_beta(t, beta=0.5):
    """ Simple function, as used in MEGA: upper_tnext(t) = t ** beta. Default to t ** 0.5. """
    return t ** beta


def tnext_log(t, scaling=1.):
    """ Other function, not the one used in MEGA, but our proposal: upper_tnext(t) = scaling * log(t). """
    return scaling * np.log(t)


# --- Class oneALOHA, for children

class oneALOHA(ChildPointer):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: the ALOHA collision avoidance protocol is implemented here.
    """

    def __init__(self, nbPlayers, mother, playerId, p0=0.5, alpha=0.5, ftnext=tnext_beta):
        super(oneALOHA, self).__init__(mother, playerId)
        self.nbPlayers = nbPlayers
        # Parameters for the ALOHA protocol
        self.p0 = p0
        self.alpha = alpha
        self.ftnext = ftnext

    def __str__(self):   # Better to recompute it automatically
        return '#{}<ALOHA, {}, p0: {}, alpha: {}, beta: {}, FIXME>'.format(self.playerId + 1, self.mother._players[self.playerId], self.p0, self.alpha, self.ftnext.__name__, FIXME)

    def startGame(self):
        super(oneALOHA, self).startGame()
        # FIXME ?

    def handleCollision(self, arm):
        # super(oneALOHA, self).handleCollision(arm)  # No need for that, there is a collision avoidance RIGHT HERE
        FIXME ?
        # print(" - A oneALOHA player {} saw a collision, so FIXME ...".format(self, FIXME))  # DEBUG

    def choice(self):
        result = super(oneALOHA, self).choice()
        # print(" - A oneALOHA player {} had to choose an arm among the best from rank {}, her choice was : {} ...".format(self, self.rank, result))  # DEBUG
        return result


# --- Class ALOHA

class ALOHA(BaseMPPolicy):
    """ ALOHA: implementation of the multi-player policy from [Concurrent bandits and cognitive radio network, O.Avner & S.Mannor, 2014](https://arxiv.org/abs/1404.5421).
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms, p0=0.5, alpha=0.5, ftnext=tnext_beta, lower=0., amplitude=1., *args, **kwargs):  # Named argument to give them in any order
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.

        - p0: initial probability p(0); p(t) is the probability of persistance on the chosenArm at time t
        - alpha: scaling in the update for p(t+1) <- alpha p(t) + (1 - alpha(t))
        - ftnext: general function, default to t -> t^beta, to know from where to sample a random time t_next(k), until when the chosenArm is unavailable. FIXME try with a t -> log(t) instead

        - *args, **kwargs: arguments, named arguments, given to playerAlgo.

        Example:
        >>> nbArms, p0, alpha, tnext = 17, 0.5, 0.5, tnext_beta
        >>> s = ALOHA(NB_PLAYERS, Thompson, nbArms, p0, alpha, beta, c, d)

        - To get a list of usable players, use s.childs.
        - Warning: s._players is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRand class has to be > 0."
        self.nbPlayers = nbPlayers
        # Interal
        self._players = [None] * nbPlayers
        self.childs = [None] * nbPlayers
        for playerId in range(nbPlayers):
            # Initialize internal algorithm (eg. UCB, Thompson etc)
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.childs[playerId] = oneALOHA(nbPlayers, self, playerId, p0=p0, alpha=alpha, ftnext=ftnext)
        self.nbArms = nbArms
        self.params = '{} x {}'.format(nbPlayers, str(self._players[0]))

    def __str__(self):
        return "ALOHA({})".format(self.params)

    # --- Proxy methods

    def _startGame_one(self, playerId):
        return self._players[playerId].startGame()

    def _getReward_one(self, playerId, arm, reward):
        return self._players[playerId].getReward(arm, reward)

    def _choiceWithRank_one(self, playerId, rank):
        return self._players[playerId].choiceWithRank(rank)

    def _handleCollision_one(self, playerId, arm):
        player = self._players[playerId]
        if hasattr(player, 'handleCollision'):
            # player.handleCollision(arm) is called to inform the user that there were a collision
            player.handleCollision(arm)
        else:
            # And if it does not have this method, call players[j].getReward() with a reward = 0 to change the internals memory of the player ?
            player.getReward(arm, 0)
            # print("   - rhoRand _handleCollision_one({}, {}) called getReward({}, 0) for player = {} ...".format(playerId, arm, arm, player))  # DEBUG
            # FIXME Strong assumption on the model
