# -*- coding: utf-8 -*-
""" CentralizedMultiplePlay: a multi-player policy where ONE policy is used by a centralized agent; asking the policy to select nbPlayers arms at each step.
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

from .BaseMPPolicy import BaseMPPolicy
from .ChildPointer import ChildPointer


class CentralizedMultiplePlay(BaseMPPolicy):
    """ CentralizedMultiplePlay: a multi-player policy where ONE policy is used by a centralized agent; asking the policy to select nbPlayers arms at each step.
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms, *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - *args, **kwargs: arguments, named arguments, given to playerAlgo.

        Examples:
        >>> s = CentralizedMultiplePlay(10, TakeFixedArm, 14)
        >>> s = CentralizedMultiplePlay(NB_PLAYERS, Softmax, nbArms, temperature=TEMPERATURE)

        - To get a list of usable players, use s.childs.
        - Warning: s._players is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for CentralizedMultiplePlay class has to be > 0."
        self.nbPlayers = nbPlayers
        # Only one policy
        self.player = playerAlgo(nbArms, *args, **kwargs)
        # But nbPlayers children
        self.childs = [None] * nbPlayers
        for playerId in range(nbPlayers):
            self.childs[playerId] = ChildPointer(self, playerId)
        self.nbArms = nbArms
        self.params = '{} x {}'.format(nbPlayers, str(self._players[0]))

    def __str__(self):
        return "CentralizedMultiplePlay({})".format(self.params)

    # --- Proxy methods

    def _startGame_one(self, playerId):
        return self._players[playerId].startGame()

    def _getReward_one(self, playerId, arm, reward):
        return self._players[playerId].getReward(arm, reward)

    def _choice_one(self, playerId):
        return self._players[playerId].choice()

    def _handleCollision_one(self, playerId, arm):
        player = self._players[playerId]
        if hasattr(player, 'handleCollision'):
            # player.handleCollision(arm) is called to inform the user that there were a collision
            player.handleCollision(arm)
        else:
            # And if it does not have this method, call players[j].getReward() with a reward = 0 to change the internals memory of the player ?
            player.getReward(arm, getattr(player, 'lower', 0))
            # FIXME Strong assumption on the model
