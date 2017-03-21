# -*- coding: utf-8 -*-
""" Selfish: a multi-player policy where every player is selfish, playing on their side.

- without knowing how many players there is,
- and not even knowing that they should try to avoid collisions. When a collision happens, the algorithm simply receive a 0 reward for the chosen arm.
"""

__author__ = "Lilian Besson"
__version__ = "0.5"

from warnings import warn

from .BaseMPPolicy import BaseMPPolicy
from .ChildPointer import ChildPointer


class SelfishChildPointer(ChildPointer):
    """ Selfish version of the ChildPointer class."""

    def __str__(self):
        return "#{}<{}, {}>".format(self.playerId + 1, self.mother.__class__.__name__, self.mother._players[self.playerId])


# Customize here the value given to a user after a collision
# PENALTY = -1
# PENALTY = 0
# XXX If it is None, then player.lower (default to 0) is used instead
PENALTY = None


class Selfish(BaseMPPolicy):
    """ Selfish: a multi-player policy where every player is selfish, playing on their side.

    - without nowing how many players there is, and
    - not even knowing that they should try to avoid collisions. When a collision happens, the algorithm simply receives a 0 reward for the chosen arm (can be changed with penalty= argument).
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms, penalty=PENALTY, lower=0., amplitude=1., *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Examples:

        >>> s = Selfish(10, TakeFixedArm, 14)
        >>> s = Selfish(NB_PLAYERS, Softmax, nbArms, temperature=TEMPERATURE)

        - To get a list of usable players, use s.children.
        - Warning: s._players is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for Selfish class has to be > 0."
        self.nbPlayers = nbPlayers
        self.penalty = penalty
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers
        self.nbArms = nbArms
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)  # Create it here!
            self.children[playerId] = SelfishChildPointer(self, playerId)
            if hasattr(self._players[playerId], 'handleCollision'):  # XXX they should not have such method!
                warn("Selfish found a player #{} which has a method 'handleCollision' : Selfish should NOT be used with bandit algorithms aware of collision-avoidance!".format(playerId), RuntimeWarning)
                # raise ValueError("Invalid child policy {} for Selfish algorithm! It should not have a collision avoidance protocol!".format(self._players[playerId]))

    def __str__(self):
        return "Selfish({} x {})".format(self.nbPlayers, str(self._players[0]))

    # --- Proxy methods

    def _handleCollision_one(self, playerId, arm):
        """Give a reward of 0, or player.lower, or self.penalty, in case of collision."""
        player = self._players[playerId]
        player.getReward(arm, getattr(player, 'lower', 0) if self.penalty is None else self.penalty)
