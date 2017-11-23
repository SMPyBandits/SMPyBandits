# -*- coding: utf-8 -*-
""" CentralizedMultiplePlay: a multi-player policy where ONE policy is used by a centralized agent; asking the policy to select nbPlayers arms at each step.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np

from .BaseMPPolicy import BaseMPPolicy
from .ChildPointer import ChildPointer


# --- Class for a child player

class CentralizedChildPointer(ChildPointer):
    """ Centralized version of the ChildPointer class."""

    def __str__(self):
        return "#{}<{}({})>".format(self.playerId + 1, self.mother.__class__.__name__, self.mother.player)

    def __repr__(self):
        return "{}({})".format(self.mother.__class__.__name__, self.mother.player)


# --- Class for the mother

class CentralizedMultiplePlay(BaseMPPolicy):
    """ CentralizedMultiplePlay: a multi-player policy where ONE policy is used by a centralized agent; asking the policy to select nbPlayers arms at each step.
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo, uniformAllocation=False, *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - uniformAllocation: Should the affectations of users always be uniform, or fixed when UCB indexes have converged? First choice is more fair, but linear nb of switches, second choice is not fair, but cst nb of switches.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Examples:

        >>> s = CentralizedMultiplePlay(10, TakeFixedArm, 14)
        >>> s = CentralizedMultiplePlay(NB_PLAYERS, Softmax, nbArms, temperature=TEMPERATURE)

        - To get a list of usable players, use ``s.children``.
        - Warning: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for CentralizedMultiplePlay class has to be > 0."
        self.nbPlayers = nbPlayers  #: Number of players
        self.player = playerAlgo(nbArms, *args, **kwargs)  #: Only one policy
        self.children = [None] * nbPlayers  #: But nbPlayers children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        for playerId in range(nbPlayers):
            self.children[playerId] = CentralizedChildPointer(self, playerId)
            print(" - One new child, of index {}, and class {} ...".format(playerId, self.children[playerId]))  # DEBUG
        #: Option: in case of multiplay plays, should the affectations of users always be uniform, or fixed when UCB indexes have converged? First choice is more fair, but linear nb of switches, second choice is not fair, but cst nb of switches
        self.uniformAllocation = uniformAllocation
        # Internal memory
        self.choices = (-10000) * np.ones(nbArms, dtype=int)  #: Choices, given by first call to internal algorithm
        self.affectation_order = np.random.permutation(nbPlayers)  #: Affectation of choices to players

    def __str__(self):
        return "{}({} x {}{})".format(self.__class__.__name__, self.nbPlayers, str(self.player), ", shuffle" if self.uniformAllocation else "")

    # --- Proxy methods

    def _startGame_one(self, playerId):
        """Pass the call to the player algorithm."""
        if playerId == 0:  # For the first player, run the method
            self.player.startGame()
        # For the other players, nothing to do? Yes
        self.affectation_order = np.random.permutation(self.nbPlayers)

    def _getReward_one(self, playerId, arm, reward):
        """Pass the call to the player algorithm."""
        self.player.getReward(arm, reward)
        # if playerId != 0:  # FIXME? We have to be sure that the internal player.t is not messed up
        #     if hasattr(self.player, 't'):
        #         self.player.t -= 1

    def _choice_one(self, playerId):
        """Use the player algorithm for the 1st decision, for each players, then use it."""
        if playerId == 0:  # For the first player, run the method
            # FIXED sort it then apply affectation_order, to fix its order ==> will have a fixed nb of switches for CentralizedMultiplePlay
            if self.uniformAllocation:
                self.choices = self.player.choiceMultiple(self.nbPlayers)
            else:
                self.choices = np.sort(self.player.choiceMultiple(self.nbPlayers))[self.affectation_order]  # XXX Increasing order...
                # self.choices = np.sort(self.player.choiceMultiple(self.nbPlayers))[self.affectation_order][::-1]  # XXX Decreasing order...
            # print("At time t = {} the {} centralized policy chosed arms = {} ...".format(self.player.t, self, self.choices))  # DEBUG
        # For the all players, use the pre-computed result
        return self.choices[playerId]

    def _handleCollision_one(self, playerId, arm, reward=None):
        """Cannot be called!"""
        raise ValueError("Error: a {} policy should always aim at orthogonal arms, so no collision should be observed, but player {} saw a collision on arm {} ...".format(self.__class__.__name__, playerId, arm))

    def _estimatedOrder_one(self, playerId):
        """Use the centralized algorithm to estimate ranking of the arms."""
        return self.player.estimatedOrder()
