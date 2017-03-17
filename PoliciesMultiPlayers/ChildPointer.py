# -*- coding: utf-8 -*-
""" ChildPointer: Class that acts as a child policy, but in fact it passes all its method calls to the mother class (that can pass it to its internal i-th player, or use any centralized computation).
"""

__author__ = "Lilian Besson"
__version__ = "0.3"

from warnings import warn


class ChildPointer(object):
    """ Class that acts as a child policy, but in fact it passes *all* its method calls to the mother class (that can pass it to its internal i-th player, or use any centralized computation).
    """

    def __init__(self, mother, playerId):
        self.mother = mother  # Pointer to the mother class.
        self.playerId = playerId

    @property
    def nbArms(self):
        """Trying to read the number of arms from mother class."""
        nbArms = "UNKNOWN"
        if hasattr(self, 'mother'):
            if hasattr(self.mother, 'nbArms'):
                nbArms = self.mother.nbArms
            elif hasattr(self.mother._players[self.playerId], 'nbArms'):
                nbArms = self.mother._players[self.playerId].nbArms
        if nbArms == "UNKNOWN":
            warn("ChildPointer: {} seems unable to get the number of arms from his mother class {} ...".format(self, self.Mother), RuntimeWarning)
        return nbArms

    def __str__(self):
        return "#{}<{}>".format(self.playerId + 1, self.mother._players[self.playerId])

    def __repr__(self):
        return "{}({})".format(self.mother.__class__.__name__, self.mother._players[self.playerId])

    # Proxy methods!

    def startGame(self):
        """ Pass the call to self.mother._startGame_one(playerId) with the player's ID number. """
        return self.mother._startGame_one(self.playerId)

    def getReward(self, arm, reward):
        """ Pass the call to self.mother._getReward_one(playerId, arm, reward) with the player's ID number. """
        return self.mother._getReward_one(self.playerId, arm, reward)

    def handleCollision(self, arm):
        """ Pass the call to self.mother._handleCollision_one(playerId, arm) with the player's ID number. """
        return self.mother._handleCollision_one(self.playerId, arm)

    def choice(self):
        """ Pass the call to self.mother._choice_one(playerId) with the player's ID number. """
        return self.mother._choice_one(self.playerId)

    def choiceWithRank(self, rank=1):
        """ Pass the call to self.mother._choiceWithRank_one(playerId) with the player's ID number. """
        return self.mother._choiceWithRank_one(self.playerId, rank)

    def choiceFromSubSet(self, availableArms='all'):
        """ Pass the call to self.mother._choiceFromSubSet_one(playerId) with the player's ID number. """
        return self.mother._choiceFromSubSet_one(self.playerId, availableArms)

    def choiceMultiple(self, nb=1):
        """ Pass the call to self.mother._choiceMultiple_one(playerId) with the player's ID number. """
        return self.mother._choiceMultiple_one(self.playerId, nb)

    def choiceIMP(self, nb=1):
        """ Pass the call to self.mother._choiceIMP_one(playerId) with the player's ID number. """
        return self.mother._choiceIMP_one(self.playerId, nb)

    def estimatedOrder(self):
        """ Pass the call to self.mother._estimatedOrder_one(playerId) with the player's ID number. """
        return self.mother._estimatedOrder_one(self.playerId)
