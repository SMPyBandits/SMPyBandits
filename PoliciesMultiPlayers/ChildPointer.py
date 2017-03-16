# -*- coding: utf-8 -*-
""" ChildPointer: Class that acts as a child policy, but in fact it passes all its method calls to the mother class (that can pass it to its internal i-th player, or use any centralized computation).
"""

__author__ = "Lilian Besson"
__version__ = "0.3"


class ChildPointer(object):
    """ Class that acts as a child policy, but in fact it passes all its method calls to the mother class (that can pass it to its internal i-th player, or use any centralized computation).
    """

    def __init__(self, mother, playerId):
        self.mother = mother  # Pointer to the mother class.
        self.playerId = playerId

    def __getattr__(self, name):
        """Generic method to pass an attribute reading call to the actual algorithm (from the mother class)."""
        return getattr(self.mother._players[self.playerId], name)

    def __str__(self):
        return "#{}<{}>".format(self.playerId + 1, self.mother._players[self.playerId])

    def __repr__(self):
        return "{}({})".format(self.mother.__class__.__name__, self.mother._players[self.playerId])

    def startGame(self):
        """ Pass the call to self.mother._startGame_one(playerId) with the player's ID number. """
        return self.mother._startGame_one(self.playerId)

    def getReward(self, arm, reward):
        """ Pass the call to self.mother._getReward_one(playerId, arm, reward) with the player's ID number. """
        return self.mother._getReward_one(self.playerId, arm, reward)

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

    def handleCollision(self, arm):
        """ Pass the call to self.mother._handleCollision_one(playerId, arm) with the player's ID number. """
        return self.mother._handleCollision_one(self.playerId, arm)
