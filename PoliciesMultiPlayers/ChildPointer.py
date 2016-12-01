# -*- coding: utf-8 -*-
""" ChildPointer: Class that acts as a child policy, but in fact it pass all its method calls to the mother class (that can pass it to its i-th player, or use any centralized computation).
"""

__author__ = "Lilian Besson"
__version__ = "0.1"


class ChildPointer(object):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class (that can pass it to its i-th player, or use any centralized computation).
    """

    def __init__(self, mother, playerId):
        self.mother = mother  # Pointer to the mother class.
        self.playerId = playerId

    def __str__(self):   # Better to recompute it automatically
        return '#{}<{}>'.format(self.playerId + 1, self.mother._players[self.playerId])

    def startGame(self):
        """ Pass the call to self.mother._startGame_one(playerId) with the player's ID number. """
        return self.mother._startGame_one(self.playerId)

    def getReward(self, arm, reward):
        """ Pass the call to self.mother._getReward_one(playerId, arm, reward) with the player's ID number. """
        return self.mother._getReward_one(self.playerId, arm, reward)

    def choice(self):
        """ Pass the call to self.mother._choice_one(playerId) with the player's ID number. """
        return self.mother._choice_one(self.playerId)

    def handleCollision(self, arm):
        """ Pass the call to self.mother._handleCollision_one(playerId, arm) with the player's ID number. """
        return self.mother._handleCollision_one(self.playerId, arm)
