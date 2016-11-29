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
        self.params = '#{}<{}>'.format(playerId + 1, self.mother._players[self.playerId])

    def __str__(self):
        # return "Child({})".format(self.params)
        return self.params

    def startGame(self):
        """ Pass the call to self.mother._startGame_one() with the player'ID number. """
        return self.mother._startGame_one(self.playerId)

    def getReward(self, arm, reward):
        """ Pass the call to self.mother._getReward_one() with the player'ID number. """
        return self.mother._getReward_one(self.playerId, arm, reward)

    def choice(self):
        """ Pass the call to self.mother._choice_one() with the player'ID number. """
        return self.mother._choice_one(self.playerId)
