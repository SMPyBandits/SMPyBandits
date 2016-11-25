# -*- coding: utf-8 -*-
""" Selfish: a multi-player policy where every player is selfish: does not try to handle the collisions.
"""

# FIXME player[i].getReward() should be called with a reward = 0 when there is collisions (to change the internals memory of the player)

__author__ = "Lilian Besson"
__version__ = "0.1"


class Selfish():
    """ Selfish: a fully uniform policy who selects randomly (uniformly) an arm among a fix set, at each step (stupid).
    """

    def __init__(self, nbPlayers, onePlayerAlgo):
        self.nbPlayers = nbPlayers
        self.players = dict()
        for i in range(nbPlayers):
            self.players[i] = onePlayerAlgo()

    def __str__(self):
        return "Selfish({})".format(self.params)

    def startGame(self):
        pass

    def getReward(self, arm, reward):
        pass

    def choice(self):
        pass
