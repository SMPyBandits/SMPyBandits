# -*- coding: utf-8 -*-
""" Selfish: a multi-player policy where every player is selfish: does not try to handle the collisions.
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

import numpy as np

# FIXME finish this


class Selfish():
    """ Selfish: a fully uniform policy who selects randomly (uniformly) an arm among a fix set, at each step (stupid).
    """

    def __init__(self, nbPlayers, onePlayerAlgo, nbArms, *args, **kwargs):
        self.nbPlayers = nbPlayers
        self.players = [None] * nbPlayers
        for i in range(nbPlayers):
            self.players[i] = onePlayerAlgo(nbArms, *args, **kwargs)
        self.params = '{} x {}'.format(nbPlayers, str(self.players[0]))

    def __str__(self):
        return "Selfish({})".format(self.params)

    def startGame(self):
        raise NotImplementedError("Method startGame() in Selfish class is not implemented yet.")
        # for player in self.players:
        #     player.startGame()

    def getReward(self, arm, reward):
        raise NotImplementedError("Method getReward() in Selfish class is not implemented yet.")
        # for player in self.players:
        #     player.getReward(arm, reward)()

    def choice(self):
        raise NotImplementedError("Method choice() in Selfish class is not implemented yet.")
        # choices = np.zeros(self.nbPlayers)
        # for i, player in enumerate(self.players):
        #     choices[i] = player.choice()
