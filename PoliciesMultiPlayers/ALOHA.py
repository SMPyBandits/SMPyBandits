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


def make_tnext_beta(beta=0.5):
    """ Returns the function t --> t ** beta. """
    def tnext(t):
        return t ** beta
    return tnext


def tnext_log(t, scaling=1.):
    """ Other function, not the one used in MEGA, but our proposal: upper_tnext(t) = scaling * log(t). """
    return scaling * np.log(t)


# --- Class oneALOHA, for children

class oneALOHA(ChildPointer):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: the ALOHA collision avoidance protocol is implemented here.
    """

    def __init__(self, nbPlayers, mother, playerId, nbArms,
                 p0=0.5, alpha_p0=0.5, ftnext=tnext_beta, beta=None,
                 lower=0., amplitude=1.):
        super(oneALOHA, self).__init__(mother, playerId)
        self.nbPlayers = nbPlayers
        # Parameters for the ALOHA protocol
        assert 0 <= p0 <= 1, "Error: parameter 'p0' for a ALOHA player should be in [0, 1]."
        self.p0 = p0  # Should not be modified
        self.p = p0   # Can be modified
        assert 0 < alpha_p0 <= 1, "Error: parameter 'alpha_p0' for a ALOHA player should be in (0, 1]."
        self.alpha_p0 = alpha_p0
        if beta is not None:
            self.ftnext = make_tnext_beta(beta)
        else:
            self.ftnext = ftnext
        # Internal memory
        self.tnext = np.ones(nbArms, dtype=int)  # Only store the delta time
        self.t = -1
        self.chosenArm = None

    def __str__(self):
        return "#{}<ALOHA, {}, p0: {}, alpha_p0: {}, ftnext: {}>".format(self.playerId + 1, self.mother._players[self.playerId], self.p0, self.alpha_p0, self.ftnext.__name__)

    def startGame(self):
        super(oneALOHA, self).startGame()  # XXX Call ChildPointer method
        self.p = self.p0
        self.tnext.fill(1)
        self.chosenArm = None

    def getReward(self, arm, reward):
        """ Receive a reward on arm of index 'arm', as described by the ALOHA protocol.

        - If not collision, receive a reward after pulling the arm.
        """
        # print("- A MEGA player receive reward = {} on arm {}, and time t = {}...".format(reward, arm, self.t))  # DEBUG
        super(oneALOHA, self).getReward(arm, reward)  # XXX Call ChildPointer method
        self.p = self.p * self.alpha_p0 + (1 - self.alpha_p0)  # Update proba p

    def handleCollision(self, arm):
        """ Handle a collision, on arm of index 'arm'.

        - Warning: this method has to be implemented in the collision model, it is NOT implemented in the EvaluatorMultiPlayers.
        - Note: we do not care on which arm the collision occured.
        """
        print("- A ALOHA player saw a collision on arm {}, and time t = {} ...".format(arm, self.t))  # DEBUG
        # 1. With proba p, persist
        if rn.random() < self.p:
            self.chosenArm = self.chosenArm  # XXX remove after
        # 2. With proba 1 - p, give up
        else:
            # Random time offset until when this arm self.chosenArm is not sampled
            delta_tnext_k = rn.randint(low=0, high=1 + int(self.ftnext(self.t)))
            self.tnext[self.chosenArm] = self.t + delta_tnext_k
            # Reinitialize the proba p
            self.p = self.p0
            # We give up this arm
            self.chosenArm = None

    def choice(self):
        """ Identify the available arms, and use the underlying single-player policy (UCB, Thompson etc) to choose an arm from this sub-set of arms.
        """
        self.t += 1
        if self.chosenArm is not None:  # We can still exploit that arm
            return self.chosenArm
        else:  # We have to chose a new arm
            # Identify available arms
            print("self.tnext = ", self.tnext)  # DEBUG
            print("self.t = ", self.t)  # DEBUG
            print("np.nonzero(self.tnext <= self.t) = ", np.nonzero(self.tnext <= self.t))  # DEBUG
            availableArms = np.nonzero(self.tnext <= self.t)[0]
            # XXX Call ChildPointer method
            result = super(oneALOHA, self).choiceFromSubSet(availableArms)
            print(" - A oneALOHA player {} had to choose an arm among the set of available arms = {}, her choice was : {} ...".format(self, availableArms, result))  # DEBUG
            self.chosenArm = result
            return result


# --- Class ALOHA

class ALOHA(BaseMPPolicy):
    """ ALOHA: implementation of the multi-player policy from [Concurrent bandits and cognitive radio network, O.Avner & S.Mannor, 2014](https://arxiv.org/abs/1404.5421).
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms, p0=0.5, alpha_p0=0.5, ftnext=tnext_beta, beta=None, lower=0., amplitude=1., *args, **kwargs):  # Named argument to give them in any order
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.

        - p0: initial probability p(0); p(t) is the probability of persistance on the chosenArm at time t
        - alpha_p0: scaling in the update for p[t+1] <- alpha_p0 p[t] + (1 - alpha_p0)
        - ftnext: general function, default to t -> t^beta, to know from where to sample a random time t_next(k), until when the chosenArm is unavailable. FIXME try with a t -> log(t) instead
        - (optional) beta: if present, overwrites ftnext which will be t --> t^beta.

        - *args, **kwargs: arguments, named arguments, given to playerAlgo.

        Example:
        >>> nbArms = 17
        >>> nbPlayers = 6
        >>> p0, alpha_p0 = 0.6, 0.5
        >>> s = ALOHA(nbPlayers, Thompson, nbArms, p0=p0, alpha_p0=alpha_p0, ftnext=tnext_log)
        >>> s = ALOHA(nbPlayers, UCBalpha, nbArms, p0=p0, alpha_p0=alpha_p0, beta=0.5, alpha=1)

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
            # Initialize proxy child
            self.childs[playerId] = oneALOHA(nbPlayers, self, playerId, nbArms, p0=p0, alpha_p0=alpha_p0, ftnext=ftnext, beta=beta, lower=lower, amplitude=amplitude)
        self.nbArms = nbArms
        self.params = '{} x {}'.format(nbPlayers, str(self._players[0]))

    def __str__(self):
        return "ALOHA({})".format(self.params)

    # --- Proxy methods

    def _startGame_one(self, playerId):
        return self._players[playerId].startGame()

    def _getReward_one(self, playerId, arm, reward):
        return self._players[playerId].getReward(arm, reward)

    def _choiceFromSubSet_one(self, playerId, availableArms):
        return self._players[playerId].choiceFromSubSet(availableArms)
