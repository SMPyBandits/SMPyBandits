# -*- coding: utf-8 -*-
""" ALOHA: generalized implementation of the single-player policy from [Concurrent bandits and cognitive radio network, O.Avner & S.Mannor, 2014](https://arxiv.org/abs/1404.5421), for a generic single-player policy.

This policy uses the collision avoidance mechanism that is inspired by the classical ALOHA protocol, and any single-player policy.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from random import random
import numpy as np
import numpy.random as rn

try:
    from .BaseMPPolicy import BaseMPPolicy
    from .ChildPointer import ChildPointer
    from .with_proba import with_proba
except ImportError:
    from BaseMPPolicy import BaseMPPolicy
    from ChildPointer import ChildPointer
    from with_proba import with_proba



# --- Functions to define [t, t + tnext] intervals

def tnext_beta(t, beta=0.5):
    r""" Simple function, as used in MEGA: ``upper_tnext(t)`` = :math:`t^{\beta}`. Default to :math:`t^{0.5}`.

    >>> tnext_beta(100, beta=0.1)  # doctest: +ELLIPSIS
    1.584...
    >>> tnext_beta(100, beta=0.5)
    10.0
    >>> tnext_beta(100, beta=0.9)  # doctest: +ELLIPSIS
    63.095...
    >>> tnext_beta(1000)  # doctest: +ELLIPSIS
    31.622...
    """
    return t ** beta


def make_tnext_beta(beta=0.5):
    r""" Returns the function :math:`t \mapsto t^{\beta}`.

    >>> tnext = make_tnext_beta(0.5)
    >>> tnext(100)
    10.0
    >>> tnext(1000)  # doctest: +ELLIPSIS
    31.622...
    """
    def tnext(t):
        return t ** beta
    return tnext


def tnext_log(t, scaling=1.):
    r""" Other function, not the one used in MEGA, but our proposal: ``upper_tnext(t)`` = :math:`\text{scaling} * \log(1 + t)`.

    >>> tnext_log(100, scaling=1)  # doctest: +ELLIPSIS
    4.615...
    >>> tnext_log(100, scaling=10)  # doctest: +ELLIPSIS
    46.151...
    >>> tnext_log(100, scaling=100)  # doctest: +ELLIPSIS
    461.512...
    >>> tnext_log(1000)  # doctest: +ELLIPSIS
    6.908...
    """
    return scaling * np.log(1 + t)


def make_tnext_log_scaling(scaling=1.):
    r""" Returns the function :math:`t \mapsto \text{scaling} * \log(1 + t)`.

    >>> tnext = make_tnext_log_scaling(1)
    >>> tnext(100)  # doctest: +ELLIPSIS
    4.615...
    >>> tnext(1000)  # doctest: +ELLIPSIS
    6.908...
    """
    def tnext(t):
        return scaling * np.log(1 + t)
    return tnext


# --- Class oneALOHA, for children

class oneALOHA(ChildPointer):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: the ALOHA collision avoidance protocol is implemented here.
    """

    def __init__(self, nbPlayers, mother, playerId, nbArms,
                 p0=0.5, alpha_p0=0.5, ftnext=tnext_beta, beta=None):
        super(oneALOHA, self).__init__(mother, playerId)
        self.nbPlayers = nbPlayers  #: Number of players
        # Parameters for the ALOHA protocol
        assert 0 <= p0 <= 1, "Error: parameter 'p0' for a ALOHA player should be in [0, 1]."  # DEBUG
        self.p0 = p0  #: Initial probability, should not be modified
        self.p = p0   #: Current probability, can be modified
        assert 0 < alpha_p0 <= 1, "Error: parameter 'alpha_p0' for a ALOHA player should be in (0, 1]."  # DEBUG
        self.alpha_p0 = alpha_p0  #: Parameter alpha for the recurrence equation for probability p(t)
        # Parameters for the ftnext function
        self.beta = beta  #: Parameter beta
        self._ftnext = ftnext  # Function to know how long arms are tagged as unavailable. Can be a callable or None
        # Find the name of the function
        if ftnext is None:
            if beta > 1:
                self._ftnext_name = "t^{%.3g}" % beta
            elif 0 < beta < 1:
                self._ftnext_name = r"\sqrt[%.3g]{t}" % (1. / beta)
            else:
                self._ftnext_name = "t"
        elif ftnext == tnext_log:
            self._ftnext_name = r"\log(t)"
        elif ftnext == tnext_beta:
            self._ftnext_name = r"\sqrt{t}"
        else:
            self._ftnext_name = self._ftnext.__name__.replace("tnext_", "")
        # Internal memory
        self.tnext = np.zeros(nbArms, dtype=int)  #: Only store the delta time
        self.t = -1  #: Internal time
        self.chosenArm = None  #: Last chosen arm

    def __str__(self):
        return r"#{}<ALOHA({}, $p_0={:.3g}$, $\alpha={:.3g}$, $f(t)={}$)>".format(self.playerId + 1, self.mother._players[self.playerId], self.p0, self.alpha_p0, self._ftnext_name)

    def startGame(self):
        """Start game."""
        self.mother._startGame_one(self.playerId)
        self.t = 0
        self.p = self.p0
        self.tnext.fill(0)
        self.chosenArm = None

    def ftnext(self, t):
        """Time until the arm is removed from list of unavailable arms."""
        if self.beta is not None:
            return t ** self.beta
        else:
            return self._ftnext(t)

    def getReward(self, arm, reward):
        """ Receive a reward on arm of index 'arm', as described by the ALOHA protocol.

        - If not collision, receive a reward after pulling the arm.
        """
        # print(" - A oneALOHA player received reward = {} on arm {}, at time t = {}...".format(reward, arm, self.t))  # DEBUG
        self.mother._getReward_one(self.playerId, arm, reward)
        self.p = self.p * self.alpha_p0 + (1 - self.alpha_p0)  # Update proba p

    def handleCollision(self, arm, reward=None):
        """ Handle a collision, on arm of index 'arm'.

        .. warning:: This method has to be implemented in the collision model, it is NOT implemented in the EvaluatorMultiPlayers.

        .. note:: We do not care on which arm the collision occured.
        """
        # print(" ---------> A oneALOHA player saw a collision on arm {}, at time t = {} ... Currently, p = {} ...".format(arm, self.t, self.p))  # DEBUG
        # self.getReward(arm, self.mother.lower)  # FIXED should we give a 0 reward ? Not in this model!
        # 1. With proba 1 - p, give up
        if with_proba(1 - self.p):
            # Random time offset until when this arm self.chosenArm is not sampled
            delta_tnext_k = rn.randint(low=0, high=1 + int(self.ftnext(self.t)))
            self.tnext[self.chosenArm] = self.t + 1 + delta_tnext_k
            # print("   - Reaction to collision on arm {}, at time t = {} : delta_tnext_k = {}, tnext[{}] = {} ...".format(arm, self.t, delta_tnext_k, self.chosenArm, self.tnext[self.chosenArm]))  # DEBUG
            self.p = self.p0  # Reinitialize the proba p
            self.chosenArm = None  # We give up this arm
        # 2. With proba p, persist: nothing to do
        # else:
        #     pass

    def choice(self):
        """ Identify the available arms, and use the underlying single-player policy (UCB, Thompson etc) to choose an arm from this sub-set of arms.
        """
        self.t += 1
        if self.chosenArm is not None:
            # We can still exploit that arm
            pass
        else:
            # We have to chose a new arm
            availableArms = np.nonzero(self.tnext <= self.t)[0]  # Identify available arms
            result = self.mother._choiceFromSubSet_one(self.playerId, availableArms)
            # print("\n - A oneALOHA player {} had to choose an arm among the set of available arms = {}, her choice was : {}, at time t = {} ...".format(self, availableArms, result, self.t))  # DEBUG
            self.chosenArm = result
        return self.chosenArm


# --- Class ALOHA

class ALOHA(BaseMPPolicy):
    """ ALOHA: implementation of the multi-player policy from [Concurrent bandits and cognitive radio network, O.Avner & S.Mannor, 2014](https://arxiv.org/abs/1404.5421), for a generic single-player policy.
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 p0=0.5, alpha_p0=0.5, ftnext=tnext_beta, beta=None,
                 *args, **kwargs):  # Named argument to give them in any order
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.

        - p0: initial probability p(0); p(t) is the probability of persistance on the chosenArm at time t
        - alpha_p0: scaling in the update for p[t+1] <- alpha_p0 p[t] + (1 - alpha_p0)
        - ftnext: general function, default to t -> t^beta, to know from where to sample a random time t_next(k), until when the chosenArm is unavailable. t -> log(1 + t) is also possible.
        - (optional) beta: if present, overwrites ftnext, which will be t --> t^beta.

        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> from Policies import *
        >>> import random; random.seed(0); import numpy as np; np.random.seed(0)
        >>> nbArms = 17
        >>> nbPlayers = 6
        >>> p0, alpha_p0 = 0.6, 0.5
        >>> s = ALOHA(nbPlayers, nbArms, Thompson, p0=p0, alpha_p0=alpha_p0, ftnext=tnext_log)
        >>> [ child.choice() for child in s.children ]
        [6, 11, 8, 4, 8, 8]
        >>> s = ALOHA(nbPlayers, nbArms, UCBalpha, p0=p0, alpha_p0=alpha_p0, beta=0.5, alpha=1)
        >>> [ child.choice() for child in s.children ]
        [1, 0, 5, 2, 15, 3]

        - To get a list of usable players, use ``s.children``.
        - Warning: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRand class has to be > 0."
        self.nbPlayers = nbPlayers  #: Number of players
        self.nbArms = nbArms  #: Number of arms
        # Internal memory
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        for playerId in range(nbPlayers):
            # Initialize internal algorithm (eg. UCB, Thompson etc)
            self._players[playerId] = playerAlgo(nbArms, *args, **kwargs)
            # Initialize proxy child
            self.children[playerId] = oneALOHA(nbPlayers, self, playerId, nbArms, p0=p0, alpha_p0=alpha_p0, ftnext=ftnext, beta=beta)

    def __str__(self):
        return "ALOHA({} x {})".format(self.nbPlayers, str(self._players[0]))


