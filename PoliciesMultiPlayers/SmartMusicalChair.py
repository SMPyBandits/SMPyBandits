# -*- coding: utf-8 -*-
""" SmartMusicalChair: our proposal for an efficient multi-players learning policy.

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i constantly aims at *one* of the M best arms, according to its index policy (where M is the number of players),
- When a collision occurs or when the currently chosen arm lies outside of the current estimate of the set M-best, a new current arm is chosen.

.. note:: This is not fully decentralized: as each child player needs to know the (fixed) number of players.

.. note:: Based on a variant of the multi-player policy rhoRand from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
import numpy.random as rn

from .BaseMPPolicy import BaseMPPolicy
from .ChildPointer import ChildPointer


#: Whether to use or not the variant with the "chair": after using an arm successfully (no collision), a player won't move after future collisions (she assumes the other will move). But she will still change her chosen arm if it lies outside of the estimated M-best.
#: **Warning** experimental!
WITHCHAIR = False
WITHCHAIR = True


# --- Class oneSmartMusicalChair, for children

class oneSmartMusicalChair(ChildPointer):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new random rank is sampled after observing a collision,
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy.
    """

    def __init__(self, maxRank, withChair, *args, **kwargs):
        super(oneSmartMusicalChair, self).__init__(*args, **kwargs)
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different.
        self.chosen_arm = None  #: Current chosen arm.
        self._withChair = withChair  # Whether to use or not the variant with the "chair".
        self.sitted = False  #: Not yet sitted. After 1 step without collision, don't react to collision (but still react when the chosen arm lies outside M-best).
        self.t = -1  #: Internal time

    def __str__(self):   # Better to recompute it automatically
        player = self.mother._players[self.playerId]
        Mbest_is_incorrect = self.t < self.nbArms or np.any(np.isinf(player.index)) or np.any(np.isnan(player.index))
        str_Mbest = "" if Mbest_is_incorrect else r", $M$-best: ${}$".format(list(self.Mbest))
        # # FIXME it messes up with the display of the titles...
        # str_Mbest = ""
        str_chosen_arm = r", arm: ${}$".format(self.chosen_arm) if self.chosen_arm is not None else ""
        return r"#{}<SmartMusicalChair[{}{}{}{}]>".format(self.playerId + 1, player, str_Mbest, str_chosen_arm, ", staying sitted" if self._withChair else "")

    def startGame(self):
        """Start game."""
        super(oneSmartMusicalChair, self).startGame()
        self.t = 0
        self.sitted = False  # Start not sitted, of course!
        self.chosen_arm = None

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def Mbest(self):
        """ Current estimate of the M-best arms. M is the maxRank given to the algorithm."""
        return self.estimatedBestArms(self.maxRank)

    def handleCollision(self, arm, reward=None):
        """ Get a new random arm from the current estimate of Mbest, and give reward to the algorithm if not None."""
        # SmartMusicalChair UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: SmartMusicalChair UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneSmartMusicalChair, self).getReward(arm, reward)
        if not (self._withChair and self.sitted):
            self.chosen_arm = rn.choice(self.Mbest)  # New random arm
            # print(" - A oneSmartMusicalChair player {} saw a collision on arm {}, so she had to select a new random arm {} from her estimate of M-best = {} ...".format(self, arm, self.chosen_arm, self.Mbest))  # DEBUG
        # else:
        #     print(" - A oneSmartMusicalChair player {} saw a collision on arm {}, but she ignores it as she plays with a chair and is now sitted ...".format(self, arm))  # DEBUG

    def getReward(self, arm, reward):
        """ Pass the call to self.mother._getReward_one(playerId, arm, reward) with the player's ID number. """
        super(oneSmartMusicalChair, self).getReward(arm, reward)
        if self.t >= self.nbArms:
            if self._withChair:
                if not self.sitted:
                    # print(" - A oneSmartMusicalChair player {} used this arm {} without any collision, so she is now sitted on this rank, and will not change in case of collision (but will change if the arm lies outside her estimate of M-best)...".format(self, self.chosen_arm))  # DEBUG
                    self.sitted = True
                # else:
                #     print(" - A oneSmartMusicalChair player {} is already sitted on this arm {} ...".format(self, self.chosen_arm))  # DEBUG
            # else:
            #     print(" - A oneSmartMusicalChair player {} is not playing with a chair, nothing to do ...".format(self)  # DEBUG

    def choice(self):
        """Use the chosen arm."""
        if self.t < self.nbArms:  # Force to sample each arm at least one
            self.chosen_arm = super(oneSmartMusicalChair, self).choice()
        else:  # But now, trust the estimated set Mbest
            current_Mbest = self.Mbest
            if self.chosen_arm not in current_Mbest:
                if self._withChair:
                    self.sitted = False
                old_arm = self.chosen_arm
                self.chosen_arm = rn.choice(current_Mbest)  # New random arm
                # print("\n - A oneSmartMusicalChair player {} had chosen arm = {}, but it lied outside of M-best = {}, so she selected a new one = {} {}...".format(self, old_arm, current_Mbest, self.chosen_arm, "and is no longer sitted" if self._withChair else "but is not playing with a chair"))  # DEBUG
        # Done
        self.t += 1
        # FIXME remove: this cost too much time!
        # XXX It's also making SmartMusicalChair[Thompson] fail : its set Mbest is RANDOM
        # assert self.chosen_arm in self.Mbest, "Error: at time t = {}, a oneSmartMusicalChair player {} chose an arm = {} which was NOT on its set Mbest(t) = {} ...".format(self.t, self, self.chosen_arm, self.Mbest)  # DEBUG
        return self.chosen_arm


# --- Class SmartMusicalChair

class SmartMusicalChair(BaseMPPolicy):
    """ SmartMusicalChair: our proposal for an efficient multi-players learning policy.
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms,
                 withChair=WITHCHAIR, maxRank=None, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the SmartMusicalChair child (default to nbPlayers, but for instance if there is 2 × SmartMusicalChair[UCB] + 2 × SmartMusicalChair[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = SmartMusicalChair(nbPlayers, Thompson, nbArms)

        - To get a list of usable players, use ``s.children``.

        .. warning:: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for SmartMusicalChair class has to be > 0."  # DEBUG
        if maxRank is None:
            maxRank = nbPlayers
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.nbPlayers = nbPlayers  #: Number of players
        self.withChair = withChair  #: Using a chair ?
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneSmartMusicalChair(maxRank, withChair, self, playerId)

    def __str__(self):
        return "SmartMusicalChair({} x {})".format(self.nbPlayers, str(self._players[0]))
