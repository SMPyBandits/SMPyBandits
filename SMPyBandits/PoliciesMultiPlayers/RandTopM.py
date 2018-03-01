# -*- coding: utf-8 -*-
r""" RandTopM: four proposals for an efficient multi-players learning policy. :class:`RandTopM` and :class:`MCTopM` are the two main algorithms, with variants (see below).

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i constantly aims at *one* of the M best arms (denoted :math:`\hat{M}^j(t)`, according to its index policy of indexes :math:`g^j_k(t)` (where M is the number of players),
- When a collision occurs or when the currently chosen arm lies outside of the current estimate of the set M-best, a new current arm is chosen.

.. note:: This is not fully decentralized: as each child player needs to know the (fixed) number of players.

- Reference: [[Multi-Player Bandits Revisited, Lilian Besson and Emilie Kaufmann, 2017]](https://hal.inria.fr/hal-01629733)
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.8"

import numpy as np
import numpy.random as rn

from .BaseMPPolicy import BaseMPPolicy
from .ChildPointer import ChildPointer


#: Whether to use or not the variant with the "chair": after using an arm successfully (no collision), a player won't move after future collisions (she assumes the other will move). But she will still change her chosen arm if it lies outside of the estimated M-best. :class:`RandTopM` (and variants) uses `False` and :class:`MCTopM` (and variants) uses `True`.
WITH_CHAIR = True
WITH_CHAIR = False


#: XXX First experimental idea: when the currently chosen arm lies outside of the estimated Mbest set, force to first try (at least once) the arm with lowest UCB indexes in this Mbest_j(t) set. Used by :class:`RandTopMCautious` and :class:`RandTopMExtraCautious`, and by :class:`MCTopMCautious` and :class:`MCTopMExtraCautious`.
OPTIM_PICK_WORST_FIRST = True
OPTIM_PICK_WORST_FIRST = False


#: XXX Second experimental idea: when the currently chosen arm becomes the worst of the estimated Mbest set, leave it (even before it lies outside of Mbest_j(t)). Used by :class:`RandTopMExtraCautious` and :class:`MCTopMExtraCautious`.
OPTIM_EXIT_IF_WORST_WAS_PICKED = True
OPTIM_EXIT_IF_WORST_WAS_PICKED = False


#: XXX Third experimental idea: when the currently chosen arm becomes the worst of the estimated Mbest set, leave it (even before it lies outside of Mbest_j(t)). **Default now!**. `False` only for :class:`RandTopMOld` and :class:`MCTopMOld`.
OPTIM_PICK_PREV_WORST_FIRST = False
OPTIM_PICK_PREV_WORST_FIRST = True


# --- Class oneRandTopM, for children

class oneRandTopM(ChildPointer):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new random arm is sampled after observing a collision,
    - And the player does not aim at the best arm, but at one of the best arm, based on her index policy.
    - (See variants for more details.)
    """

    def __init__(self, maxRank,
                 withChair, pickWorstFirst, exitIfWorstWasPicked, pickPrevWorstFirst,
                 *args, **kwargs):
        super(oneRandTopM, self).__init__(*args, **kwargs)
        # self.nbArms = self.mother._players[self.playerId].nbArms  #: Number of arms
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different.
        self.chosen_arm = None  #: Current chosen arm.
        self._withChair = withChair  # Whether to use or not the variant with the "chair".
        self._pickWorstFirst = pickWorstFirst  # XXX Whether to use or not the variant with the "cautious choice".
        self._exitIfWorstWasPicked = exitIfWorstWasPicked  # XXX Whether to use or not the second variant with the "even more cautious choice"...
        self._pickPrevWorstFirst = pickPrevWorstFirst  # XXX Whether to use or not the third variant with the "smart and cautious choice"...
        self.sitted = False  #: Not yet sitted. After 1 step without collision, don't react to collision (but still react when the chosen arm lies outside M-best).
        self.prevWorst = []  #: Keep track of the last arms worst than the chosen one (at previous time step).
        self.t = -1  #: Internal time

    def __str__(self):   # Better to recompute it automatically
        player = self.mother._players[self.playerId]
        Mbest_is_incorrect = self.t < self.nbArms or np.any(np.isinf(player.index)) or np.any(np.isnan(player.index))
        str_Mbest = "" if Mbest_is_incorrect else r", $M$-best: ${}$".format(list(self.Mbest()))
        str_chosen_arm = r", arm: ${}$".format(self.chosen_arm) if self.chosen_arm is not None else ""
        # WARNING remember to change here when defining a new variant of RandTopM
        return r"#{}<{}TopM{}{}-{}{}{}>".format(self.playerId + 1, "MC" if self._withChair else "Rand", ("ExtraCautious" if self._exitIfWorstWasPicked else "Cautious") if self._pickWorstFirst else "", "" if self._pickPrevWorstFirst else "Old", player, str_Mbest, str_chosen_arm)

    def startGame(self):
        """Start game."""
        super(oneRandTopM, self).startGame()
        self.t = 0
        self.sitted = False  # Start not sitted, of course!
        self.chosen_arm = None

    def Mbest(self):
        """ Current estimate of the M-best arms. M is the maxRank given to the algorithm."""
        return self.estimatedBestArms(self.maxRank)

    def worst_Mbest(self):
        """ Index of the worst of the current estimate of the M-best arms. M is the maxRank given to the algorithm."""
        order = self.estimatedOrder()
        return order[-self.maxRank]

    def worst_previous__and__current_Mbest(self, current_arm):
        r""" Return the set from which to select a random arm for :class:`MCTopM` (the optimization is now the default):

        .. math:: \hat{M}^j(t) \cap \{ m : g^j_m(t-1) \leq g^j_k(t-1) \}.
        """
        current_Mbest = self.Mbest()
        prev_WorstThenChair = self.prevWorst
        if prev_WorstThenChair is None or len(prev_WorstThenChair) == 0:
            if self.t > 1:  print("WARNING for the MCTopM variant using the 'pickPrevWorstFirst' optimization, it should ever find an empty set 'prev_WorstThenChair' ...")  # DEBUG
            return current_Mbest
        else:
            return np.intersect1d(current_Mbest, prev_WorstThenChair)

    def handleCollision(self, arm, reward=None):
        """ Get a new random arm from the current estimate of Mbest, and give reward to the algorithm if not None."""
        # RandTopM UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: RandTopM UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneRandTopM, self).getReward(arm, reward)
        if not (self._withChair and self.sitted):
            self.chosen_arm = rn.choice(self.Mbest())  # New random arm
            # print(" - A oneRandTopM player {} saw a collision on arm {}, so she had to select a new random arm {} from her estimate of M-best = {} ...".format(self, arm, self.chosen_arm, self.Mbest()))  # DEBUG
        # else:
        #     print(" - A oneRandTopM player {} saw a collision on arm {}, but she ignores it as she plays with a chair and is now sitted ...".format(self, arm))  # DEBUG

    def getReward(self, arm, reward):
        """ Pass the call to self.mother._getReward_one(playerId, arm, reward) with the player's ID number. """
        super(oneRandTopM, self).getReward(arm, reward)
        if self.t >= self.nbArms:
            if self._withChair:
                if not self.sitted:
                    # print(" - A oneRandTopM player {} used this arm {} without any collision, so she is now sitted on this rank, and will not change in case of collision (but will change if the arm lies outside her estimate of M-best)...".format(self, self.chosen_arm))  # DEBUG
                    self.sitted = True
                # else:
                #     print(" - A oneRandTopM player {} is already sitted on this arm {} ...".format(self, self.chosen_arm))  # DEBUG
            # else:
            #     print(" - A oneRandTopM player {} is not playing with a chair, nothing to do ...".format(self)  # DEBUG

    def choice(self):
        """Reconsider the choice of arm, and then use the chosen arm.

        - For all variants, if the chosen arm is no longer in the current estimate of the Mbest set, a new one is selected,
        - The basic RandTopM selects uniformly an arm in estimate Mbest,
        - MCTopM starts by being *"non sitted"* on its new chosen arm,
        - MCTopMCautious is forced to first try the arm with *lowest* UCB indexes (or whatever index policy is used).
        """
        if self.t < self.nbArms:  # Force to sample each arm at least one
            self.chosen_arm = super(oneRandTopM, self).choice()
        else:  # But now, trust the estimated set Mbest
            current_Mbest = self.Mbest()
            worst_current = self.worst_Mbest()
            # XXX optimization for MCTopMExtraCautious
            if (self.chosen_arm not in current_Mbest) or (self._exitIfWorstWasPicked and self.chosen_arm == worst_current):
                if self._withChair:
                    self.sitted = False
                # old_arm = self.chosen_arm
                # XXX optimization for MCTopMCautious and MCTopMExtraCautious
                if self._pickWorstFirst:
                    # aim at the worst of the current estimated arms
                    self.chosen_arm = worst_current
                elif self._pickPrevWorstFirst:
                    self.chosen_arm = rn.choice(self.worst_previous__and__current_Mbest(self.chosen_arm))
                else:
                    self.chosen_arm = rn.choice(current_Mbest)  # New random arm
                # print("\n - A oneRandTopM player {} had chosen arm = {}, but it lied outside of M-best = {}, so she selected a new one = {} {}...".format(self, old_arm, current_Mbest, self.chosen_arm, "and is no longer sitted" if self._withChair else "but is not playing with a chair"))  # DEBUG
        # Done
        self.t += 1
        # XXX optimization for MCTopM
        if self._pickPrevWorstFirst:
            index = self._index()
            self.prevWorst = np.where(index <= index[self.chosen_arm])[0]
        return self.chosen_arm

    def _index(self):
        """ Update and return the indexes of the underlying index policy."""
        self.mother._players[self.playerId].computeAllIndex()
        return self.mother._players[self.playerId].index


# --- Class RandTopM

class RandTopM(BaseMPPolicy):
    """ RandTopM: a proposal for an efficient multi-players learning policy.
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 withChair=WITH_CHAIR,
                 pickWorstFirst=OPTIM_PICK_WORST_FIRST,
                 exitIfWorstWasPicked=OPTIM_EXIT_IF_WORST_WAS_PICKED,
                 pickPrevWorstFirst=OPTIM_PICK_PREV_WORST_FIRST,
                 maxRank=None, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - withChair: see ``WITH_CHAIR``,
        - pickWorstFirst: see ``OPTIM_PICK_WORST_FIRST``,
        - exitIfWorstWasPicked: see ``EXIT_IF_WORST_WAS_PICKED``,
        - pickPrevWorstFirst: see ``OPTIM_PICK_PREV_WORST_FIRST``,
        - maxRank: maximum rank allowed by the RandTopM child (default to nbPlayers, but for instance if there is 2 × RandTopM[UCB] + 2 × RandTopM[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = RandTopM(nbPlayers, Thompson, nbArms)

        - To get a list of usable players, use ``s.children``.

        .. warning:: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for RandTopM class has to be > 0."  # DEBUG
        if maxRank is None:
            maxRank = nbPlayers
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.nbPlayers = nbPlayers  #: Number of players
        self.withChair = withChair  #: Using a chair ?
        self.pickWorstFirst = pickWorstFirst  #: Using first optimization ?
        self.exitIfWorstWasPicked = exitIfWorstWasPicked  #: Using second optimization ?
        self.pickPrevWorstFirst = pickPrevWorstFirst  #: Using third optimization ? Default to yes now.
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRandTopM(maxRank, withChair, pickWorstFirst, exitIfWorstWasPicked, pickPrevWorstFirst, self, playerId)

    def __str__(self):
        return "RandTopM({} x {})".format(self.nbPlayers, str(self._players[0]))


class RandTopMCautious(RandTopM):
    """ RandTopMCautious: another proposal for an efficient multi-players learning policy, more "stationary" than RandTopM.

    .. warning:: Still very experimental! But it seems to be the most efficient decentralized MP algorithm we have so far...
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 maxRank=None, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the RandTopMCautious child (default to nbPlayers, but for instance if there is 2 × RandTopMCautious[UCB] + 2 × RandTopMCautious[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.
        """
        super(RandTopMCautious, self).__init__(nbPlayers, nbArms, playerAlgo, withChair=False, pickWorstFirst=True, exitIfWorstWasPicked=False, pickPrevWorstFirst=False, maxRank=maxRank, lower=lower, amplitude=amplitude, *args, **kwargs)

    def __str__(self):
        return "RandTopMCautious({} x {})".format(self.nbPlayers, str(self._players[0]))


class RandTopMExtraCautious(RandTopM):
    """ RandTopMExtraCautious: another proposal for an efficient multi-players learning policy, more "stationary" than RandTopM.

    .. warning:: Still very experimental! But it seems to be the most efficient decentralized MP algorithm we have so far...
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 maxRank=None, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the RandTopMExtraCautious child (default to nbPlayers, but for instance if there is 2 × RandTopMExtraCautious[UCB] + 2 × RandTopMExtraCautious[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.
        """
        super(RandTopMExtraCautious, self).__init__(nbPlayers, nbArms, playerAlgo, withChair=False, pickWorstFirst=True, exitIfWorstWasPicked=True, pickPrevWorstFirst=False, maxRank=maxRank, lower=lower, amplitude=amplitude, *args, **kwargs)

    def __str__(self):
        return "RandTopMExtraCautious({} x {})".format(self.nbPlayers, str(self._players[0]))


class RandTopMOld(RandTopM):
    """ RandTopMOld: another proposal for an efficient multi-players learning policy, more "stationary" than RandTopM.
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 maxRank=None, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the RandTopMOld child (default to nbPlayers, but for instance if there is 2 × RandTopMOld[UCB] + 2 × RandTopMOld[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.
        """
        super(RandTopMOld, self).__init__(nbPlayers, nbArms, playerAlgo, withChair=False, pickWorstFirst=False, exitIfWorstWasPicked=False, pickPrevWorstFirst=False, maxRank=maxRank, lower=lower, amplitude=amplitude, *args, **kwargs)

    def __str__(self):
        return "RandTopMOld({} x {})".format(self.nbPlayers, str(self._players[0]))


class MCTopM(RandTopM):
    """ MCTopM: another proposal for an efficient multi-players learning policy, more "stationary" than RandTopM.

    .. warning:: Still very experimental! But it seems to be the most efficient decentralized MP algorithm we have so far...
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 maxRank=None, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the MCTopM child (default to nbPlayers, but for instance if there is 2 × MCTopM[UCB] + 2 × MCTopM[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.
        """
        super(MCTopM, self).__init__(nbPlayers, nbArms, playerAlgo, withChair=True, pickWorstFirst=False, exitIfWorstWasPicked=False, pickPrevWorstFirst=True, maxRank=maxRank, lower=lower, amplitude=amplitude, *args, **kwargs)

    def __str__(self):
        return "MCTopM({} x {})".format(self.nbPlayers, str(self._players[0]))


class MCTopMCautious(RandTopM):
    """ MCTopMCautious: another proposal for an efficient multi-players learning policy, more "stationary" than RandTopM.

    .. warning:: Still very experimental! But it seems to be the most efficient decentralized MP algorithm we have so far...
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 maxRank=None, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the MCTopMCautious child (default to nbPlayers, but for instance if there is 2 × MCTopMCautious[UCB] + 2 × MCTopMCautious[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.
        """
        super(MCTopMCautious, self).__init__(nbPlayers, nbArms, playerAlgo, withChair=True, pickWorstFirst=True, exitIfWorstWasPicked=False, pickPrevWorstFirst=False, maxRank=maxRank, lower=lower, amplitude=amplitude, *args, **kwargs)

    def __str__(self):
        return "MCTopMCautious({} x {})".format(self.nbPlayers, str(self._players[0]))


class MCTopMExtraCautious(RandTopM):
    """ MCTopMExtraCautious: another proposal for an efficient multi-players learning policy, more "stationary" than RandTopM.

    .. warning:: Still very experimental! But it seems to be the most efficient decentralized MP algorithm we have so far...
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 maxRank=None, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the MCTopMExtraCautious child (default to nbPlayers, but for instance if there is 2 × MCTopMExtraCautious[UCB] + 2 × MCTopMExtraCautious[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.
        """
        super(MCTopMExtraCautious, self).__init__(nbPlayers, nbArms, playerAlgo, withChair=True, pickWorstFirst=True, exitIfWorstWasPicked=True, pickPrevWorstFirst=False, maxRank=maxRank, lower=lower, amplitude=amplitude, *args, **kwargs)

    def __str__(self):
        return "MCTopMExtraCautious({} x {})".format(self.nbPlayers, str(self._players[0]))


class MCTopMOld(RandTopM):
    """ MCTopMOld: another proposal for an efficient multi-players learning policy, more "stationary" than RandTopM.

    .. warning:: Still very experimental! But it seems to be one of the most efficient decentralized MP algorithm we have so far... The two other variants of MCTopM seem even better!
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 maxRank=None, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - maxRank: maximum rank allowed by the MCTopMOld child (default to nbPlayers, but for instance if there is 2 × MCTopMOld[UCB] + 2 × MCTopMOld[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.
        """
        super(MCTopMOld, self).__init__(nbPlayers, nbArms, playerAlgo, withChair=True, pickWorstFirst=False, exitIfWorstWasPicked=False, pickPrevWorstFirst=False, maxRank=maxRank, lower=lower, amplitude=amplitude, *args, **kwargs)

    def __str__(self):
        return "MCTopMOld({} x {})".format(self.nbPlayers, str(self._players[0]))

