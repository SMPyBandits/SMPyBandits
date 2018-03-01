# -*- coding: utf-8 -*-
r""" RandTopMEstEst: four proposals for an efficient multi-players learning policy. :class:`RandTopMEstEst` and :class:`MCTopMEstEst` are the two main algorithms, with variants (see below).

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i constantly aims at *one* of the M best arms (denoted :math:`\hat{M}^j(t)`, according to its index policy of indexes :math:`g^j_k(t)` (where M is the number of players),
- When a collision occurs or when the currently chosen arm lies outside of the current estimate of the set M-best, a new current arm is chosen.
- The (fixed) number of players is learned on the run.

.. note:: This is **fully decentralized**: player do not need to know the (fixed) number of players!

- Reference: [[Multi-Player Bandits Revisited, Lilian Besson and Emilie Kaufmann, 2017]](https://hal.inria.fr/hal-01629733)

.. warning:: This is still very experimental!

.. note:: For a more generic approach, see the wrapper defined in :class:`EstimateM.EstimateM`.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.8"

import numpy as np

from .BaseMPPolicy import BaseMPPolicy
from .ChildPointer import ChildPointer
from .RandTopM import oneRandTopM
from .EstimateM import threshold_on_t_with_horizon, threshold_on_t_doubling_trick, threshold_on_t


class oneRandTopMEst(oneRandTopM):
    r""" Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - The procedure to estimate :math:`\hat{M}_i(t)` is not so simple, but basically everyone starts with :math:`\hat{M}_i(0) = 1`, and when colliding :math:`\hat{M}_i(t+1) = \hat{M}_i(t) + 1`, for some time (with a complicated threshold).
    """

    def __init__(self, threshold, *args, **kwargs):
        if 'horizon' in kwargs:
            self.horizon = kwargs['horizon']
            del kwargs['horizon']
        else:
            self.horizon = None
        super(oneRandTopMEst, self).__init__(*args, **kwargs)
        # Parameters
        self.maxRank = 1
        self.threshold = threshold  #: Threshold function
        # Internal variables
        self.nbPlayersEstimate = 1  #: Number of players. Optimistic: start by assuming it is alone!
        self.collisionCount = np.zeros(self.nbArms, dtype=int)  #: Count collisions on each arm, since last increase of nbPlayersEstimate
        self.timeSinceLastCollision = 0  #: Time since last collision. Don't remember why I thought using this could be useful... But it's not!
        self.t = 0  #: Internal time

    def __str__(self):   # Better to recompute it automatically
        player = self.mother._players[self.playerId]
        Mbest_is_incorrect = self.t < self.nbArms or np.any(np.isinf(player.index)) or np.any(np.isnan(player.index))
        str_Mbest = "" if Mbest_is_incorrect else r", $M$-best: ${}$".format(list(self.Mbest()))
        str_chosen_arm = r", arm: ${}$".format(self.chosen_arm) if self.chosen_arm is not None else ""
        # WARNING remember to change here when defining a new variant of RandTopM
        return r"#{}<{}TopM{}{}-Est{}-{}{}{}{}>".format(self.playerId + 1, "MC" if self._withChair else "Rand", ("ExtraCautious" if self._exitIfWorstWasPicked else "Cautious") if self._pickWorstFirst else "", "" if self._pickPrevWorstFirst else "Old", "Plus" if self.horizon else "", player, str_Mbest, str_chosen_arm, "($T={}$)".format(self.horizon) if self.horizon else "")

    def startGame(self):
        """Start game."""
        super(oneRandTopMEst, self).startGame()
        # Est part
        self.nbPlayersEstimate = self.maxRank = 1  # Optimistic: start by assuming it is alone!
        self.collisionCount.fill(0)
        self.timeSinceLastCollision = 0
        self.t = 0

    def handleCollision(self, arm, reward=None):
        """Select a new rank, and maybe update nbPlayersEstimate."""
        super(oneRandTopMEst, self).handleCollision(arm, reward=reward)

        # we can be smart, and stop all this as soon as M = K !
        if self.nbPlayersEstimate < self.nbArms:
            self.collisionCount[arm] += 1
            # print("\n - A oneRandTopMEst player {} saw a collision on {}, since last update of nbPlayersEstimate = {} it is the {} th collision on that arm {}...".format(self, arm, self.nbPlayersEstimate, self.collisionCount[arm], arm))  # DEBUG

            # Then, estimate the current ranking of the arms and the set of the M best arms
            currentBest = self.estimatedBestArms(self.nbPlayersEstimate)
            # print("Current estimation of the {} best arms is {} ...".format(self.nbPlayersEstimate, currentBest))  # DEBUG

            collisionCount_on_currentBest = np.sum(self.collisionCount[currentBest])
            # print("Current count of collision on the {} best arms is {} ...".format(self.nbPlayersEstimate, collisionCount_on_currentBest))  # DEBUG

            # And finally, compare the collision count with the current threshold
            threshold = self.threshold(self.t, self.nbPlayersEstimate, self.horizon)
            # print("Using timeSinceLastCollision = {}, and t = {}, threshold = {:.3g} ...".format(self.timeSinceLastCollision, self.t, threshold))

            if collisionCount_on_currentBest > threshold:
                self.nbPlayersEstimate = self.maxRank = min(1 + self.nbPlayersEstimate, self.nbArms)
                # print("The collision count {} was larger than the threshold {:.3g} se we restart the collision count, and increase the nbPlayersEstimate to {}.".format(collisionCount_on_currentBest, threshold, self.nbPlayersEstimate))  # DEBUG
                self.collisionCount.fill(0)
            # Finally, restart timeSinceLastCollision
            self.timeSinceLastCollision = 0

    def getReward(self, arm, reward):
        """One transmission without collision."""
        self.t += 1
        # Obtaining a reward, even 0, means no collision on that arm for this time
        # So, first, we count one more step without collision
        self.timeSinceLastCollision += 1
        # print("Time since last collision = {} ...".format(self.timeSinceLastCollision))  # DEBUG
        # Then use the reward for the arm learning algorithm
        return super(oneRandTopMEst, self).getReward(arm, reward)


# --- Class RandTopMEst


#: Whether to use or not the variant with the "chair": after using an arm successfully (no collision), a player won't move after future collisions (she assumes the other will move). But she will still change her chosen arm if it lies outside of the estimated M-best. :class:`RandTopMEst` (and variants) uses `False` and :class:`MCTopMEst` (and variants) uses `True`.
WITH_CHAIR = True
WITH_CHAIR = False


#: XXX First experimental idea: when the currently chosen arm lies outside of the estimated Mbest set, force to first try (at least once) the arm with lowest UCB indexes in this Mbest_j(t) set. Used by :class:`RandTopMEstCautious` and :class:`RandTopMEstExtraCautious`, and by :class:`MCTopMEstCautious` and :class:`MCTopMEstExtraCautious`.
OPTIM_PICK_WORST_FIRST = True
OPTIM_PICK_WORST_FIRST = False


#: XXX Second experimental idea: when the currently chosen arm becomes the worst of the estimated Mbest set, leave it (even before it lies outside of Mbest_j(t)). Used by :class:`RandTopMEstExtraCautious` and :class:`MCTopMEstExtraCautious`.
OPTIM_EXIT_IF_WORST_WAS_PICKED = True
OPTIM_EXIT_IF_WORST_WAS_PICKED = False


#: XXX Third experimental idea: when the currently chosen arm becomes the worst of the estimated Mbest set, leave it (even before it lies outside of Mbest_j(t)). **Default now!**. `False` only for :class:`RandTopMEstOld` and :class:`MCTopMEstOld`.
OPTIM_PICK_PREV_WORST_FIRST = False
OPTIM_PICK_PREV_WORST_FIRST = True

class RandTopMEst(BaseMPPolicy):
    """ RandTopMEst: a proposal for an efficient multi-players learning policy, with no prior knowledge of the number of player.
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 withChair=WITH_CHAIR,
                 pickWorstFirst=OPTIM_PICK_WORST_FIRST,
                 exitIfWorstWasPicked=OPTIM_EXIT_IF_WORST_WAS_PICKED,
                 pickPrevWorstFirst=OPTIM_PICK_PREV_WORST_FIRST,
                 threshold=threshold_on_t_doubling_trick, lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - withChair: see ``WITH_CHAIR``,
        - pickWorstFirst: see ``OPTIM_PICK_WORST_FIRST``,
        - exitIfWorstWasPicked: see ``EXIT_IF_WORST_WAS_PICKED``,
        - pickPrevWorstFirst: see ``OPTIM_PICK_PREV_WORST_FIRST``,
        - threshold: the threshold function to use, see :func:`EstimateM.threshold_on_t_with_horizon`, :func:`EstimateM.threshold_on_t_doubling_trick` or :func:`EstimateM.threshold_on_t` above.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = RandTopMEst(nbPlayers, Thompson, nbArms)

        - To get a list of usable players, use ``s.children``.

        .. warning:: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for RandTopMEst class has to be > 0."  # DEBUG
        self.nbPlayers = nbPlayers  #: Number of players
        self.withChair = withChair  #: Using a chair ?
        self.pickWorstFirst = pickWorstFirst  #: Using first optimization ?
        self.exitIfWorstWasPicked = exitIfWorstWasPicked  #: Using second optimization ?
        self.pickPrevWorstFirst = pickPrevWorstFirst  #: Using third optimization ? Default to yes now.
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        fake_maxRank = None
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRandTopMEst(threshold, fake_maxRank, withChair, pickWorstFirst, exitIfWorstWasPicked, pickPrevWorstFirst, self, playerId)

    def __str__(self):
        return "RandTopMEst({} x {})".format(self.nbPlayers, str(self._players[0]))

class RandTopMEstPlus(BaseMPPolicy):
    """ RandTopMEstPlus: a proposal for an efficient multi-players learning policy, with no prior knowledge of the number of player.
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo, horizon,
                 withChair=WITH_CHAIR,
                 pickWorstFirst=OPTIM_PICK_WORST_FIRST,
                 exitIfWorstWasPicked=OPTIM_EXIT_IF_WORST_WAS_PICKED,
                 pickPrevWorstFirst=OPTIM_PICK_PREV_WORST_FIRST,
                 lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - horizon: need to know the horizon :math:`T`.
        - withChair: see ``WITH_CHAIR``,
        - pickWorstFirst: see ``OPTIM_PICK_WORST_FIRST``,
        - exitIfWorstWasPicked: see ``EXIT_IF_WORST_WAS_PICKED``,
        - pickPrevWorstFirst: see ``OPTIM_PICK_PREV_WORST_FIRST``,
        - threshold: the threshold function to use, see :func:`threshold_on_t_with_horizon` or :func:`threshold_on_t` above.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = RandTopMEstPlus(nbPlayers, Thompson, nbArms)

        - To get a list of usable players, use ``s.children``.

        .. warning:: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for RandTopMEstPlus class has to be > 0."  # DEBUG
        self.nbPlayers = nbPlayers  #: Number of players
        self.withChair = withChair  #: Using a chair ?
        self.pickWorstFirst = pickWorstFirst  #: Using first optimization ?
        self.exitIfWorstWasPicked = exitIfWorstWasPicked  #: Using second optimization ?
        self.pickPrevWorstFirst = pickPrevWorstFirst  #: Using third optimization ? Default to yes now.
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        fake_maxRank = None
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRandTopMEst(threshold_on_t_with_horizon, fake_maxRank, withChair, pickWorstFirst, exitIfWorstWasPicked, pickPrevWorstFirst, self, playerId, horizon=horizon)

    def __str__(self):
        return "RandTopMEstPlus({} x {})".format(self.nbPlayers, str(self._players[0]))


class MCTopMEst(RandTopMEst):
    """ MCTopMEst: another proposal for an efficient multi-players learning policy, more "stationary" than RandTopMEst.

    .. warning:: Still very experimental! But it seems to be the most efficient decentralized MP algorithm we have so far...
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo,
                 lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.
        """
        super(MCTopMEst, self).__init__(nbPlayers, nbArms, playerAlgo, withChair=True, pickWorstFirst=False, exitIfWorstWasPicked=False, pickPrevWorstFirst=True, lower=lower, amplitude=amplitude, *args, **kwargs)

    def __str__(self):
        return "MCTopMEst({} x {})".format(self.nbPlayers, str(self._players[0]))


class MCTopMEstPlus(RandTopMEstPlus):
    """ MCTopMEstPlus: another proposal for an efficient multi-players learning policy, more "stationary" than RandTopMEst.

    .. warning:: Still very experimental! But it seems to be the most efficient decentralized MP algorithm we have so far...
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo, horizon,
                 lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.
        """
        super(MCTopMEstPlus, self).__init__(nbPlayers, nbArms, playerAlgo, horizon, withChair=True, pickWorstFirst=False, exitIfWorstWasPicked=False, pickPrevWorstFirst=True, lower=lower, amplitude=amplitude, *args, **kwargs)

    def __str__(self):
        return "MCTopMEstPlus({} x {})".format(self.nbPlayers, str(self._players[0]))
