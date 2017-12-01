# -*- coding: utf-8 -*-
r""" EstimateM: generic wrapper on a multi-player decentralized learning policy, to learn on the run the number of players, adapted from rhoEst from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

- The procedure to estimate :math:`\hat{M}_i(t)` is not so simple, but basically everyone starts with :math:`\hat{M}_i(0) = 1`, and when colliding :math:`\hat{M}_i(t+1) = \hat{M}_i(t) + 1`, for some time (with a complicated threshold).

- My choice for the threshold function, see :func:`threshold_on_t`, does not need the horizon either, and uses :math:`t` instead.

.. note:: This is fully decentralized: each child player does NOT need to know the number of players and does NOT require the horizon :math:`T`.

.. warning:: This is still very experimental!

.. note:: For a less generic approach, see the policies defined in :class:`rhoEst.rhoEst` (generalizing :class:`rhoRand.rhoRand`) and :class:`RandTopMEst.RandTopMEst` (generalizing :class:`RandTopM.RandTopM`).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.8"

import numpy as np
import numpy.random as rn

from .BaseMPPolicy import BaseMPPolicy
from .ChildPointer import ChildPointer


# --- threshold function xi(n, k)


def threshold_on_t_with_horizon(t, nbPlayersEstimate, horizon=None):
    r""" Function :math:`\xi(T, k)` used as a threshold in :class:`rhoEstPlus`.

    - `0` if `nbPlayersEstimate` is `0`,
    - `1` if `nbPlayersEstimate` is `1`,
    - any function such that: :math:`\xi(T, k) = \omega(\log T)` for all `k > 1`. (cf. http://mathworld.wolfram.com/Little-OmegaNotation.html). I choose :math:`\log(1 + T)^2` or :math:`\log(1 + T) \log(1 + \log(1 + T))`, as it seems to work just fine and satisfies the condition (25) from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).

    .. warning:: It requires the horizon :math:`T`, and does not use the current time :math:`t`.
    """
    # print("Calling threshold function 'threshold_on_t_with_horizon' with t = {}, nbPlayersEstimate = {} and horizon = {} ...".format(t, nbPlayersEstimate, horizon))  # DEBUG
    if nbPlayersEstimate <= 1:
        return nbPlayersEstimate
    else:
        if horizon is None:
            horizon = t
        return np.log(1 + horizon) * np.log(1 + np.log(1 + horizon))
        # return np.log(1 + horizon) ** 2
        # return float(horizon) ** 0.7
        # return float(horizon) ** 0.5
        # return float(horizon) ** 0.1
        # return float(horizon)


def threshold_on_t_doubling_trick(t, nbPlayersEstimate, horizon=None, base=2, min_fake_horizon=1000, T0=1):
    r""" A trick to have a threshold depending on a growing horizon (doubling-trick).

    - Instead of using :math:`t` or :math:`T`, a fake horizon :math:`T_t` is used, corresponding to the horizon a doubling-trick algorithm would be using at time :math:`t`.
    - :math:`T_t = T_0 b^{\lceil \log_b(t) \rceil}` is the default choice, for :math:`b=2` :math:`T_0 = 10`.
    - If :math:`T_t` is too small, ``min_fake_horizon`` is used instead.

    .. warning:: This is ongoing research!
    """
    fake_horizon_now = max(T0 * (base ** (np.ceil(np.log(1 + t) / np.log(base)))), min_fake_horizon)
    return threshold_on_t_with_horizon(t, nbPlayersEstimate, horizon=fake_horizon_now)


def threshold_on_t(t, nbPlayersEstimate, horizon=None):
    r""" Function :math:`\xi(t, k)` used as a threshold in :class:`rhoEst`.

    - `0` if `nbPlayersEstimate` is `0`,
    - `1` if `nbPlayersEstimate` is `1`,
    - My heuristic to be any-time (ie, without needing to know the horizon) is to use a function of :math:`t` (current time) and not :math:`T` (horizon).
    - The choice which seemed to perform the best in practice was :math:`\xi(t, k) = c t` for a small constant :math:`c` (like 5 or 10).
    """
    # print("Calling threshold function 'threshold_on_t' with t = {}, nbPlayersEstimate = {} and horizon = {} ...".format(t, nbPlayersEstimate, horizon))  # DEBUG
    if nbPlayersEstimate <= 1:
        return nbPlayersEstimate
    else:
        return np.log(1 + t) ** 2
        # return float(t) ** 0.7
        # return float(t) ** 0.5
        # return float(t) ** 0.1
        # return float(t)
        # return 10 * float(t)


# --- Class oneEstimateM, for children

class oneEstimateM(ChildPointer):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - The procedure to estimate :math:`\hat{M}_i(t)` is not so simple, but basically everyone starts with :math:`\hat{M}_i(0) = 1`, and when colliding :math:`\hat{M}_i(t+1) = \hat{M}_i(t) + 1`, for some time (with a complicated threshold).
    """

    def __init__(self, nbArms, playerAlgo, threshold, decentralizedPolicy, *args,
                 lower=0., amplitude=1., horizon=None,
                 args_decentralizedPolicy=None, kwargs_decentralizedPolicy=None,
                 **kwargs):
        self.horizon = horizon
        super(oneEstimateM, self).__init__(*args, **kwargs)
        # Creating of the underlying policy (e.g., oneRhoRand, oneRandTopM etc)
        if args_decentralizedPolicy is None:
            args_decentralizedPolicy = ()
        if kwargs_decentralizedPolicy is None:
            kwargs_decentralizedPolicy = {}
        self._fakeMother = decentralizedPolicy(
            1, nbArms, playerAlgo,
            *args_decentralizedPolicy,
            lower=lower, amplitude=amplitude,
            **kwargs_decentralizedPolicy
        )
        self._policy = self._fakeMother.children[0]
        # Parameters
        self.threshold = threshold  #: Threshold function
        # Internal variables
        self.nbPlayersEstimate = 1  #: Number of players. Optimistic: start by assuming it is alone!
        self.updateNbPlayers()
        self.collisionCount = np.zeros(self.nbArms, dtype=int)  #: Count collisions on each arm, since last increase of nbPlayersEstimate
        self.timeSinceLastCollision = 0  #: Time since last collision. Don't remember why I thought using this could be useful... But it's not!
        self.t = 0  #: Internal time

    def __str__(self):   # Better to recompute it automatically
        parts = self._policy.__str__().split('<')
        if len(parts) == 1:
            return "EstimateM-{}".format(parts[0])
        else:
            return parts[0] + '<EstimateM-' + '<'.join(parts[1:])
        # EstimateM-#1<RhoRand-KLUCB, rank:2> --> #1<EstimateM-RhoRand-KLUCB, rank:2>

    def updateNbPlayers(self, nbPlayers=None):
        """Change the value of ``nbPlayersEstimate``, and propagate the change to the underlying policy, for parameters called ``maxRank`` or ``nbPlayers``."""
        # print("DEBUG calling updateNbPlayers for self = {} and nbPlayers = {} and self.nbPlayersEstimate = {} ...".format(self, nbPlayers, self.nbPlayersEstimate))  # DEBUG
        if nbPlayers is None:
            nbPlayers = self.nbPlayersEstimate
        else:
            self.nbPlayersEstimate = nbPlayers
        if hasattr(self._policy, 'maxRank'):
            self._policy.maxRank = nbPlayers
            # print("DEBUG in updateNbPlayers, propagating the value {} as new maxRank for self._policy = {} ...".format(nbPlayers, self._policy))  # DEBUG
        if hasattr(self._policy, 'nbPlayers'):
            self._policy.nbPlayers = nbPlayers
            # print("DEBUG in updateNbPlayers, propagating the value {} as new nbPlayers for self._policy = {} ...".format(nbPlayers, self._policy))  # DEBUG

    def startGame(self):
        """Start game."""
        self._policy.startGame()
        self.nbPlayersEstimate = 1  # Optimistic: start by assuming it is alone!
        self.updateNbPlayers()
        self.collisionCount.fill(0)
        self.timeSinceLastCollision = 0
        self.t = 0

    def handleCollision(self, arm, reward=None):
        """Select a new rank, and maybe update nbPlayersEstimate."""
        self._policy.handleCollision(arm, reward=reward)

        # we can be smart, and stop all this as soon as M = K !
        if self.nbPlayersEstimate < self.nbArms:
            self.collisionCount[arm] += 1
            # print("\n - A oneRhoEst player {} saw a collision on {}, since last update of nbPlayersEstimate = {} it is the {} th collision on that arm {}...".format(self, arm, self.nbPlayersEstimate, self.collisionCount[arm], arm))  # DEBUG

            # Then, estimate the current ranking of the arms and the set of the M best arms
            currentBest = self.estimatedBestArms(self.nbPlayersEstimate)
            # print("Current estimation of the {} best arms is {} ...".format(self.nbPlayersEstimate, currentBest))  # DEBUG

            collisionCount_on_currentBest = np.sum(self.collisionCount[currentBest])
            # print("Current count of collision on the {} best arms is {} ...".format(self.nbPlayersEstimate, collisionCount_on_currentBest))  # DEBUG

            # And finally, compare the collision count with the current threshold
            threshold = self.threshold(self.t, self.nbPlayersEstimate, self.horizon)
            # print("Using timeSinceLastCollision = {}, and t = {}, threshold = {:.3g} ...".format(self.timeSinceLastCollision, self.t, threshold))

            if collisionCount_on_currentBest > threshold:
                self.nbPlayersEstimate = min(1 + self.nbPlayersEstimate, self.nbArms)
                self.updateNbPlayers()
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
        # Then use the reward for the arm learning algorithm
        return self._policy.getReward(arm, reward)

    def choice(self):
        """ Pass the call to self._policy.choice() with the player's ID number. """
        return self._policy.choice()

    def choiceWithRank(self, rank=1):
        """ Pass the call to self._policy.choiceWithRank() with the player's ID number. """
        return self._policy.choiceWithRank(rank)

    def choiceFromSubSet(self, availableArms='all'):
        """ Pass the call to self._policy.choiceFromSubSet() with the player's ID number. """
        return self._policy.choiceFromSubSet(availableArms)

    def choiceMultiple(self, nb=1):
        """ Pass the call to self._policy.choiceMultiple() with the player's ID number. """
        return self._policy.choiceMultiple(nb)

    def choiceIMP(self, nb=1):
        """ Pass the call to self._policy.choiceIMP() with the player's ID number. """
        return self._policy.choiceIMP(nb)

    def estimatedOrder(self):
        """ Pass the call to self._policy.estimatedOrder() with the player's ID number. """
        return self._policy.estimatedOrder()

    def estimatedBestArms(self, M=1):
        """ Pass the call to self._policy.estimatedBestArms() with the player's ID number. """
        return self._policy.estimatedBestArms(M=M)


class EstimateM(BaseMPPolicy):
    """ EstimateM: a generic wrapper for an efficient multi-players learning policy, with no prior knowledge of the number of player, and using any other MP policy.
    """

    def __init__(self, nbPlayers, nbArms, decentralizedPolicy, playerAlgo,
                 policyArgs=None, horizon=None,
                 threshold=threshold_on_t_doubling_trick,
                 lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - nbArms: number of arms.
        - decentralizedPolicy: base MP decentralized policy.
        - threshold: the threshold function to use, see :func:`threshold_on_t_with_horizon`, :func:`threshold_on_t_doubling_trick` or :func:`threshold_on_t` above.
        - `policyArgs`: named arguments (dictionnary), given to ``decentralizedPolicy``.
        - `*args`, `**kwargs`: arguments, named arguments, given to ``decentralizedPolicy`` (will probably be given to the single-player decentralized policy under the hood, don't care).

        Example:

        >>> s = EstimateM(nbPlayers, nbArms, rhoRand, UCB, alpha=0.5)

        - To get a list of usable players, use ``s.children``.

        .. warning:: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for RandTopMEst class has to be > 0."  # DEBUG
        self.nbPlayers = nbPlayers  #: Number of players
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.nbArms = nbArms  #: Number of arms
        if policyArgs is None:
            policyArgs = {}
        args_decentralizedPolicy = args
        kwargs_decentralizedPolicy = kwargs
        for playerId in range(nbPlayers):
            self.children[playerId] = oneEstimateM(nbArms, playerAlgo, threshold, decentralizedPolicy, self, playerId, lower=lower, amplitude=amplitude, horizon=horizon, args_decentralizedPolicy=args_decentralizedPolicy, kwargs_decentralizedPolicy=kwargs_decentralizedPolicy, **policyArgs)
            self._players[playerId] = self.children[playerId]._policy

    def __str__(self):
        return "EstimateM({} x {})".format(self.nbPlayers, str(self._players[0]))
