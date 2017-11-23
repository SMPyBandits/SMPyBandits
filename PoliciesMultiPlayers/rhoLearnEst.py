# -*- coding: utf-8 -*-
r""" rhoLearnEst: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/), using a learning algorithm instead of a random exploration for choosing the rank, and without knowing the number of users.

- It generalizes :class:`PoliciesMultiPlayers.rhoLearn.rhoLearn` simply by letting the ranks be :math:`\{1,\dots,K\}` and not in :math:`\{1,\dots,M\}`, by hoping the learning algorithm will be *"smart enough"* and learn by itself that ranks should be :math:`\leq M`.
- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has a random rank_i from 1 to M, and when a collision occurs, rank_i is given by a second learning algorithm, playing on arms = ranks from [1, .., M], where M is the number of player.
- If rankSelection = Uniform, this is like rhoRand, but if it is a smarter policy, it *might* be better! Warning: no theoretical guarantees exist!
- Reference: [Proof-of-Concept System for Opportunistic Spectrum Access in Multi-user Decentralized Networks, S.J.Darak, C.Moy, J.Palicot, EAI 2016](https://dx.doi.org/10.4108/eai.5-9-2016.151647), algorithm 2. (for BayesUCB only)

.. note:: This is fully decentralized: each child player does *not* need to know the (fixed) number of players, it will learn to select ranks only in :math:`\{1,\dots,M\}` instead of :math:`\{1,\dots,K\}`.

.. warning:: This policy does not work very well!
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

from .rhoLearn import rhoLearn, Uniform, oneRhoLearn, CHANGE_RANK_EACH_STEP


class oneRhoLearnEst(oneRhoLearn):

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<RhoLearnEst[{}, rank{} ~ {}]>".format(self.playerId + 1, self.mother._players[self.playerId], "" if self.rank is None else (": %i" % self.rank), self.rankSelection)


# --- Class rhoRand

class rhoLearnEst(rhoLearn):
    """ rhoLearnEst: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/), using a learning algorithm instead of a random exploration for choosing the rank, and without knowing the number of users.
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo, rankSelectionAlgo=Uniform,
                 lower=0., amplitude=1., change_rank_each_step=CHANGE_RANK_EACH_STEP,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - rankSelectionAlgo: algorithm to use for selecting the ranks.
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Difference with :class:`PoliciesMultiPlayers.rhoLearn.rhoLearn`:

        - maxRank: maximum rank allowed by the rhoRand child, is not an argument, but it is always nbArms (= K).

        Example:

        >>> s = rhoLearnEst(nbPlayers, Thompson, nbArms, Uniform)  # rhoRand but with a wrong estimate of maxRanks!
        >>> s = rhoLearnEst(nbPlayers, Thompson, nbArms, UCB)      # Possibly better than rhoRand!

        - To get a list of usable players, use ``s.children``.
        - Warning: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRand class has to be > 0."
        self.nbPlayers = nbPlayers  #: Number of players
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.rankSelectionAlgo = rankSelectionAlgo  #: Policy to use to chose the ranks
        self.nbArms = nbArms  #: Number of arms
        self.change_rank_each_step = change_rank_each_step  #: Change rank at every steps?
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRhoLearnEst(nbArms, rankSelectionAlgo, change_rank_each_step, self, playerId)
        # Fake rankSelection algorithm, for pretty print
        self._rankSelection = rankSelectionAlgo(nbArms)

    def __str__(self):
        return "rhoLearnEst({} x {}, ranks ~ {})".format(self.nbPlayers, str(self._players[0]), self._rankSelection)
