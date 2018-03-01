# -*- coding: utf-8 -*-
""" rhoLearn: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/), using a learning algorithm instead of a random exploration for choosing the rank.

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has a random rank_i from 1 to M, and when a collision occurs, rank_i is given by a second learning algorithm, playing on arms = ranks from [1, .., M], where M is the number of player.
- If rankSelection = Uniform, this is like rhoRand, but if it is a smarter policy, it *might* be better! Warning: no theoretical guarantees exist!
- Reference: [Proof-of-Concept System for Opportunistic Spectrum Access in Multi-user Decentralized Networks, S.J.Darak, C.Moy, J.Palicot, EAI 2016](https://dx.doi.org/10.4108/eai.5-9-2016.151647), algorithm 2. (for BayesUCB only)


.. note:: This is not fully decentralized: as each child player needs to know the (fixed) number of players.

"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

try:
    from sys import path
    path.insert(0, '..')
    from Policies import Uniform
except ImportError:
    try:
        from SMPyBandits.Policies import Uniform
    except ImportError:
        print("Warning: ../Policies/Uniform.py was not imported correctly...")  # DEBUG
        # ... Just reimplement it here manually, if not found in ../Policies/Uniform.py
        from random import randint

        class Uniform():
            """Quick reimplementation of Policies.Uniform"""
            def __init__(self, nbArms, lower=0., amplitude=1.):
                self.nbArms = nbArms

            def startGame(self):
                pass

            def getReward(self, arm, reward):
                pass

            def choice(self):
                return randint(0, self.nbArms - 1)

from .rhoRand import oneRhoRand, rhoRand


#: Should oneRhoLearn players select a (possibly new) rank *at each step* ?
#: The algorithm P2 from https://dx.doi.org/10.4108/eai.5-9-2016.151647 suggests to do so.
#: But I found it works better **without** this trick.
CHANGE_RANK_EACH_STEP = True
CHANGE_RANK_EACH_STEP = False


# --- Class oneRhoLearn, for children

class oneRhoLearn(oneRhoRand):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a (possibly new) rank is sampled after observing a collision, from the rankSelection algorithm.
    - When no collision is observed on a arm, a small reward is given to the rank used for this play, in order to learn the best ranks with rankSelection.
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy.
    """

    def __init__(self, maxRank, rankSelectionAlgo, change_rank_each_step, *args, **kwargs):
        super(oneRhoLearn, self).__init__(maxRank, *args, **kwargs)
        self.rankSelection = rankSelectionAlgo(maxRank)
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.rank = None  #: Current rank, starting to 1
        self.change_rank_each_step = change_rank_each_step  #: Change rank at each step?

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<RhoLearn[{}, rank{} ~ {}]>".format(self.playerId + 1, self.mother._players[self.playerId], "" if self.rank is None else (": %i" % self.rank), self.rankSelection)

    def startGame(self):
        """Initialize both rank and arm selection algorithms."""
        self.rankSelection.startGame()
        super(oneRhoLearn, self).startGame()
        self.rank = 1 + self.rankSelection.choice()  # XXX Start with a rank given from the algorithm (probably uniformly at random, not important)

    def getReward(self, arm, reward):
        """Give a 1 reward to the rank selection algorithm (no collision), give reward to the arm selection algorithm, and if self.change_rank_each_step, select a (possibly new) rank."""
        # Obtaining a reward, even 0, means no collision on that arm for this time
        # So, first, we count one more step for this rank

        # First give a reward to the rank selection learning algorithm (== collision avoidance)
        self.rankSelection.getReward(self.rank - 1, 1)
        # Note: this is NOTHING BUT a heuristic! See equation (13) in https://dx.doi.org/10.4108/eai.5-9-2016.151647

        # Then, use the rankSelection algorithm to select a (possibly new) rank
        if self.change_rank_each_step:  # That's new! rhoLearn (can) change its rank at ALL steps!
            self.rank = 1 + self.rankSelection.choice()
            # print(" - A oneRhoLearn player {} received a reward {:.3g}, and selected a (possibly new) rank from her algorithm {} : {} ...".format(self, reward, self.rankSelection, self.rank))  # DEBUG
        # else:
        #     print(" - A oneRhoLearn player {} received a reward {:.3g}, without selecting a new rank...".format(self, reward))  # DEBUG

        # Then use the reward for the arm learning algorithm
        return super(oneRhoLearn, self).getReward(arm, reward)

    def handleCollision(self, arm, reward=None):
        """Give a 0 reward to the rank selection algorithm, and select a (possibly new) rank."""
        # rhoRand UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: rhoRand UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneRhoLearn, self).getReward(arm, reward)

        # And give a 0 reward to this rank
        self.rankSelection.getReward(self.rank - 1, 0)

        # Then, use the rankSelection algorithm to select a (possibly new) rank
        self.rank = 1 + self.rankSelection.choice()
        # print(" - A oneRhoLearn player {} saw a collision, so she had to select a (possibly new) rank from her algorithm {} : {} ...".format(self, self.rankSelection, self.rank))  # DEBUG


# --- Class rhoRand

class rhoLearn(rhoRand):
    """ rhoLearn: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/), using a learning algorithm instead of a random exploration for choosing the rank.
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo, rankSelectionAlgo=Uniform,
                 lower=0., amplitude=1., maxRank=None, change_rank_each_step=CHANGE_RANK_EACH_STEP,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - rankSelectionAlgo: algorithm to use for selecting the ranks.
        - maxRank: maximum rank allowed by the rhoRand child (default to nbPlayers, but for instance if there is 2 × rhoRand[UCB] + 2 × rhoRand[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = rhoLearn(nbPlayers, Thompson, nbArms, Uniform)  # Exactly rhoRand!
        >>> s = rhoLearn(nbPlayers, Thompson, nbArms, UCB)      # Possibly better than rhoRand!

        - To get a list of usable players, use ``s.children``.
        - Warning: ``s._players`` is for internal use ONLY!
        """
        assert nbPlayers > 0, "Error, the parameter 'nbPlayers' for rhoRand class has to be > 0."
        if maxRank is None:
            maxRank = nbPlayers
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.nbPlayers = nbPlayers  #: Number of players
        self._players = [None] * nbPlayers
        self.children = [None] * nbPlayers  #: List of children, fake algorithms
        self.rankSelectionAlgo = rankSelectionAlgo  #: Policy to use to chose the ranks
        self.nbArms = nbArms  #: Number of arms
        self.change_rank_each_step = change_rank_each_step  #: Change rank at every steps?
        for playerId in range(nbPlayers):
            self._players[playerId] = playerAlgo(nbArms, *args, lower=lower, amplitude=amplitude, **kwargs)
            self.children[playerId] = oneRhoLearn(maxRank, rankSelectionAlgo, change_rank_each_step, self, playerId)
        # Fake rankSelection algorithm, for pretty print
        self._rankSelection = rankSelectionAlgo(maxRank)

    def __str__(self):
        return "rhoLearn({} x {}, ranks ~ {})".format(self.nbPlayers, str(self._players[0]), self._rankSelection)
