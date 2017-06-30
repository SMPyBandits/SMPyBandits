# -*- coding: utf-8 -*-
""" rhoLearnExp3: implementation of a variant of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/), using the Exp3 learning algorithm instead of a random exploration for choosing the rank.

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has rank_i = 1, but when a collision occurs, rank_i is given by a second learning algorithm, playing on arms = ranks from [1, .., M], where M is the number of player.
- If rankSelection = Uniform, this is like rhoRand, but if it is a smarter policy (like Exp3 here), it *might* be better! Warning: no theoretical guarantees exist!
- Reference: [Proof-of-Concept System for Opportunistic Spectrum Access in Multi-user Decentralized Networks, S.J.Darak, C.Moy, J.Palicot, EAI 2016](https://dx.doi.org/10.4108/eai.5-9-2016.151647), algorithm 2. (for BayesUCB only)


.. note:: This is not fully decentralized: as each child player needs to know the (fixed) number of players.

"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.6"

try:
    from sys import path
    path.insert(0, '..')
    from Policies import Exp3
except ImportError as e:
    print("Warning: ../Policies/Exp3.py was not imported correctly...")  # DEBUG
    raise e

from .rhoRand import oneRhoRand, rhoRand


#: Should oneRhoLearnExp3 players select a new rank *at each step* ?
#: The algorithm P2 from https://dx.doi.org/10.4108/eai.5-9-2016.151647 suggests to do so.
#: But I found it works better *without* this trick.
CHANGE_RANK_EACH_STEP = True
CHANGE_RANK_EACH_STEP = False


def successful_transmission(sensing, collision):
    r""" Count 1 iff the sensing authorized to communicate and no collision was observed.

    .. math::

       \mathrm{reward}(\text{user}\;j, \text{time}\;t) &:= r_{j,t} = F_{m,t} \times (1 - c_{m,t}), \\
       \text{where}\;\; F_{m,t} \; \text{is the sensing feedback (1 iff channel is free)}, \\
       \text{and}  \;\; c_{m,t} \; \text{is the collision feedback (1 iff user j experienced a collision)}.
    """
    assert 0 <= sensing <= 1, "Error: 'sensing' argument was not in [0, 1] (was {:.3g}).".format(sensing)  # DEBUG
    if sensing not in {0, 1}:
        print("Warning: 'sensing' argument was not 0 or 1, but this policy rhoLearnExp3 was only designed for binary sensing model... (was {:.3g}).".format(sensing))  # DEBUG
    assert collision in {0, 1}, "Error: 'collision' argument was not binary, it can only be 0 or 1 (was {:.3g}).".format(collision)  # DEBUG
    return sensing * (1 - collision)


def ternary_feedback(sensing, collision):
    r""" Count 1 iff the sensing authorized to communicate and no collision was observed, 0 if no communication, and -1 iff communication but a collision was observed.

    .. math::

       \mathrm{reward}(\text{user}\;j, \text{time}\;t) &:= F_{m,t} \times (2 r_{m,t} - 1), \\
       \text{where}\;\; r_{j,t} &:= F_{m,t} \times (1 - c_{m,t}), \\
       \text{and}  \;\; F_{m,t} &\; \text{is the sensing feedback (1 iff channel is free)}, \\
       \text{and}  \;\; c_{m,t} &\; \text{is the collision feedback (1 iff user j experienced a collision)}.
    """
    assert 0 <= sensing <= 1, "Error: 'sensing' argument was not in [0, 1] (was {:.3g}).".format(sensing)  # DEBUG
    if sensing not in {0, 1}:
        print("Warning: 'sensing' argument was not 0 or 1, but this policy rhoLearnExp3 was only designed for binary sensing model... (was {:.3g}).".format(sensing))  # DEBUG
    assert collision in {0, 1}, "Error: 'collision' argument was not binary, it can only be 0 or 1 (was {:.3g}).".format(collision)  # DEBUG
    first_reward = sensing * (1 - collision)
    assert 0 <= first_reward <= 1, "Error: variable 'first_reward' should have been only binary 0 or 1 (was {:.3g}).".format(first_reward)
    reward = sensing * (2 * first_reward - 1)
    assert -1 <= reward <= 1, "Error: variable 'reward' should have been only binary 0 or 1 (was {:.3g}).".format(reward)
    if reward not in {-1, 0, 1}:
        print("Warning: 'reward' argument was not -1, 0 or 1, but this function should give ternary reward... (was {:.3g}).".format(reward))  # DEBUG
    return reward


def generic_ternary_feedback(sensing, collision, bonus=1, malus=-1):
    r""" Count 'bonus' iff the sensing authorized to communicate and no collision was observed, 'malus' iff communication but a collision was observed, and 0 if no communication.
    """
    reward = ternary_feedback
    assert malus < bonus, "Error: parameters 'malus' = {:.3g} is supposed to be smaller than 'bonus' = {:.3g}".format(malus, bonus)  # DEBUG
    mapped_reward = 0
    if reward == -1:
        mapped_reward = malus
    elif reward == +1:
        mapped_reward = bonus
    assert malus <= mapped_reward <= bonus, "Error: 'mapped_reward' = {:.3g} is supposed to be either 'malus' = {:.3g}, 0 or 'bonus' = {:.3g}".format(mapped_reward, malus, bonus)  # DEBUG
    return mapped_reward


def generic_continuous_feedback(sensing, collision, bonus=1, malus=-1):
    r""" Count 'bonus' iff the sensing authorized to communicate and no collision was observed, 'malus' iff communication but a collision was observed, *but possibly does not count* 0 if no communication.

    .. math::

       \mathrm{reward}(\text{user}\;j, \text{time}\;t) &:= \mathrm{malus} + (\mathrm{bonus} - \mathrm{malus}) \times \frac{r'_{j,t} + 1}{2}, \\
       \text{where}\;\; r'_{j,t} &:= F_{m,t} \times (2 r_{m,t} - 1), \\
       \text{where}\;\; r_{j,t} &:= F_{m,t} \times (1 - c_{m,t}), \\
       \text{and}  \;\; F_{m,t} &\; \text{is the sensing feedback (1 iff channel is free)}, \\
       \text{and}  \;\; c_{m,t} &\; \text{is the collision feedback (1 iff user j experienced a collision)}.
    """
    reward = ternary_feedback
    assert malus < bonus, "Error: parameters 'malus' = {:.3g} is supposed to be smaller than 'bonus' = {:.3g}".format(malus, bonus)  # DEBUG
    mapped_reward = malus + (bonus - malus) * (reward + 1) / 2.
    assert malus <= mapped_reward <= bonus, "Error: 'mapped_reward' = {:.3g} is supposed to be between 'malus' = {:.3g} and 'bonus' = {:.3g}".format(mapped_reward, malus, bonus)  # DEBUG
    return mapped_reward


#: List of possible function mapping the two decoupled feedback to a numerical reward
functions__reward_from_decoupled_feedback = [
    successful_transmission,
    ternary_feedback,
    generic_ternary_feedback,
    generic_continuous_feedback,
]

#: Decide the default function to use
reward_from_decoupled_feedback = successful_transmission
reward_from_decoupled_feedback = ternary_feedback


# --- Class oneRhoLearnExp3, for children

class oneRhoLearnExp3(oneRhoRand):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a new rank is sampled after observing a collision, from the rankSelection algorithm.
    - When no collision is observed on a arm, a small reward is given to the rank used for this play, in order to learn the best ranks with rankSelection.
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy.
    """

    def __init__(self, maxRank, rankSelectionAlgo, change_rank_each_step, *args, **kwargs):
        super(oneRhoLearnExp3, self).__init__(maxRank, *args, **kwargs)
        self.rankSelection = rankSelectionAlgo(maxRank)  # FIXME I should give it more arguments?
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.rank = None  #: Current rank, starting to 1
        self.change_rank_each_step = change_rank_each_step  #: Change rank at each step?
        # Keep in memory how many times a rank could be used while giving no collision
        # self.timesUntilCollision = np.zeros(maxRank, dtype=int)  # XXX not used anymore!

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<{}[{}, rank{} ~ {}]>".format(self.playerId + 1, r"$\rho^{\mathrm{Learn}}$", self.mother._players[self.playerId], "" if self.rank is None else (": %i" % self.rank), self.rankSelection)

    def startGame(self):
        """Initialize both rank and arm selection algorithms."""
        self.rankSelection.startGame()
        super(oneRhoLearnExp3, self).startGame()
        self.rank = 1  # Start with a rank = 1: assume she is alone.
        # self.timesUntilCollision.fill(0)  # XXX not used anymore!

    def getReward(self, arm, reward):
        """Give a 1 reward to the rank selection algorithm (no collision), give reward to the arm selection algorithm, and if self.change_rank_each_step, select a new rank."""
        # Obtaining a reward, even 0, means no collision on that arm for this time
        # So, first, we count one more step for this rank
        # self.timesUntilCollision[self.rank - 1] += 1  # XXX not used anymore!

        # First give a reward to the rank selection learning algorithm (== collision avoidance)
        self.rankSelection.getReward(self.rank - 1, 1)
        # Note: this is NOTHING BUT a heuristic! See equation (13) in https://dx.doi.org/10.4108/eai.5-9-2016.151647

        # Then, use the rankSelection algorithm to select a new rank
        if self.change_rank_each_step:  # That's new! rhoLearnExp3 (can) change its rank at ALL steps!
            self.rank = 1 + self.rankSelection.choice()

        # Then use the reward for the arm learning algorithm
        return super(oneRhoLearnExp3, self).getReward(arm, reward)

    def handleCollision(self, arm, reward=None):
        """Give a 0 reward to the rank selection algorithm, and select a new rank."""
        # rhoRand UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: rhoRand UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneRhoLearnExp3, self).getReward(arm, reward)

        # First, reset the time until collisions for that rank
        # self.timesUntilCollision[self.rank - 1] = 0  # XXX not used anymore!

        # And give a 0 reward to this rank
        self.rankSelection.getReward(self.rank - 1, 0)

        # Then, use the rankSelection algorithm to select a new rank
        self.rank = 1 + self.rankSelection.choice()
        # print(" - A oneRhoLearnExp3 player {} saw a collision, so she had to select a new rank from her algorithm {} : {} ...".format(self, self.rankSelection, self.rank))  # DEBUG


# --- Class rhoRand

class rhoLearnExp3(rhoRand):
    """ rhoLearnExp3: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/), using a learning algorithm instead of a random exploration for choosing the rank.
    """

    def __init__(self, nbPlayers, playerAlgo, nbArms, rankSelectionAlgo=Exp3,
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

        >>> s = rhoLearnExp3(nbPlayers, Thompson, nbArms, Uniform)  # Exactly rhoRand!
        >>> s = rhoLearnExp3(nbPlayers, Thompson, nbArms, UCB)      # Possibly better than rhoRand!

        - To get a list of usable players, use s.children.
        - Warning: s._players is for internal use ONLY!
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
            self.children[playerId] = oneRhoLearnExp3(maxRank, rankSelectionAlgo, change_rank_each_step, self, playerId)
        # Fake rankSelection algorithm, for pretty print
        self._rankSelection = rankSelectionAlgo(maxRank)

    def __str__(self):
        return "rhoLearnExp3({} x {}, ranks ~ {})".format(self.nbPlayers, str(self._players[0]), self._rankSelection)
