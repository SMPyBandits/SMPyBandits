# -*- coding: utf-8 -*-
""" rhoLearnExp3: implementation of a variant of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/), using the Exp3 learning algorithm instead of a random exploration for choosing the rank.

- Each child player is selfish, and plays according to an index policy (any index policy, e.g., UCB, Thompson, KL-UCB, BayesUCB etc),
- But instead of aiming at the best (the 1-st best) arm, player i aims at the rank_i-th best arm,
- At first, every player has a random rank_i from 1 to M, and when a collision occurs, rank_i is given by a second learning algorithm, playing on arms = ranks from [1, .., M], where M is the number of player.
- If rankSelection = Uniform, this is like rhoRand, but if it is a smarter policy (like Exp3 here), it *might* be better! Warning: no theoretical guarantees exist!
- Reference: [Proof-of-Concept System for Opportunistic Spectrum Access in Multi-user Decentralized Networks, S.J.Darak, C.Moy, J.Palicot, EAI 2016](https://dx.doi.org/10.4108/eai.5-9-2016.151647), algorithm 2. (for BayesUCB only)


.. note:: This is not fully decentralized: as each child player needs to know the (fixed) number of players.

For the Exp3 algorithm:

- Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi, §3.1](http://research.microsoft.com/en-us/um/people/sebubeck/SurveyBCB12.pdf)
- See also [Evaluation and Analysis of the Performance of the EXP3 Algorithm in Stochastic Environments, Y. Seldin & C. Szepasvari & P. Auer & Y. Abbasi-Adkori, 2012](http://proceedings.mlr.press/v24/seldin12a/seldin12a.pdf).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
import numpy.random as rn

try:
    from sys import path
    path.insert(0, '..')
    from Policies import Exp3, Exp3Decreasing
except ImportError as e:
    try:
        from SMPyBandits.Policies import Exp3, Exp3Decreasing
    except ImportError:
        print("Warning: ../Policies/Exp3.py was not imported correctly...")  # DEBUG
        raise e

from .rhoRand import oneRhoRand, rhoRand


# --- Define various function mapping the two decoupled feedback to a numerical reward

def binary_feedback(sensing, collision):
    r""" Count 1 iff the sensing authorized to communicate and no collision was observed.

    .. math::

       \mathrm{reward}(\text{user}\;j, \text{time}\;t) &:= r_{j,t} = F_{m,t} \times (1 - c_{m,t}), \\
       \text{where}\;\; F_{m,t} &\; \text{is the sensing feedback (1 iff channel is free)}, \\
       \text{and}  \;\; c_{m,t} &\; \text{is the collision feedback (1 iff user j experienced a collision)}.
    """
    assert 0 <= sensing <= 1, "Error: 'sensing' argument was not in [0, 1] (was {:.3g}).".format(sensing)  # DEBUG
    if sensing not in [0, 1]:
        print("Warning: 'sensing' argument was not 0 or 1, but this policy rhoLearnExp3 was only designed for binary sensing model... (was {:.3g}).".format(sensing))  # DEBUG
    assert collision in [0, 1], "Error: 'collision' argument was not binary, it can only be 0 or 1 (was {:.3g}).".format(collision)  # DEBUG
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
    if sensing not in [0, 1]:
        print("Warning: 'sensing' argument was not 0 or 1, but this policy rhoLearnExp3 was only designed for binary sensing model... (was {:.3g}).".format(sensing))  # DEBUG
    assert collision in [0, 1], "Error: 'collision' argument was not binary, it can only be 0 or 1 (was {:.3g}).".format(collision)  # DEBUG
    first_reward = sensing * (1 - collision)
    assert 0 <= first_reward <= 1, "Error: variable 'first_reward' should have been only binary 0 or 1 (was {:.3g}).".format(first_reward)  # DEBUG
    reward = sensing * (2 * first_reward - 1)
    assert -1 <= reward <= 1, "Error: variable 'reward' should have been only binary 0 or 1 (was {:.3g}).".format(reward)  # DEBUG
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


def make_generic_ternary_feedback(bonus=1, malus=-1):
    if bonus is None:
        bonus = 1
    if malus is None:
        malus = -1
    return lambda sensing, collision: generic_ternary_feedback(sensing, collision, bonus=bonus, malus=malus)


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


def make_generic_continuous_feedback(bonus=1, malus=-1):
    if bonus is None:
        bonus = 1
    if malus is None:
        malus = -1
    return lambda sensing, collision: generic_continuous_feedback(sensing, collision, bonus=bonus, malus=malus)


NAME_OF_FEEDBACKS = {
    "binary_feedback": "$0/1$",
    "ternary_feedback": "$-1/0/1$"
}

#: Decide the default function to use.
#: FIXME try all of them!
reward_from_decoupled_feedback = binary_feedback
# reward_from_decoupled_feedback = ternary_feedback


#: Should oneRhoLearnExp3 players select a (possibly new) rank *at each step* ?
#: The algorithm P2 from https://dx.doi.org/10.4108/eai.5-9-2016.151647 suggests to do so.
#: But I found it works better *without* this trick.
CHANGE_RANK_EACH_STEP = True
CHANGE_RANK_EACH_STEP = False


# --- Class oneRhoLearnExp3, for children

class oneRhoLearnExp3(oneRhoRand):
    """ Class that acts as a child policy, but in fact it pass all its method calls to the mother class, who passes it to its i-th player.

    - Except for the handleCollision method: a (possibly new) rank is sampled after observing a collision, from the rankSelection algorithm.
    - When no collision is observed on a arm, a small reward is given to the rank used for this play, in order to learn the best ranks with rankSelection.
    - And the player does not aim at the best arm, but at the rank-th best arm, based on her index policy.
    """

    def __init__(self, maxRank,
                 rankSelectionAlgo, change_rank_each_step, feedback_function,
                 *args, **kwargs):
        super(oneRhoLearnExp3, self).__init__(maxRank, *args, **kwargs)
        self.rankSelection = rankSelectionAlgo(maxRank)
        self.maxRank = maxRank  #: Max rank, usually nbPlayers but can be different
        self.rank = None  #: Current rank, starting to 1
        self.change_rank_each_step = change_rank_each_step  #: Change rank at each step?
        self.feedback_function = feedback_function  #: Feedback function: (sensing, collision) -> reward
        feedback_name = str(feedback_function.__name__)
        self.feedback_function_label = ", feedback:{}".format(NAME_OF_FEEDBACKS[feedback_name]) if feedback_name in NAME_OF_FEEDBACKS else ""

    def __str__(self):   # Better to recompute it automatically
        return r"#{}<RhoLearn[{}, rank{} ~ {}{}]>".format(self.playerId + 1, self.mother._players[self.playerId], "" if self.rank is None else (": %i" % self.rank), self.rankSelection, self.feedback_function_label)

    def startGame(self):
        """Initialize both rank and arm selection algorithms."""
        self.rankSelection.startGame()
        super(oneRhoLearnExp3, self).startGame()
        self.rank = 1 + rn.randint(self.maxRank)  # XXX Start with a random rank, safer to avoid first collisions.

    def getReward(self, arm, reward):
        """Give a "good" reward to the rank selection algorithm (no collision), give reward to the arm selection algorithm, and if self.change_rank_each_step, select a (possibly new) rank."""
        # Obtaining a reward, even 0, means no collision on that arm for this time

        # First give a reward to the rank selection learning algorithm (== collision avoidance)
        reward_on_rank = self.feedback_function(reward, 0)
        self.rankSelection.getReward(self.rank - 1, reward_on_rank)
        # Note: this is NOTHING BUT a heuristic! See equation (13) in https://dx.doi.org/10.4108/eai.5-9-2016.151647

        # Then, use the rankSelection algorithm to select a (possibly new) rank
        if self.change_rank_each_step:  # That's new! rhoLearnExp3 (can) change its rank at ALL steps!
            self.rank = 1 + self.rankSelection.choice()
            # print(" - A oneRhoLearnExp3 player {} received a reward {:.3g}, and selected a (possibly new) rank from her algorithm {} : {} ...".format(self, reward, self.rankSelection, self.rank))  # DEBUG
        # else:
        #     print(" - A oneRhoLearnExp3 player {} received a reward {:.3g}, without selecting a new rank...".format(self, reward))  # DEBUG

        # Then use the reward for the arm learning algorithm
        return super(oneRhoLearnExp3, self).getReward(arm, reward)

    # WARNING here reward=None is NOT present: reward is MANDATORY HERE
    def handleCollision(self, arm, reward):
        """Give a "bad" reward to the rank selection algorithm, and select a (possibly new) rank."""
        # rhoRand UCB indexes learn on the SENSING, not on the successful transmissions!
        if reward is not None:
            # print("Info: rhoRand UCB internal indexes DOES get updated by reward, in case of collision, learning is done on SENSING, not successful transmissions!")  # DEBUG
            super(oneRhoLearnExp3, self).getReward(arm, reward)

        # And give a reward to this rank
        reward_on_rank = self.feedback_function(reward, 1)
        self.rankSelection.getReward(self.rank - 1, reward_on_rank)

        # Then, use the rankSelection algorithm to select a (possibly new) rank
        self.rank = 1 + self.rankSelection.choice()
        # print(" - A oneRhoLearnExp3 player {} saw a collision, so she had to select a (possibly new) rank from her algorithm {} : {} ...".format(self, self.rankSelection, self.rank))  # DEBUG


# --- Class rhoRand

class rhoLearnExp3(rhoRand):
    """ rhoLearnExp3: implementation of the multi-player policy from [Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/), using a learning algorithm instead of a random exploration for choosing the rank.
    """

    def __init__(self, nbPlayers, nbArms, playerAlgo, rankSelectionAlgo=Exp3Decreasing,
                 maxRank=None, change_rank_each_step=CHANGE_RANK_EACH_STEP,
                 feedback_function=reward_from_decoupled_feedback,
                 lower=0., amplitude=1.,
                 *args, **kwargs):
        """
        - nbPlayers: number of players to create (in self._players).
        - playerAlgo: class to use for every players.
        - nbArms: number of arms, given as first argument to playerAlgo.
        - rankSelectionAlgo: algorithm to use for selecting the ranks.
        - maxRank: maximum rank allowed by the rhoRand child (default to nbPlayers, but for instance if there is 2 × rhoRand[UCB] + 2 × rhoRand[klUCB], maxRank should be 4 not 2).
        - `*args`, `**kwargs`: arguments, named arguments, given to playerAlgo.

        Example:

        >>> s = rhoLearnExp3(nbPlayers, BayesUCB, nbArms, Uniform)  # Exactly rhoRand!
        >>> s = rhoLearnExp3(nbPlayers, BayesUCB, nbArms)           # Possibly better than rhoRand!

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
            self.children[playerId] = oneRhoLearnExp3(maxRank, rankSelectionAlgo, change_rank_each_step, feedback_function, self, playerId)
        # Fake rankSelection algorithm, for pretty print
        self._rankSelection = rankSelectionAlgo(maxRank)

    def __str__(self):
        return "rhoLearnExp3({} x {}, ranks ~ {})".format(self.nbPlayers, str(self._players[0]), self._rankSelection)
