"""
author: Julien Seznec

Oracle and near-minimax policy for rotting bandits without noise.

Reference: [Heidari et al., 2016, https://www.ijcai.org/Proceedings/16/Papers/224.pdf]
Tight Policy Regret Bounds for Improving and Decaying Bandits.
Hoda Heidari, Michael Kearns, Aaron Roth.
International Joint Conference on Artificial Intelligence (IJCAI) 2016, 1562.
"""
from .IndexPolicy import IndexPolicy
import numpy as np

class GreedyPolicy(IndexPolicy):
    """
    Greedy Policy for rotting bandits (A2 in the reference below).
    Selects arm with best last value.
    Reference: [Heidari et al., 2016, https://www.ijcai.org/Proceedings/16/Papers/224.pdf]
    """
    def __init__(self, nbArms):
        super(GreedyPolicy, self).__init__(nbArms)
        self.last_pull = [np.inf for _ in range(nbArms)]

    def getReward(self, arm, reward):
        super(GreedyPolicy, self).getReward(arm, reward)
        self.last_pull[arm] = reward

    def computeAllIndex(self):
        return self.last_pull

    def computeIndex(self,arm):
        """ Compute the mean of the h last value """
        return self.last_pull[arm]

    def startGame(self):
        super(GreedyPolicy, self).startGame()
        self.last_pull = [np.inf for _ in self.last_pull]


class GreedyOracle(IndexPolicy):
    """
    Greedy Oracle for rotting bandits (A0 in the reference below).
    Look 1 step forward and select next best value.
    Optimal policy for rotting bandits problem.
    Reference: [Heidari et al., 2016, https://www.ijcai.org/Proceedings/16/Papers/224.pdf]
    """
    def __init__(self,nbArms, arms):
        super(GreedyOracle, self).__init__(nbArms)
        self.arms = arms

    def computeIndex(self, arm):
        return self.arms[arm].mean

