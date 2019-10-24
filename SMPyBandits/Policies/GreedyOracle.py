from .IndexPolicy import IndexPolicy
import numpy as np

class GreedyPolicy(IndexPolicy):
    """ Greedy Oracle for rotting bandits.
    Selects arm with best last value.
    Reference: [Heidari , Kearns & Roth, 2016]
    """
    def __init__(self, nbArms):
        super(GreedyPolicy, self).__init__(nbArms)
        self.arms_history = [np.array([]) for arm in range(nbArms)]

    def getReward(self, arm, reward):
        super(GreedyOracle, self).getReward(arm, reward)
        self.arms_history[arm] = np.insert(self.arms_history[arm], 0, 0) + reward

    def computeAllIndex(self,arms):
        """ Compute the mean of the h last value """
        for i, arm in enumerate(arms):
            self.index[i] = self.arms_history[arm][0]

class GreedyOracle(IndexPolicy):
    """ Greedy Oracle for rotting bandits.
        Look 1 step forward and select next best value.
        Optimal policy for rotting bandits problem.
    Reference: [Heidari , Kearns & Roth, 2016]
    """
    def __init__(self,nbArms, arms):
        super(GreedyOracle, self).__init__(nbArms)
        self.arms = arms

    def computeIndex(self, arm):
        return self.arms[arm].mean

