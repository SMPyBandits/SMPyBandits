from Policies import *

policy = UCB(10)
print(policy)

def choice():
    result = policy.choice()
    return result

def getReward(arm, reward):
    result = policy.getReward(arm, reward)
    return result
