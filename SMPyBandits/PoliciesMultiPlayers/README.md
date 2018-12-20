# [Multi-Player policies](https://smpybandits.github.io/docs/PoliciesMultiPlayers.html)
> See here the documentation: [docs/PoliciesMultiPlayers](https://smpybandits.github.io/docs/PoliciesMultiPlayers.html)


## List of Policies
`PoliciesMultiPlayers` folder : contains various collision-avoidance protocol for the multi-players setting.

- [`Selfish`](Selfish.py): a multi-player policy where every player is selfish, they do not try to handle the collisions.

- [`CentralizedNotFair`](CentralizedNotFair.py): a multi-player policy which uses a centralize intelligence to affect users to a FIXED arm.
- [`CentralizedFair`](CentralizedFair.py): a multi-player policy which uses a centralize intelligence to affect users an offset, each one take an orthogonal arm based on (offset + t) % nbArms.

- [`CentralizedMultiplePlay`](CentralizedMultiplePlay.py) and [`CentralizedIMP`](CentralizedIMP.py): multi-player policies that use centralized but non-omniscient learning to select K = nbPlayers arms at each time step.

- [`OracleNotFair`](OracleNotFair.py): a multi-player policy with full knowledge and centralized intelligence to affect users to a FIXED arm, among the best arms.
- [`OracleFair`](OracleFair.py): a multi-player policy which uses a centralized intelligence to affect users an offset, each one take an orthogonal arm based on (offset + t) % nbBestArms, among the best arms.

- [`rhoRand`](rhoRand.py), [`ALOHA`](ALOHA.py): implementation of generic collision avoidance algorithms, relying on a single-player bandit policy (eg. [`UCB`](UCB.py), [`Thompson`](Thompson.py) etc). And variants, [`rhoRandRand`](rhoRandRand.py), [`rhoRandSticky`](rhoRandSticky.py), [`rhoRandRotating`](rhoRandRotating.py), [`rhoRandEst`](rhoRandEst.py), [`rhoLearn`](rhoLearn.py), [`rhoLearnEst`](rhoLearnEst.py), [`rhoLearnExp3`](rhoLearnExp3.py), [`rhoRandALOHA`](rhoRandALOHA.py),
- [`rhoCentralized`](rhoCentralized.py) is a semi-centralized version where orthogonal ranks 1..M are given to the players, instead of just giving them the value of M, but a decentralized learning policy is still used to learn the best arms.
- [`RandTopM`](RandTopM.py) is another approach, similar to [`rhoRandSticky`](rhoRandSticky.py) and [`MusicalChair`](MusicalChair.py), but we hope it will be better, and we succeed in analyzing more easily.

## API
All policies have the same interface, as described in [`BaseMPPolicy`](BaseMPPolicy.py) for decentralized policies,
and [`BaseCentralizedPolicy`](BaseCentralizedPolicy.py) for centralized policies,
in order to use them in any experiment with the following approach:

```python
my_policy_MP = Policy_MP(nbPlayers, nbArms)
children = my_policy_MP.children             # get a list of usable single-player policies
for one_policy in children:
    one_policy.startGame()                       # start the game
for t in range(T):
    for i in range(nbPlayers):
        k_t[i] = children[i].choice()            # chose one arm, for each player
    for k in range(nbArms):
        players_who_played_k = [ k_t[i] for i in range(nbPlayers) if k_t[i] == k ]
        reward = reward_t[k] = sampled from the arm k     # sample a reward
        if len(players_who_played_k) > 1:
            reward = 0
        for i in players_who_played_k:
            children[i].getReward(k, reward)
```