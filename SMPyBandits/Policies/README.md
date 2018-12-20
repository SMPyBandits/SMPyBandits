# [Single-Player policies](https://smpybandits.github.io/docs/Policies.html)
> See here the documentation: [docs/Policies](https://smpybandits.github.io/docs/Policies.html)

## List of policies
`Policies` module : contains various (single-player) bandits algorithms:

- "Stupid" algorithms: [`Uniform`](Uniform.py), [`UniformOnSome`](UniformOnSome.py), [`TakeFixedArm`](TakeFixedArm.py), [`TakeRandomFixedArm`](TakeRandomFixedArm.py),

- Greedy algorithms: [`EpsilonGreedy`](EpsilonGreedy.py), [`EpsilonFirst`](EpsilonFirst.py), [`EpsilonDecreasing`](EpsilonDecreasing.py),
- And two variants of the Explore-Then-Commit policy: [`ExploreThenCommit.ETC_KnownGa`p](ExploreThenCommit.py), [`ExploreThenCommit.ETC_RandomStop`](ExploreThenCommit.py),

- Probabilistic weighting algorithms: [`Hedge`](Hedge.py), [`Softmax`](Softmax.py), [`Softmax.SoftmaxDecreasing`](Softmax.py), [`Softmax.SoftMix`](Softmax.py), [`Softmax.SoftmaxWithHorizon`](Softmax.py), [`Exp3`](Exp3.py), [`Exp3.Exp3Decreasin`g](Exp3.py), [`Exp3.Exp3SoftMix`](Exp3.py), [`Exp3.Exp3WithHorizon`](Exp3.py), [`Exp3.Exp3ELM`](Exp3.py), [`ProbabilityPursuit`](ProbabilityPursuit.py), [`Exp3PlusPlus`](Exp3PlusPlus.py), and a smart variant [`BoltzmannGumbel`](BoltzmannGumbel.py),

- Index based UCB algorithms: [`EmpiricalMeans`](EmpiricalMeans.py), [`UCB`](UCB.py), [`UCBlog10`](UCBlog10.py), [`UCBwrong`](UCBwrong.py), [`UCBlog10alpha`](UCBlog10alpha.py), [`UCBalpha`](UCBalpha.py), [`UCBmin`](UCBmin.py), [`UCBplus`](UCBplus.py), [`UCBrandomInit`](UCBrandomInit.py), [`UCBV`](UCBV.py), [`UCBVtuned`](UCBVtuned.py), [`UCBH`](UCBH.py), [`CPUCB`](CPUCB.py),

- Index based MOSS algorithms: [`MOSS`](MOSS.py), [`MOSSH`](MOSSH.py), [`MOSSAnytime`](MOSSAnytime.py), [`MOSSExperimental`](MOSSExperimental.py),

- Bayesian algorithms: [`Thompson`](Thompson.py), [`ThompsonRobust`](ThompsonRobust.py), [`BayesUCB`](BayesUCB.py),

- Based on Kullback-Leibler divergence: [`klUCB`](klUCB.py), [`klUCBlog10`](klUCBlog10.py), [`klUCBloglog`](klUCBloglog.py), [`klUCBloglog10`](klUCBloglog10.py), [`klUCBPlus`](klUCBPlus.py), [`klUCBH`](klUCBH.py), [`klUCBHPlus`](klUCBHPlus.py), [`klUCBPlusPlus`](klUCBPlusPlus.py),

- Empirical KL-UCB algorithm: [`KLempUCB`](KLempUCB.py) (FIXME),

- Other index algorithms: [`DMED`](DMED.py), [`DMED.DMEDPlus`](DMED.py), [`OCUCB`](OCUCB.py), [`UCBdagger`](UCBdagger.py),

- Hybrids algorithms, mixing Bayesian and UCB indexes: [`AdBandits`](AdBandits.py),

- Aggregation algorithms: [`Aggregator`](Aggregator.py) (mine, it's awesome, go on try it!), and [`CORRAL`](CORRAL.py), [`LearnExp`](LearnExp.py),

- Finite-Horizon Gittins index, approximated version: [`ApproximatedFHGittins`](ApproximatedFHGittins.py),

- An *experimental* policy, using Unsupervised Learning: [`UnsupervisedLearning`](UnsupervisedLearning.py),

- An *experimental* policy, using Black-box optimization: [`BlackBoxOpt`](BlackBoxOpt.py),

- An experimental policy, using a sliding window of for instance 100 draws, and reset the algorithm as soon as the small empirical average is too far away from the full history empirical average (or just restart for one arm, if possible), [`SlidingWindowRestart`](SlidingWindowRestart.py), and 3 versions for UCB, UCBalpha and klUCB: [`SlidingWindowRestart.SWR_UCB`](SlidingWindowRestart.py), [`SlidingWindowRestart.SWR_UCBalpha`](SlidingWindowRestart.py), [`SlidingWindowRestart.SWR_klUCB`](SlidingWindowRestart.py) (my algorithm, unpublished yet),

- An experimental policy, using just a sliding window of for instance 100 draws, [`SlidingWindowUCB.SWUCB`](SlidingWindowUCB.py), and [`SlidingWindowUCB.SWUCBPlu`s](SlidingWindowUCB.py) if the horizon is known.

- Another experimental policy with a discount factor, [`DiscountedUCB`](DiscountedUCB.py) and [`DiscountedUCB.DiscountedUCBPlus`](DiscountedUCB.py).

- A policy designed to tackle sparse stochastic bandit problems, [`SparseUCB`](SparseUCB.py), [`SparseklUCB`](SparseklUCB.py), and [`SparseWrapper`](SparseWrapper.py) that can be used with *any* index policy.

- A policy that implements a "smart doubling trick" to turn any horizon-dependent policy into a horizon-independent policy without loosing in performances: [`DoublingTrickWrapper`](DoublingTrickWrapper.py),

- An *experimental* policy, implementing a another kind of doubling trick to turn any policy that needs to know the range `[a,b]` of rewards a policy that don't need to know the range, and that adapt dynamically from the new observations, [`WrapRange`](WrapRange.py),

- The *Optimal Sampling for Structured Bandits* (OSSB) policy: [`OSSB`](OSSB.py) (it is more generic and can be applied to almost any kind of bandit problem, it works fine for classical stationary bandits but it is not optimal),

- **New!** The Best Empirical Sampled Average (BESA) policy: [`BESA`](BESA.py) (it works crazily well),

- Some are designed only for (fully decentralized) multi-player games: [`MusicalChair`](MusicalChair.py), [`MEGA`](MEGA.py).


## API
All policies have the same interface, as described in [`BasePolicy`](BasePolicy.py),
in order to use them in any experiment with the following approach:

```python
my_policy = Policy(nbArms)
my_policy.startGame()  # start the game
for t in range(T):
    chosen_arm_t = k_t = my_policy.choice()  # chose one arm
    reward_t     = sampled from an arm k_t   # sample a reward
    my_policy.getReward(k_t, reward_t)       # give it the the policy
```