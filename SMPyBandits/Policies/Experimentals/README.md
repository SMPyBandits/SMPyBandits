# [Single-Player policies](https://smpybandits.github.io/docs/Policies.Experimentals.html)
> See here the documentation: [docs/Policies.Experimentals](https://smpybandits.github.io/docs/Policies.Experimentals.html)

## List of experimental policies
``Policies.Experimentals.Experimentals`` module : contains experimental or unfinished (single-player) bandits algorithms:

- Index based UCB algorithms: [`UCBlog10`](UCBlog10.py), [`UCBwrong`](UCBwrong.py), [`UCBlog10alpha`](UCBlog10alpha.py), [`UCBcython`](UCBcython.py), [`UCBjulia`](UCBjulia.py) (with [`UCBjulia.jl`](UCBjulia.jl)),

- Based on Kullback-Leibler divergence: [`klUCBlog10`](klUCBlog10.py), [`klUCBloglog10`](klUCBloglog10.py),

- Empirical KL-UCB algorithm: [`KLempUCB`](KLempUCB.py) (does not work with the C optimized version of [`kullback`](kullback.py),

- An *experimental* policy, using Unsupervised Learning: [`UnsupervisedLearning`](UnsupervisedLearning.py),

- An *experimental* policy, using Black-box optimization: [`BlackBoxOpt`](BlackBoxOpt.py),

- Bayesian algorithms: [`ThompsonRobust`](ThompsonRobust.py),

- **New!** The UCBoost (Upper Confidence bounds with Boosting) policies, first with no boosting, in module [`UCBoost_faster`](UCBoost_faster.py): `UCBoost_faster.UCB_sq`, `UCBoost_faster.UCB_bq`, `UCBoost_faster.UCB_h`, `UCBoost_faster.UCB_lb`, `UCBoost_faster.UCB_t`, and then the ones with non-adaptive boosting: `UCBoost_faster.UCBoost_bq_h_lb`, `UCBoost_faster.UCBoost_bq_h_lb_t`, `UCBoost_faster.UCBoost_bq_h_lb_t_sq`, `UCBoost_faster.UCBoost`, and finally the epsilon-approximation boosting with `UCBoost_faster.UCBoostEpsilon`. These versions use Cython for some functions.

- **New!** The UCBoost (Upper Confidence bounds with Boosting) policies, first with no boosting, in module [`UCBoost_cython`](UCBoost_cython.py): `UCBoost_cython.UCB_sq`, `UCBoost_cython.UCB_bq`, `UCBoost_cython.UCB_h`, `UCBoost_cython.UCB_lb`, `UCBoost_cython.UCB_t`, and then the ones with non-adaptive boosting: `UCBoost_cython.UCBoost_bq_h_lb`, `UCBoost_cython.UCBoost_bq_h_lb_t`, `UCBoost_cython.UCBoost_bq_h_lb_t_sq`, `UCBoost_cython.UCBoost`, and finally the epsilon-approximation boosting with `UCBoost_cython.UCBoostEpsilon`. These versions use Cython for the whole code.


## API
All policies have the same interface, as described in [`BasePolicy`](../BasePolicy.py),
in order to use them in any experiment with the following approach:

```python
my_policy = Policy(nbArms)
my_policy.startGame()  # start the game
for t in range(T):
    chosen_arm_t = k_t = my_policy.choice()  # chose one arm
    reward_t     = sampled from an arm k_t   # sample a reward
    my_policy.getReward(k_t, reward_t)       # give it the the policy
```
