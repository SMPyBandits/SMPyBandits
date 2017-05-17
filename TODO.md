# :boom: TODO
> For others things to do, and issues to solve, see [the issue tracker on GitHub](https://github.com/Naereen/AlgoBandits/issues).

## Initial things to do! - OK
- [x] Clean up initial code, keep it clean and commented, OK.
- [x] Lint the code and make it (almost) "perfect" regarding [Python style recommandation](https://www.python.org/dev/peps/pep-0008/), OK.
- [x] Pass it to Python 3.4 (while still being valid Python 2.7), OK. It is valid Python, both v2 (2.7), and v3 (3.4, 3.5, 3.6).
- [x] Add more arms: [Gaussian](Arms/Gaussian.py), [Exponential](Arms/Exponential.py), [Poisson](Arms/Poisson.py), [Uniform](Arms/Uniform.py), OK.
- [x] Add my [aggregated bandit algorithm](Policies/Aggr.py), OK. FIXME finish to work on it!

## Improve the code? - OK
- [x] In fact, [exhaustive grid search](http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search) cannot be easily used as it cannot run *on-line*! Sadly, OK.
- [x] add plots that show the percentage of optimal arms play ([e.g., as done in this paper](http://www.cs.mcgill.ca/~vkules/bandits.pdf#page=11))
- [x] fully profile my code, with [`cProfile`](https://docs.python.org/2/library/profile.html) for functions and [`line_profiler`](https://github.com/rkern/line_profiler) for line-by-line. No surprise here: [`Beta.py`](Policies/Beta.py) is the slow part, as it takes time to sample and compute the quantiles (even by using the good `numpy.random`/`scipy.stats` functions). See for instance [this log file (with `cProfile`)](logs/main_py3_profile_log.txt) or [this one (with `line_profiler`)](logs/main_py3_line_profiler_log.txt).
- [x] I could have tried to improve the efficiency bottlenecks, with smart `numpy`/`scipy` code (I did not find anything useful), or [`numba.jit`](http://numba.pydata.org/) ? (does not seem to be possible), or [`cython`](http://cython.org/) code (not so easy, not so interesting)... Maybe using [`numexpr`](https://github.com/pydata/numexpr/wiki/Numexpr-Users-Guide): nope! Maybe using [`glumpy`](http://glumpy.readthedocs.io/en/latest/) for faster vizualisations: nope! Using [pypy](http://pypy.org/compat.html) is impossible as it does not support all of `numpy` yet, and none of `matplotlib`.
- [x] Explore the behavior of my Aggr algorithm, and understand it better (and improve it? it would only be by parameter tweaking, not interesting, so NOPE).
- [x] Rewards not in `{0, 1}` are handled with a trick, with a "random binarization", cf., [[Agrawal & Goyal, 2012]](http://jmlr.org/proceedings/papers/v23/agrawal12/agrawal12.pdf) (algorithm 2): when reward `r_t \in [0, 1]` is observed, the player receives the result of a [Bernoulli](Arms/Bernoulli.py) sample of average `r_t`: `r_t <- sample from Bernoulli(r_t)` so it is well in `{0, 1}`. Works fine for `Exponential` arms, for instance.
- [x] Test again (and adapt, if needed) each single-player policy against non-[Bernoulli](Arms/Bernoulli.py) arms ([Gaussian](Arms/Gaussian.py), [Exponential](Arms/Exponential.py), [Poisson](Arms/Poisson.py)).
- [x] My framework can also handle rewards which are not bounded in `[0, 1]`, but can not handle unbounded rewards (eg. non-truncated Gaussian or Poisson), *yet*.

## More single-player MAB algorithms? - OK
- [x] I implemented two variants of the [`KL-UCB`](Policies/klUCB.py) policy: [`KL-UCB-Plus`](Policies/klUCBPlus.py) from [[Cappé et al., 2013]](https://arxiv.org/pdf/1210.1136.pdf), [`KL-UCB-H-Plus`](Policies/klUCBHPlus.py) from [[Lai, 1987]](https://projecteuclid.org/download/pdf_1/euclid.aos/1176350495), and [`KL-UCB-++`](Policies/klUCBPlusPlus.py) from [[Ménard & Garivier, 2017]](https://arxiv.org/pdf/1702.07211.pdf).
- [x] I implemented all the algorithms from [this document](http://www.cs.mcgill.ca/~vkules/bandits.pdf), and from [this repository](https://github.com/johnmyleswhite/BanditsBook/blob/master/python/algorithms/exp3/exp3.py) ([`Exp3`](Policies/Exp3.py)).
- [x] I don't see any other simple bandit algorithm I could implement, I wrote all the one described in [this reference survey [Bubeck & Cesa-Bianchi, 2012]](http://homes.di.unimi.it/~cesabian/Pubblicazioni/banditSurvey.pdf).

-----

## Publicly release it and document it - OK
- [x] keep it up-to-date [on GitHub](https://github.com/Naereen/AlgoBandits)
- [x] I could document this project better. Now there is a [Sphinx](http://sphinx-doc.org/) documentation (`make -B apidoc doc` then open `_build/html/index.html`). Each file has a nice docstring, some useful comments for the interesting parts, and the GitHub project contains [insights on how to use the framework](README.md#configuration) as well as [the organization of the code](README.md#code-organization). I added the [`API.md`](API.md) file to document the arms and policies API.

## Better storing of the simulation results
- [ ] use [hdf5](https://www.hdfgroup.org/HDF5/) (with [`h5py`](http://docs.h5py.org/en/latest/quick.html#core-concepts)) to store the data, *on the run* (to never lose data, even if the simulation gets killed).
- [ ] even more "secure": be able to *interrupt* the simulation, *save* its state and then *load* it back if needed (for instance if you want to leave the office for the weekend).

-----

## Multi-players simulations!
- [x] implement a multi-player simulation environment as well! Done, in [EvaluatorMultiPlayers](Environment/EvaluatorMultiPlayers.py). TODO Keep improving it!
- [x] implement [different collision models](Environment/CollisionModels.py) (4 different models as far as now), and try it on each, with different setting (K < M, M = K, M < K, static or dynamic, Bernoulli or non-Bernoulli arms etc).
- [x] implement the basic multi-player policies: the fully decentralized [`Selfish`](PoliciesMultiPlayers/Selfish.py) policy (where every player runs her own policy, without even knowing that there is other player, but receiving `0` reward in case of collision), two stupid centralized policies [`CentralizedFixed`](PoliciesMultiPlayers/CentralizedFixed.py) and [`CentralizedCycling`](PoliciesMultiPlayers/CentralizedCycling.py), and two oracle policies [`OracleNotFair`](PoliciesMultiPlayers/OracleNotFair.py) and [`OracleFair`](PoliciesMultiPlayers/OracleFair.py).
- [x] implement a centralized non-oracle policy, which is just a multiple-play single-player policy, in [`CentralizedMultiplePlay`](PoliciesMultiPlayers/CentralizedMultiplePlay.py). The single-player policy uses the `choiceMultiple(nb)` method to chose directly `M` arms for the `M` players. It works very well: no collision, and very fast identification of the best `M` arms!
- [x] plot several "multi-players" policy on the same graphs (e.g., the cumulative centralized regret of `M` players following `Selfish[UCB]` against the regret of `M` players following `Selfish[klCUB]`, or `ALOHA` vs `rhoRand` vs `MusicalChair`). TODO Keep improving it!

### State-of-the-art algorithms
- [x] I implemented the ["Musical Chair"](https://arxiv.org/abs/1512.02866) policy, from [[Shamir et al., 2015]](https://arxiv.org/abs/1512.02866), in [`MusicalChair`](Policies/MusicalChair.py). FIXME the ["Dynamic Musical Chair"](https://arxiv.org/abs/1512.02866) that regularly reinitialize "Musical Chair"...
- [x] I implemented the ["rho_rand"](http://ieeexplore.ieee.org/document/5462144/) from [[Anandkumar et al., 2009]](http://ieeexplore.ieee.org/document/5462144/), in [`rhoRand`](PoliciesMultiPlayers/rhoRand.py). It consists of the rho_rand collision avoidance protocol, and *any* single-player policy. FIXME the `rhoEst` policy that estimate the number of users from collision.
- [x] I implemented the ["MEGA"](https://arxiv.org/abs/1404.5421) multi-player policy from [[Avner & Mannor, 2014]](https://arxiv.org/abs/1404.5421), in [`MEGA`](Policies/MEGA.py). It consists of the ALOHA collision avoidance protocol and a Epsilon-greedy arm selection algorithm.
- [x] I also generalized it by implementing the ALOHA collision avoidance protocol for *any* single-player policy, in [`ALOHA`](PoliciesMultiPlayers/ALOHA.py). FIXME it has too much parameter, how to chose them??
- [ ] TODO ["TDFS"](https://arxiv.org/abs/0910.2065v3) from [[Liu & Zhao, 2009]](https://arxiv.org/abs/0910.2065v3).

### Dynamic settings
- [ ] add the possibility to have a varying number of dynamic users for multi-users simulations in AlgoBandits.git
- [ ] implement the experiments from [Musical Chair], [rhoRand] articles, and Navik Modi's experiments
