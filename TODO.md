# :boom: TODO
## Initial things to do - OK
- [x] clean up initial code, keep it clean and commented, OK.
- [x] lint the code and make it (almost) "perfect" regarding [Python style recommandation](https://www.python.org/dev/peps/pep-0008/), OK.
- [x] pass it to Python 3.4 (while still being valid Python 2.7), OK. It is valid Python 2.7, 3.4, 3.5, 3.6.
- [x] add more arms: Gaussian, Exponential, Poisson, OK.
- [x] add my aggregated bandit algorithm, OK.

## Improve the code - almost OK
- [x] In fact, [exhaustive grid search](http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search) cannot be easily used as it cannot run *on-line*! Sadly, OK.
- [x] add plots that show the percentage of optimal arms play ([e.g., as done in this paper](http://www.cs.mcgill.ca/~vkules/bandits.pdf#page=11))
- [x] fully profile my code, with [`cProfile`](https://docs.python.org/2/library/profile.html) for functions and [`line_profiler`](https://github.com/rkern/line_profiler) for line-by-line. No surprise here: [`Beta.py`](Policies/Beta.py) is the slow part, as it takes time to sample and compute the quantiles (even by using the good `numpy.random`/`scipy.stats` functions). See for instance [this log file (with `cProfile`)](logs/main_py3_profile_log.txt) or [this one (with `line_profiler`)](logs/main_py3_line_profiler_log.txt).
- [ ] I could have tried to improve the bottlenecks, with smart `numpy`/`scipy` code (I did not find anything useful), or [`numba` ?](http://numba.pydata.org/) (does not seem to be possible), or [`cython`](http://cython.org/) code (not so easy, not so interesting)...
- [x] explore the behavior of my Aggr algorithm, and understand it better (and improve it? it would only be by parameter tweaking, not interesting, so NOPE).
- [ ] test again (and adapt, if needed) each single-player policy against non-Bernoulli arms (Gaussian, Exponential, Poisson). FIXME how to handle rewards which are NOT in `[0, 1]` ?

## Better storing of the simulation results
- [ ] use [hdf5](https://www.hdfgroup.org/HDF5/) (with [`h5py`](http://docs.h5py.org/en/latest/quick.html#core-concepts)) to store the data, on the run (to never lose data, even if the simulation gets killed)

## Publicly release it ?
- [x] keep it up-to-date [on GitHub](https://github.com/Naereen/AlgoBandits)
- [x] I could document this project better. Well, there is no [Sphinx](http://sphinx-doc.org/) documentation yet, but each file has a docstring, some useful comments for the interesting part, and this very page you are reading contains [insights on how to use the framework](#configuration) as well as [the organization of the code](#code-organization).

## More MAB algorithms (EXP-3 etc)
- [ ] implement some more algorithms, e.g., from [this repository](https://github.com/johnmyleswhite/BanditsBook/blob/master/python/algorithms/exp3/exp3.py) (EXP3), or [this survey](http://homes.di.unimi.it/~cesabian/Pubblicazioni/banditSurvey.pdf), or [this document](http://www.cs.mcgill.ca/~vkules/bandits.pdf)

## Multi-players simulations
- [x] implement a multi-player simulation environment as well! Done, in [EvaluatorMultiPlayers](Environment/EvaluatorMultiPlayers.py).
- [x] implement [different collision models](Environment/CollisionModels.py) (4 different models as far as now), and try it on each, with different setting (K < M, M = K, M < K, static or dynamic, Bernoulli or non-Bernoulli arms).
- [x] implement the basic multi-player policies, as [`Selfish`](PoliciesMultiPlayers/Selfish.py), [`CentralizedNotFair`](PoliciesMultiPlayers/CentralizedNotFair.py), [`CentralizedFair`](PoliciesMultiPlayers/CentralizedFair.py), [`OracleNotFair`](PoliciesMultiPlayers/OracleNotFair.py), [`OracleFair`](PoliciesMultiPlayers/OracleFair.py).
- [x] I implemented the ["Musical Chair"](https://arxiv.org/abs/1512.02866) policy, from [[Shamir et al., 2015]](https://arxiv.org/abs/0910.2065v3), in [`MusicalChair`](Policies/MusicalChair.py). TODO ["Dynamic Musical Chair"](https://arxiv.org/abs/1512.02866).
- [ ] implement the state-of-the-art algorithms: ["rho_rand"](http://ieeexplore.ieee.org/document/5462144/) from [[Anandkumar et al., 2009]](http://ieeexplore.ieee.org/document/5462144/), ["TDFS"](https://arxiv.org/abs/0910.2065v3) from [[Liu & Zhao, 2009]](https://arxiv.org/abs/0910.2065v3), ["MEGA"](https://arxiv.org/abs/1404.5421) from [[Avner & Mannor, 2014]](https://arxiv.org/abs/1404.5421).
