# :boom: TODO
## Initial things to do - OK
- [x] clean up code, OK
- [x] lint the code and make it "perfect", OK
- [x] pass it to Python 3.5 (while still being valid Python 2.7), OK
- [x] add more arms: Gaussian, Exponential, Poisson, OK
- [x] add my aggregated bandit algorithm, OK

## Improve the code
- [x] In fact, [exhaustive grid search](http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search) cannot be easily used as it cannot run *on-line*! Sadly, OK.
- [x] add plots that show the percentage of optimal arms play ([e.g., as done in this paper](http://www.cs.mcgill.ca/~vkules/bandits.pdf#page=11))
- [x] fully profile my code, with [`cProfile`](https://docs.python.org/2/library/profile.html) for functions and [`line_profiler`](https://github.com/rkern/line_profiler) for line-by-line. No surprise here: [`Beta.py`](Policies/Beta.py) is the slow part, as it takes time to sample and compute the quantiles (even by using the good `numpy.random`/`scipy.stats` functions). See for instance [this log file (with `cProfile`)](logs/main_py3_profile_log.txt) or [this one (with `line_profiler`)](logs/main_py3_line_profiler_log.txt).
- [ ] I could have tried to improve the bottlenecks, with smart `numpy`/`scipy` code, or [`numba` ?](http://numba.pydata.org/), or [`cython`](http://cython.org/) code ? Not so easy, not so interesting...
- [ ] explore the behavior of my Aggr algorithm, and understand it better (and improve it?)

## Better storing of the simulation results
- [ ] use [hdf5](https://www.hdfgroup.org/HDF5/) (with [`h5py`](http://docs.h5py.org/en/latest/quick.html#core-concepts)) to store the data, on the run (to never lose data, even if the simulation gets killed)

## Publicly release it ?
- [x] keep it up-to-date [on GitHub](https://github.com/Naereen/AlgoBandits)
- [x] I could document this project better. But, well, there is no [Sphinx](http://sphinx-doc.org/) documentation yet, but each file has a docstring, some useful comments for the interesting part, and this very page you are reading contains [insights on how to use the framework](#configuration) as well as [the organization of the code](#code-organization).

## More MAB algorithms
- [ ] implement some more algorithms, e.g., from [this repository](https://github.com/johnmyleswhite/BanditsBook/blob/master/python/algorithms/exp3/exp3.py)
- [ ] add more basic algorithms, e.g., from [this survey](http://homes.di.unimi.it/~cesabian/Pubblicazioni/banditSurvey.pdf) or [this document](http://www.cs.mcgill.ca/~vkules/bandits.pdf)

## Multi-players simulations
- [ ] implement a multi-player simulation environment as well!
- [ ] implement the [`rho_rand`](http://ieeexplore.ieee.org/document/5462144/), [`TDFS`](https://arxiv.org/abs/0910.2065v3), [`MEGA`](https://arxiv.org/abs/1404.5421), [`Musical Chair` and `Dynamic Musical Chair`](https://arxiv.org/abs/1512.02866) multi-player algorithms
