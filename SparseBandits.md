# **Structure and Sparsity of Stochastic Multi-Armed Bandits**

TODO explain!

----

## Article

I am writing a small research article on that topic, it is a better introduction as a self-contained document to explain this idea and the algorithms. Reference: [[Structure and Sparsity of Stochastic Multi-Arm Bandits, Lilian Besson and Emilie Kaufmann, 2018]](https://hal.inria.fr/hal-XXX).

----

## Example of simulation configuration

A simple python file, [`configuration_sparse.py`](SMPyBandits/configuration_sparse.py), is used to import the [arm classes](Arms/), the [policy classes](Policies/) and define the problems and the experiments.

For example, we can compare the standard [`UCB`](SMPyBandits/Policies/UCB.py) and  [`BayesUCB`](SMPyBandits/Policies/BayesUCB.py) algorithms, non aware of the sparsity, against the sparsity-aware [`SparseUCB`](SMPyBandits/Policies/SparseUCB.py) algorithm, as well as 4 versions of [`SparseWrapper`](SMPyBandits/Policies/SparseWrapper.py) applied to [`BayesUCB`](SMPyBandits/Policies/BayesUCB.py).

```python
configuration = {
    "horizon": 10000,    # Finite horizon of the simulation
    "repetitions": 100,  # number of repetitions
    "n_jobs": -1,        # Maximum number of cores for parallelization: use ALL your CPU
    "verbosity": 5,      # Verbosity for the joblib calls
    # Environment configuration, you can set up more than one.
    "environment": [
        {   # sparsity = nb of >= 0 mean, = 3 here
            "arm_type": Bernoulli,
            "params": 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3
        }
    ],
    # Policies that should be simulated, and their parameters.
    "policies": [
        {"archtype": UCB, "params": {} },
        {"archtype": SparseUCB, "params": { "sparsity": 3 } },
        {"archtype": BayesUCB, "params": { } },
    ]
}
```

Then add a [Sparse-Wrapper](SMPyBandits/Policies/SparseWrapper.py) bandit algorithm ([`SparseWrapper` class](SMPyBandits/Policies/SparseWrapper.py)), you can use this piece of code:

```python
configuration["policies"] += [
    {
        "archtype": SparseWrapper,
        "params": {
            "policy": BayesUCB,
            "use_ucb_for_set_J": use_ucb_for_set_J,
            "use_ucb_for_set_K": use_ucb_for_set_K,
        }
    }
    for use_ucb_for_set_J in [ True, False ]
    for use_ucb_for_set_K in [ True, False ]
]
```

----

## [How to run the experiments ?](How_to_run_the_code.md)

You should use the provided [`Makefile`](Makefile) file to do this simply:
```bash
make install  # install the requirements ONLY ONCE
make sparse   # run and log the main.py script
```

----

## Some illustrations

Here are some plots illustrating the performances of the different [policies](SMPyBandits/Policies/) implemented in this project, against various sparse problems (with [`Bernoulli`](Arms/Bernoulli.py) or [`UnboundedGaussian`](SMPyBandits/Arms/Gaussian.py) arms only):

### 3 variants of [Sparse-Wrapper](SMPyBandits/Policies/SparseWrapper.py) for UCB, on a "simple" sparse Bernoulli problem
![3 variants of Sparse-Wrapper for UCB, on a "simple" sparse Bernoulli problem](plots/main____env1-1_XXX.png)

FIXME run some simulations and explain them!

> These illustrations come from my (work in progress) article, [[Structure and Sparsity of Stochastic Multi-Arm Bandits, Lilian Besson and Emilie Kaufmann, 2018]](https://hal.inria.fr/hal-XXX).


----

### :scroll: License ? [![GitHub license](https://img.shields.io/github/license/SMPyBandits/SMPyBandits.svg)](https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel)](https://GitHub.com/SMPyBandits/SMPyBandits/)
![PyPI version](https://img.shields.io/pypi/v/smpybandits.svg)
![PyPI implementation](https://img.shields.io/pypi/implementation/SMPyBandits.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/SMPyBandits.svg)
[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)
