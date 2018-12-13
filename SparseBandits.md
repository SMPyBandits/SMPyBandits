# **Structure and Sparsity of Stochastic Multi-Armed Bandits**

This page explains shortly what I studied on sparse stochastic multi-armed bandits.
Assume a MAB problem with `$K$` arms, each parametrized by its *mean* `$\mu_k\in\mathbb{R}$`.
If you know in advance that only a small subset (of size `$s$`) of the arms have a positive arm, it sounds reasonable to hope to be more efficient in playing the bandit game compared to an approach which is non aware of the sparsity.

The [`SparseUCB`](docs/Policies.SparseUCB.html#Policies.SparseUCB.SparseUCB) is an extension of the well-known [`UCB`](docs/Policies.UCB.html), and requires to known **exactly** the value of `$s$`.
It works by identifying as fast as possible (actually, in a sub-logarithmic number of samples) the arms with non-positive means.
Then it only plays in the "good" arms with positive means, with a regular UCB policy.

I studied extensions of this idea, first of all the [`SparseklUCB`](docs/Policies.SparseklUCB.html#Policies.SparseklUCB.SparseklUCB) policy as it was suggested in the original research paper, but mainly a generic "wrapper" black-box approach.
For more details, see [`SparseWrapper`](docs/Policies.SparseWrapper.html#Policies.SparseWrapper.SparseWrapper).

- Reference: [["Sparse Stochastic Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)]. Note that this algorithm only works for sparse [Gaussian]((docs/Arms.Gaussian.html)) (or sub-Gaussian) stochastic bandits, and it includes [Bernoulli arms](docs/Arms.Bernoulli.html).

----

## Article

> TODO finish! I am writing a small research article on that topic, it is a better introduction as a self-contained document to explain this idea and the algorithms. Reference: [[Structure and Sparsity of Stochastic Multi-Arm Bandits, Lilian Besson and Emilie Kaufmann, 2018]](https://hal.inria.fr/hal-XXX).

----

## Example of simulation configuration

A simple python file, [`configuration_sparse.py`](https://smpybandits.github.io/docs/configuration_sparse.html), is used to import the [arm classes](Arms/), the [policy classes](Policies/) and define the problems and the experiments.

For example, we can compare the standard [`UCB`](docs/Policies.UCB.html) and  [`BayesUCB`](docs/Policies.BayesUCB.html) algorithms, non aware of the sparsity, against the sparsity-aware [`SparseUCB`](docs/Policies.SparseUCB.html) algorithm, as well as 4 versions of [`SparseWrapper`](docs/Policies.SparseWrapper.html) applied to [`BayesUCB`](docs/Policies.BayesUCB.html).

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

Then add a [Sparse-Wrapper](docs/Policies.SparseWrapper.html) bandit algorithm ([`SparseWrapper` class](docs/Policies.SparseWrapper.html)), you can use this piece of code:

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

Here are some plots illustrating the performances of the different [policies](docs/Policies.) implemented in this project, against various sparse problems (with [`Bernoulli`](Arms/Bernoulli.html) or [`UnboundedGaussian`](https://smpybandits.github.io/docs/Arms.Gaussian.html) arms only):

### 3 variants of [Sparse-Wrapper](docs/Policies.SparseWrapper.html) for UCB, on a "simple" sparse Bernoulli problem
![3 variants of Sparse-Wrapper for UCB, on a "simple" sparse Bernoulli problem](plots/main____env1-1_XXX.png)

FIXME run some simulations and explain them!

> These illustrations come from my (work in progress) article, [[Structure and Sparsity of Stochastic Multi-Arm Bandits, Lilian Besson and Emilie Kaufmann, 2018]](https://hal.inria.fr/hal-XXX).


----

### :scroll: License ? [![GitHub license](https://img.shields.io/github/license/SMPyBandits/SMPyBandits.svg)](https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/SMPyBandits/SMPyBandits/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel)](https://GitHub.com/SMPyBandits/SMPyBandits/)
![![PyPI version](https://img.shields.io/pypi/v/smpybandits.svg)](https://pypi.org/project/SMPyBandits)
![![PyPI implementation](https://img.shields.io/pypi/implementation/smpybandits.svg)](https://pypi.org/project/SMPyBandits)
[![![PyPI pyversions](https://img.shields.io/pypi/pyversions/smpybandits.svg?logo=python)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![![PyPI download](https://img.shields.io/pypi/dm/smpybandits.svg)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![![PyPI status](https://img.shields.io/pypi/status/smpybandits.svg)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![Documentation Status](https://readthedocs.org/projects/smpybandits/badge/?version=latest)](https://SMPyBandits.ReadTheDocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/SMPyBandits/SMPyBandits.svg?branch=master)](https://travis-ci.org/SMPyBandits/SMPyBandits)
[![Stars of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/stars/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/stargazers)
[![Releases of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/release/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/releases)
