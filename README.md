# Aggregated bandits
This repository contains the code for some numerical simulations on single-player [Multi-Armed Bandits (MAB)](https://en.wikipedia.org/wiki/Multi-armed_bandit) algorithms.

## The policy aggregation algorithm
Specifically, [I (Lilian Besson)](http://perso.crans.org/besson/) designed and added the [`Aggr`](Policies/Aggr.py) policy, in order to test it.

It is a simple **voting algorithm to combine multiple bandit algorithms into one**.
Basically, it behaves like the simple [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling), where arms are the child algorithms, each running in "parallel".

It can be seen as the Multi-Armed Bandits (i.e., sequential reinforcement learning) counterpart of an ensemble voting technique, as used for classifiers or regression algorithm in usual supervised machine learning (see, e.g., [`sklearn.ensemble.VotingClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier) in [scikit-learn](http://scikit-learn.org/)).

Another approach could be to do some sort of [grid search](http://scikit-learn.org/stable/modules/grid_search.html).

## Remarks
- [joblib](https://pythonhosted.org/joblib/) is used for the [`Evaluator`](Environment/Evaluator.py) class, so the simulations can easily be parallelized. (Put `n_jobs = -1` or `PARALLEL = True` to use all your CPU cores, as it is by default).
- Most of the code comes from the [pymabandits](http://mloss.org/software/view/415/) project, but some of them were refactored. Thanks to the initial project!

----

## Configuration:
A simple python file [`configuration.py`](configuration.py) is used.
For example:

```python
configuration = {
    # Finite horizon of the simulation
    "horizon": 10000,
    # number of repetitions
    "repetitions": 100,
    # Number of cores for parallelization
    "n_jobs": 4,
    # Verbosity for the joblib
    "verbosity": 5,
    # Environment configuration, yeah you can set up more than one.
    # I striped some code that were not published yet, but you can implement
    # your own arms.
    "environment": [
        {
            "arm_type": Bernoulli,
            "probabilities": [0.02, 0.02, 0.02, 0.10, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01]
        }
    ],
    # Policies that should be simulated, and their parameters.
    "policies": [
        {
            "archtype": UCB,
            "params": {}
        },
        {
            "archtype": Thompson,
            "params": {}
        },
        {
            "archtype": klUCB,
            "params": {}
        },
        {
            "archtype": AdBandit,
            "params": {
                "alpha": 0.5,
                "horizon": 10000
            }
        }
    ]
}
```

## How to run
First, install the requirements:
```bash
pip2 install -r requirements.txt
```

It should be very straight forward. This will plot the results.
```bash
python2 main.py
```

### With a [`Makefile`](Makefile)
You can also use the provided [`Makefile`](Makefile) file to do this simply:
```bash
make install  # install the requirements
make main     # run and log the main.py script
```

It can be used to check [the quality of the code](pylint.log.txt) with [pylint](https://www.pylint.org/):
```bash
make pylint  # check the code with pylint
```

----

## :boom: TODO
- [x] clean up code
- [X] pass it to Python 3.5 (while still being valid Python 2.7)
- [ ] improve it : add all the bandits algorithms, more arms (Gaussian, Exponentials, ...)
- [X] add my aggregated bandit algorithm
- [ ] explore the behavior of my algorithm, and understand it better (and improve it?)
- [ ] document all that, at least a little bit
- [ ] see if [exhaustive grid search](http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search) could be easily used (but it won't be online!)
- [ ] keep publishing it on GitHub, make the repository public
- [ ] use [hdf5](https://www.hdfgroup.org/HDF5/) with [`h5py`](http://docs.h5py.org/en/latest/quick.html#core-concepts) to store the data
- [ ] implement some algorithms from [this repository](https://github.com/johnmyleswhite/BanditsBook/blob/master/python/algorithms/exp3/exp3.py)

----

## :scroll: License ? [![GitHub license](https://img.shields.io/github/license/Naereen/AlgoBandits.svg)](https://github.com/Naereen/AlgoBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2012 [Olivier Cappé](http://perso.telecom-paristech.fr/%7Ecappe/), [Aurélien Garivier](https://www.math.univ-toulouse.fr/%7Eagarivie/), [Émilie Kaufmann](http://chercheurs.lille.inria.fr/ekaufman/) and for the initial [pymaBandits v1.0](http://mloss.org/software/view/415/) project, and © 2016 [Lilian Besson](https://GitHub.com/Naereen) for the rest.

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/AlgoBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/Naereen/AlgoBandits/README.md?pixel)](https://GitHub.com/Naereen/AlgoBandits/)

[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)

[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://GitHub.com/Naereen/)
