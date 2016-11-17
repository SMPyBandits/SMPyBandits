# Aggregated bandits

Most of the code comes from the [pymabandits](http://mloss.org/software/view/415/) project, but some of them were refactored.

[joblib](https://pythonhosted.org/joblib/) is used for the [`Evaluator`](Environment/Evaluator.py) class, so the simulations can easily be parallelized. (Put `n_jobs = -1` or `PARALLEL = True` to use all your CPU cores).

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

----

## :boom: TODO
- clean up code
- pass to Python 3.5
- improve it : add all the bandits algorithms, more arms (Gaussian, Exponentials, ...)
- add my aggregated bandit algorithms, explore it and understand it better
- document it a little bit
- publish it on GitHub

----

## :scroll: License ? [![GitHub license](https://img.shields.io/github/license/Naereen/AlgoBandits.svg)](https://github.com/Naereen/AlgoBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2012 [Olivier Cappé](http://mloss.org/software/author/olivier-cappe) [Aurélien Garivier](http://mloss.org/software/author/aurelien-garivier) [Émilie Kaufmann](http://mloss.org/software/author/emilie-kaufmann) and for the initial [pymaBandits v1.0](http://mloss.org/software/view/415/) project, and © 2016 [Lilian Besson](https://GitHub.com/Naereen) for the rest.

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/AlgoBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/Naereen/AlgoBandits/README.md?pixel)](https://GitHub.com/Naereen/AlgoBandits/)

[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)

[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://GitHub.com/Naereen/)
