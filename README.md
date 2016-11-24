# Bandit algorithms and the aggregated bandit
This repository contains the code for some numerical simulations on *single*-player [Multi-Armed Bandits (MAB)](https://en.wikipedia.org/wiki/Multi-armed_bandit) algorithms.

## The **policy aggregation algorithm**
Specifically, [I (Lilian Besson)](http://perso.crans.org/besson/) designed and added the [`Aggr`](Policies/Aggr.py) policy, in order to test it.

It is a simple **voting algorithm to combine multiple bandit algorithms into one**.
Basically, it behaves like the simple [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling), where arms are the child algorithms `A_1 .. A_N`, each running in "parallel".

### More mathematical explanations
Initially, every child algorithms `A_i` has the same "trust" probability `p_i`, and at every step, the aggregated bandit first listen to the decision from all its children `A_i` (`a_{i,t}` in `1 .. K`), and then decide which arm to select by a probabilistic vote: the probability of selecting arm `k` is the sum of the trust probability of the children who voted for arm `k`.
It could also be done the other way: the aggregated bandit could first decide which children to listen to, then trust him.

But we want to update the trust probability of all the children algorithms, not only one, when it was wised to trust them.
Mathematically, when the aggregated arm choose to pull the arm `k` at step `t`, if it yielded a positive reward `r_{k,t}`, then the probability of all children algorithms `A_i` who decided (independently) to chose `k` (i.e., `a_{i,t} = k`) are increased multiplicatively: `p_i <- p_i * exp(+ beta * r_{k,t})` where `beta` is a positive *learning rate*, e.g., `beta = 0.1`.

It is also possible to decrease multiplicatively the trust of all the children algorithms who did not decided to chose the arm `k` at every step `t`: if  `a_{i,t} != k` then `p_i <- p_i * exp(- beta * r_{k,t})`. I did not observe any difference of behavior between these two options (implemented with the Boolean parameter `updateAllChildren`).

### Ensemble voting for MAB algorithms
This algorithm can be seen as the Multi-Armed Bandits (i.e., sequential reinforcement learning) counterpart of an *ensemble voting* technique, as used for classifiers or regression algorithm in usual supervised machine learning (see, e.g., [`sklearn.ensemble.VotingClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier) in [scikit-learn](http://scikit-learn.org/)).

Another approach could be to do some sort of [grid search](http://scikit-learn.org/stable/modules/grid_search.html).

----

## Remarks
- [G.Varoquaux](http://gael-varoquaux.info/)'s [joblib](https://pythonhosted.org/joblib/) is used for the [`Evaluator`](Environment/Evaluator.py) class, so the simulations can easily be parallelized. (Put `n_jobs = -1` or `PARALLEL = True` to use all your CPU cores, as it is by default).
- Most of the code comes from the [pymaBandits](http://mloss.org/software/view/415/) project, but some of them were refactored. Thanks to the initial project!

### Warning
- This work is still in **its early stage of development**!
- This aggregated bandit algorithm has NO THEORETICAL warranties what so ever - *yet*.

----

## Configuration:
A simple python file, [`configuration.py`](configuration.py), is used to import the [arm classes](Arms/), the [policy classes](Policies/) and define the problems and the experiments.

For example, this will compare the classical MAB algorithms [`UCB`](Policies/UCB.py), [`Thompson`](Policies/Thompson.py), [`BayesUCB`](Policies/BayesUCB.py), [`klUCB`](Policies/klUCB.py), and the less classical [`AdBandit`](Policies/AdBandit.py) algorithms.

```python
configuration = {
    "horizon": 10000,    # Finite horizon of the simulation
    "repetitions": 100,  # number of repetitions
    "n_jobs": -1,        # Maximum number of cores for parallelization: use ALL your CPU
    "verbosity": 5,      # Verbosity for the joblib calls
    # Environment configuration, you can set up more than one.
    "environment": [
        {
            "arm_type": Bernoulli,  # Only Bernoulli is available as far as now
            "probabilities": [0.02, 0.02, 0.02, 0.10, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01]
        }
    ],
    # Policies that should be simulated, and their parameters.
    "policies": [
        {"archtype": UCB, "params": {} },
        {"archtype": Thompson, "params": {} },
        {"archtype": klUCB, "params": {} },
        {"archtype": BayesUCB, "params": {} },
        {"archtype": AdBandit, "params": {
                "alpha": 0.5, "horizon": 10000  # AdBandit require to know the horizon
        } }
    ]
}
```

To add an aggregated bandit algorithm ([`Aggr` class](Policies/Aggr.py)), you can use this piece of code, to aggregate all the algorithms defined before and dynamically add it to `configuration`:
```python
current_policies = configuration["policies"]
configuration["policies"] = current_policies +
    [{  # Add one Aggr policy, from all the policies defined above
        "archtype": Aggr,
        "params": {
            "learningRate": 0.05,  # Tweak this if needed
            "updateAllChildren": True,
            "children": current_policies,
        },
    }]
```

----

## How to run the experiments ?
*First*, install the requirements:
```bash
pip install -r requirements.txt
```

*Then*, it should be very straight forward to run some experiment.
This will run the simulation, average them (by `repetitions`) and plot the results:
```bash
python main.py
```

### Or with a [`Makefile`](Makefile) ?
You can also use the provided [`Makefile`](Makefile) file to do this simply:
```bash
make install  # install the requirements
make run      # run and log the main.py script
```

It can be used to check [the quality of the code](logs/main_pylint_log.txt) with [pylint](https://www.pylint.org/):
```bash
make lint lint3  # check the code with pylint
```

----

## Some illustrations
Here are some plots illustrating the performances of the different [policies](Policies/) implemented in this project, against various problems (with [`Bernoulli`](Arms/Bernoulli.py) arms only):

### Small tests
[![5 tests - AdBandit and Aggr](plots/5_tests_AdBandit__et_Aggr.png)](plots/5_tests_AdBandit__et_Aggr.png)
[![2000 steps - 100 repetition](plots/2000_steps__100_average.png)](plots/2000_steps__100_average.png)

### Larger tests
[![10000 steps - 50 repetition - 6 policies - With 4 Aggr](plots/10000_steps__50_repetition_6_policies_4_Aggr.png)](plots/10000_steps__50_repetition_6_policies_4_Aggr.png)
[![10000 steps - 50 repetition - 6 policies - With Softmax and 1 Aggr](plots/10000_steps__50_repetition_6_policies_with_Softmax_1_Aggr.png)](plots/10000_steps__50_repetition_6_policies_with_Softmax_1_Aggr.png)

### Some examples where [`Aggr`](Policies/Aggr.py) performs well
[![Aggr is the best here](plots/Aggr_is_the_best_here.png)](plots/Aggr_is_the_best_here.png)
[![one Aggr does very well](plots/one_Aggr_does_very_well.png)](plots/one_Aggr_does_very_well.png)

### One last example
The [`Aggr`](Policies/Aggr.py) can have a fixed learning rate, whose value has a great effect on its performance, as illustrated here:
[![20000 steps - 100 repetition - 6 policies - With 5 Aggr](plots/20000_steps__100_repetition_6_policies_5_Aggr.png)](plots/20000_steps__100_repetition_6_policies_5_Aggr.png)

### One a harder problem
[![example harder problem](plots/example_harder_problem.png)](plots/example_harder_problem.png)

----

## A note on execution times, speed and profiling.
- About (time) profiling with Python (2 or 3): `cProfile` or `profile` [in Python 2 documentation](https://docs.python.org/2/library/profile.html) ([in Python 3 documentation](https://docs.python.org/2/library/profile.html)), [this StackOverflow thread](https://stackoverflow.com/a/7693928/5889533), [this blog post](https://www.huyng.com/posts/python-performance-analysis), and the documentation of [`line_profiler`](https://github.com/rkern/line_profiler) (to profile lines instead of functions) and [`pycallgraph`](http://pycallgraph.slowchop.com/en/master/) (to illustrate function calls) and [`yappi`](https://pypi.python.org/pypi/yappi/) (which seems to be thread aware).
- See also [`pyreverse`](https://www.logilab.org/blogentry/6883) to get nice UML-like diagrams illustrating the relationships of packages and classes between each-other.

----

## Code organization
### Remarks (bragging):
- Everything here is done in an imperative, object oriented style.
- The code is [clean](logs/main_pylint_log.txt), valid for both [Python 2](logs/main_pylint_log.txt) and [Python 3](logs/main_pylint3_log.txt).

### Layout of the code:
- Arms are defined in [this folder (`Arms/`)](Arms/), see for example [`Arms.Bernoulli`](Arms/Bernoulli.py)
- MAB algorithms (also called policies) are defined in [this folder (`Policies/`)](Policies/), see for example [`Policies.Dummy`](Policies/Dummy.py) for a fully random policy, [`Policies.EpsilonGreedy`](Policies/EpsilonGreedy.py) for the epsilon-greedy random policy, [`Policies.UCB`](Policies/UCB.py) for the "simple" UCB algorithm, or also [`Policies.BayesUCB`](Policies/BayesUCB.py), [`Policies.klUCB`](Policies/klUCB.py) for two UCB-like algorithms, [`Policies.AdBandits`](Policies/AdBandits.py) for the [AdBandits](https://github.com/flaviotruzzi/AdBandits/) algorithm, and [`Policies.Aggr`](Policies/Aggr.py) for my *aggregated bandits* algorithms.
- Environments to encapsulate date are defined in [this folder (`Environment/`)](Environment/): MAB problem use the class [`Environment.MAB`](Environment/MAB.py), simulation results are stored in a [`Environment.Result`](Environment/Result.py), and the class to evaluate multi-policy single-player multi-env is [`Environment.Evaluate`](Environment/Evaluate.py).
- [`configuration.py`](configuration.py) imports all the classes, and define the simulation parameters as a dictionary (JSON-like).
- [`main.py`](main.py) runs the simulations, then display the final ranking of the different policies and plots the results (saved to [this folder (`plots/`)](plots/)).

### UML diagrams
For more details, see [these UML diagrams](uml_diagrams/):

- Packages: organization of the different files:
[![UML Diagram - Packages of AlgoBandits.git](uml_diagrams/packages_AlgoBandits.png)](uml_diagrams/packages_AlgoBandits.svg)
- Classes: inheritance diagrams of the different classes:
[![UML Diagram - Classes of AlgoBandits.git](uml_diagrams/classes_AlgoBandits.png)](uml_diagrams/classes_AlgoBandits.svg)

----

## :boom: TODO
### Initial things to do - OK
- [x] clean up code, OK
- [x] lint the code and make it "perfect", OK
- [x] pass it to Python 3.5 (while still being valid Python 2.7), OK
- [x] add more arms: Gaussian, Exponential, Poisson, OK
- [x] add my aggregated bandit algorithm, OK

### Improve the code
- [x] In fact, [exhaustive grid search](http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search) cannot be easily used as it cannot run *on-line*! Sadly, OK.
- [x] add plots that show the percentage of optimal arms play ([e.g., as done in this paper](http://www.cs.mcgill.ca/~vkules/bandits.pdf#page=11))
- [x] fully profile my code, with [`cProfile`](https://docs.python.org/2/library/profile.html) for functions and [`line_profiler`](https://github.com/rkern/line_profiler) for line-by-line. No surprise here: [`Beta.py`](Policies/Beta.py) is the slow part, as it takes time to sample and compute the quantiles (even by using the good `numpy.random`/`scipy.stats` functions). See for instance [this log file (with `cProfile`)](logs/main_py3_profile_log.txt) or [this one (with `line_profiler`)](logs/main_py3_line_profiler_log.txt).
- [ ] I could have tried to improve the bottlenecks, with smart `numpy`/`scipy` code, or [`numba` ?](http://numba.pydata.org/), or [`cython`](http://cython.org/) code ? Not so easy, not so interesting...
- [ ] explore the behavior of my Aggr algorithm, and understand it better (and improve it?)

### Better storing of the simulation results
- [ ] use [hdf5](https://www.hdfgroup.org/HDF5/) (with [`h5py`](http://docs.h5py.org/en/latest/quick.html#core-concepts)) to store the data, on the run (to never lose data, even if the simulation gets killed)

### Publicly release it ?
- [x] keep it up-to-date [on GitHub](https://github.com/Naereen/AlgoBandits)
- [x] I could document this project better. But, well, there is no [Sphinx](http://sphinx-doc.org/) documentation yet, but each file has a docstring, some useful comments for the interesting part, and this very page you are reading contains [insights on how to use the framework](#configuration) as well as [the organization of the code](#code-organization).

### More MAB algorithms
- [ ] implement some more algorithms, e.g., from [this repository](https://github.com/johnmyleswhite/BanditsBook/blob/master/python/algorithms/exp3/exp3.py)
- [ ] add more basic algorithms, e.g., from [this survey](http://homes.di.unimi.it/~cesabian/Pubblicazioni/banditSurvey.pdf) or [this document](http://www.cs.mcgill.ca/~vkules/bandits.pdf)

### Multi-players simulations ?
- [ ] implement a multi-player simulation environment as well!
- [ ] implement the [`rho_rand`](http://ieeexplore.ieee.org/document/5462144/), [`TDFS`](https://arxiv.org/abs/0910.2065v3), [`MEGA`](https://arxiv.org/abs/1404.5421), [`Musical Chair` and `Dynamic Musical Chair`](https://arxiv.org/abs/1512.02866) multi-player algorithms

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
