# Bandit algorithms, Lilian Besson's AlgoBandits project
This repository contains the code of [my](http://perso.crans.org/besson/) numerical environment, written in [Python](https://www.python.org/), in order to perform numerical simulations on *single*-player and *multi*-players [Multi-Armed Bandits (MAB)](https://en.wikipedia.org/wiki/Multi-armed_bandit) algorithms.

![PyPI implementation](https://img.shields.io/pypi/implementation/ansicolortags.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/AlgoBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)

[I (Lilian Besson)](http://perso.crans.org/besson/) have [started my PhD](http://perso.crans.org/besson/phd/) in October 2016, and this is a part of my **on going** research since December 2016.

----

> [List of research publications using AlgoBandits](PublicationsWithAlgoBandits.md)

## 1st article, using the [**policy aggregation algorithm**](Aggregation.md)
I designed and added the [`Aggregator`](Policies/Aggregator.py) policy, in order to test its validity and performance.

It is a "simple" **voting algorithm to combine multiple bandit algorithms into one**.
Basically, it behaves like a simple MAB bandit just based on empirical means (even simpler than UCB), where *arms* are the child algorithms `A_1 .. A_N`, each running in "parallel".

> **For more details**, refer to this file: [`Aggregation.md`](Aggregation.md) and [this research article](https://hal.inria.fr/hal-01705292).

----

## 2nd article, using [**Multi-players simulation environment**](MultiPlayers.md)
There is another point of view: instead of comparing different single-player policies on the same problem, we can make them play against each other, in a multi-player setting.
The basic difference is about **collisions** : at each time `t`, if two or more user chose to sense the same channel, there is a *collision*. Collisions can be handled in different way from the base station point of view, and from each player point of view.

> **For more details**, refer to this file: [`MultiPlayers.md`](MultiPlayers.md) and [this research article](https://hal.inria.fr/hal-01629733).

----

## 3rd article, using [**Doubling Trick for Multi-Armed Bandits**](DoublingTrick.md)
I studied what Doubling Trick can and can't do to obtain efficient anytime version of non-anytime optimal Multi-Armed Bandits algorithms.

> **For more details**, refer to this file: [`DoublingTrick.md`](DoublingTrick.md) and [this research article](https://hal.inria.fr/hal-XXX).

----

## Other interesting things
### [Single-player Policies](http://banditslilian.gforge.inria.fr/docs/Policies.html)
- [`SparseWrapper`](http://banditslilian.gforge.inria.fr/docs/Policies.SparseWrapper.html#module-Policies.SparseWrapper) is a generalization of [the SparseUCB from this article](https://arxiv.org/pdf/1706.01383/).
- Implementation of very recent Multi-Armed Bandits algorithms, e.g., [`kl-UCB++`](http://banditslilian.gforge.inria.fr/docs/Policies.klUCBPlusPlus.html) (from [this article](https://hal.inria.fr/hal-01475078)), [`UCB-dagger`](http://banditslilian.gforge.inria.fr/docs/Policies.UCBdagger.html) (from [this article](https://arxiv.org/pdf/1507.07880)),  or [`MOSS-anytime`](http://banditslilian.gforge.inria.fr/docs/Policies.MOSSAnytime.html) (from [this article](http://proceedings.mlr.press/v48/degenne16.pdf)).
- Experimental policies: [`BlackBoxOpt`](http://banditslilian.gforge.inria.fr/docs/Policies.BlackBoxOpt.html) or [`UnsupervisedLearning`](http://banditslilian.gforge.inria.fr/docs/Policies.UnsupervisedLearning.html) (using Gaussian processes to learn the arms distributions).

----

## Remarks
- Everything here is done in an imperative, object oriented style. The API of the Arms, Policy and MultiPlayersPolicy classes is documented [in this file (`API.md`)](API.md).
- The code is [clean](logs/main_pylint_log.txt), valid for both [Python 2](logs/main_pylint_log.txt) and [Python 3](logs/main_pylint3_log.txt).
- Some piece of code come from the [pymaBandits](http://mloss.org/software/view/415/) project, but most of them were refactored. Thanks to the initial project!
- [G.Varoquaux](http://gael-varoquaux.info/)'s [joblib](https://pythonhosted.org/joblib/) is used for the [`Evaluator`](Environment/Evaluator.py) and [`EvaluatorMultiPlayers`](Environment/EvaluatorMultiPlayers.py) classes, so the simulations are easily parallelized on multi-core machines. (Put `n_jobs = -1` or `PARALLEL = True` in the config file to use all your CPU cores, as it is by default).

## [How to run the experiments ?](How_to_run_the_code.md)
> See this document: [`How_to_run_the_code.md`](How_to_run_the_code.md) for more details (or [this documentation page](How_to_run_the_code.html)).

----

### :boom: Warning
- This work is still **experimental**! It's [active research](https://github.com/Naereen/AlgoBandits/graphs/contributors).
- I don't except issues or pull requests on this project, but you are welcome to.
- This project is NOT meant to be a library that you can use elsewhere, but a research tool. In particular, I don't take ensure that any of the Python modules can be imported from another directory than the main directory.

## Contributing?
Contributions (issues, questions, pull requests) are of course welcome, but this project is and will stay a personal environment designed for quick research experiments, and will never try to be an industry-ready module for applications of Multi-Armed Bandits algorithms.

If you want to contribute, please have a look to the [CONTRIBUTING.md](CONTRIBUTING.md) file, and if you want to be more seriously involved, read the [CODEOFCONDUCT.md](CODEOFCONDUCT.md) file.

- You are welcome to [submit an issue](https://github.com/Naereen/AlgoBandits/issues/new), if it was not previously answered,
- If you have interesting example of use of AlgoBandits, please share it! ([Jupyter Notebooks](https://www.jupyter.org/) are preferred). And fill a pull request to [add it to the notebooks examples](notebooks/).

## :boom: [TODO](TODO.md)
> See this file [`TODO.md`](TODO.md), and [the issues on GitHub](https://github.com/Naereen/AlgoBandits/issues).

----

### UML diagrams
For more details, see [these UML diagrams](uml_diagrams/):

- Packages: organization of the different files:
  [![UML Diagram - Packages of AlgoBandits.git](uml_diagrams/packages_AlgoBandits.png)](uml_diagrams/packages_AlgoBandits.svg)
- Classes: inheritance diagrams of the different classes:
  [![UML Diagram - Classes of AlgoBandits.git](uml_diagrams/classes_AlgoBandits.png)](uml_diagrams/classes_AlgoBandits.svg)

----

## :scroll: License ? [![GitHub license](https://img.shields.io/github/license/Naereen/badges.svg)](https://github.com/Naereen/AlgoBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen)

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/AlgoBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/Naereen/AlgoBandits/README.md?pixel)](https://GitHub.com/Naereen/AlgoBandits/)
![PyPI implementation](https://img.shields.io/pypi/implementation/ansicolortags.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)
[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://GitHub.com/Naereen/)
