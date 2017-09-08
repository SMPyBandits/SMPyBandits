# Bandit algorithms, Lilian Besson's AlgoBandits project
This repository contains the code of [my](http://perso.crans.org/besson/) numerical environment, written in [Python](https://www.python.org/), in order to perform numerical simulations on *single*-player and *multi*-players [Multi-Armed Bandits (MAB)](https://en.wikipedia.org/wiki/Multi-armed_bandit) algorithms.

![PyPI implementation](https://img.shields.io/pypi/implementation/ansicolortags.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/AlgoBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)

[I (Lilian Besson)](http://perso.crans.org/besson/) have [started my PhD](http://perso.crans.org/besson/phd/) in October 2016, and this is a part of my **on going** research since December 2016.

----

## 1st contribution: The [**policy aggregation algorithm**](Aggregation.md)
I designed and added the [`Aggregator`](Policies/Aggregator.py) policy, in order to test its validity and performance.

It is a "simple" **voting algorithm to combine multiple bandit algorithms into one**.
Basically, it behaves like a simple MAB bandit just based on empirical means (even simpler than UCB), where *arms* are the child algorithms `A_1 .. A_N`, each running in "parallel".

> **For more details**, refer to this file: [`Aggregation.md`](Aggregation.md).

----

## 2nd contribution: [**Multi-players simulation environment**](MultiPlayers.md)
There is another point of view: instead of comparing different single-player policies on the same problem, we can make them play against each other, in a multi-player setting.
The basic difference is about **collisions** : at each time `t`, if two or more user chose to sense the same channel, there is a *collision*. Collisions can be handled in different way from the base station point of view, and from each player point of view.

> **For more details**, refer to this file: [`MultiPlayers.md`](MultiPlayers.md).

----

## [How to run the experiments ?](How_to_run_the_code.md)
> See this document: [`How_to_run_the_code.md`](How_to_run_the_code.md) for more details (or [this documentation page](How_to_run_the_code.html)).

----

## Remarks
- Everything here is done in an imperative, object oriented style. The API of the Arms, Policy and MultiPlayersPolicy classes is documented [in this file (`API.md`)](API.md).
- The code is [clean](logs/main_pylint_log.txt), valid for both [Python 2](logs/main_pylint_log.txt) and [Python 3](logs/main_pylint3_log.txt).
- Some piece of code come from the [pymaBandits](http://mloss.org/software/view/415/) project, but most of them were refactored. Thanks to the initial project!
- [G.Varoquaux](http://gael-varoquaux.info/)'s [joblib](https://pythonhosted.org/joblib/) is used for the [`Evaluator`](Environment/Evaluator.py) and [`EvaluatorMultiPlayers`](Environment/EvaluatorMultiPlayers.py) classes, so the simulations are easily parallelized on multi-core machines. (Put `n_jobs = -1` or `PARALLEL = True` in the config file to use all your CPU cores, as it is by default).

### Should you use bandits?

> In 2015, [Chris Stucchio advised against](https://www.chrisstucchio.com/blog/2015/dont_use_bandits.html)
> the use of bandits, in the context of improving A/B testings, opposed to
> his [2013 blog post](https://www.chrisstucchio.com/blog/2012/bandit_algorithms_vs_ab.html)
> in favor of bandits, also for A/B testings. Both articles are worth
> reading, but in this research we are not studying A/B testing, and it
> has been already proved how efficient bandit algorithms can be for
> real-world and simulated cognitive radio networks. (See for instance
> [this article by Wassim Jouini, Christophe Moy and Jacques Palicot](https://scholar.google.com/scholar?q=Multi-armed+bandit+based+policies+for+cognitive+radio%27s+decision+making+issues+by+W+Jouini%2C+D+Ernst%2C+C+Moy%2C+J+Palicot+2009&btnG=&hl=fr&as_sdt=0%2C39),
> [\["Multi-armed bandit based policies for cognitive radio's decision making issues", W Jouini, D Ernst, C Moy, J Palicot 2009\]](http://orbi.ulg.be/bitstream/2268/16757/1/SCS09_Jouini_Wassim.pdf)).


### Warning
- This work is still in **its early stage of development**! It's [active research](https://github.com/Naereen/AlgoBandits/graphs/contributors).
- This aggregated bandit algorithm has no theoretical warranties what so ever - *yet*.

### UML diagrams
For more details, see [these UML diagrams](uml_diagrams/):

- Packages: organization of the different files:
  [![UML Diagram - Packages of AlgoBandits.git](uml_diagrams/packages_AlgoBandits.png)](uml_diagrams/packages_AlgoBandits.svg)
- Classes: inheritance diagrams of the different classes:
  [![UML Diagram - Classes of AlgoBandits.git](uml_diagrams/classes_AlgoBandits.png)](uml_diagrams/classes_AlgoBandits.svg)

----

## :boom: [TODO](TODO.md)
> See this file [`TODO.md`](TODO.md), and [the issues on GitHub](https://github.com/Naereen/AlgoBandits/issues).

----

## :scroll: License ? [![GitHub license](https://img.shields.io/github/license/Naereen/badges.svg)](https://github.com/Naereen/AlgoBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2012 [Olivier Cappé](http://perso.telecom-paristech.fr/%7Ecappe/), [Aurélien Garivier](https://www.math.univ-toulouse.fr/%7Eagarivie/), [Émilie Kaufmann](http://chercheurs.lille.inria.fr/ekaufman/) and for the initial [pymaBandits v1.0](http://mloss.org/software/view/415/) project, and © 2016-2017 [Lilian Besson](https://GitHub.com/Naereen) for the rest.

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/AlgoBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/Naereen/AlgoBandits/README.md?pixel)](https://GitHub.com/Naereen/AlgoBandits/)
![PyPI implementation](https://img.shields.io/pypi/implementation/ansicolortags.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)
[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://GitHub.com/Naereen/)
