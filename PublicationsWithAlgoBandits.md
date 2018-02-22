# List of research publications using Lilian Besson's AlgoBandits project

[I (Lilian Besson)](http://perso.crans.org/besson/) have [started my PhD](http://perso.crans.org/besson/phd/) in October 2016, and [this project](https://github.com/Naereen/AlgoBandits/) is a part of my **on going** research since December 2016.

----

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
- More than 65 algorithms, including all known variants of the [`UCB`](Policies/UCB.py), [kl-UCB](Policies/klUCB.py), [`MOSS`](Policies/MOSS.py) and [Thompson Sampling](Policies/Thompson.py) algorithms, as well as other less known algorithms ([`OCUCB`](Policies/OCUCB.py), [`BESA`](Policies/OCUCB.py), [`OSSB`](Policies/OSSB.py) etc).
- [`SparseWrapper`](http://banditslilian.gforge.inria.fr/docs/Policies.SparseWrapper.html#module-Policies.SparseWrapper) is a generalization of [the SparseUCB from this article](https://arxiv.org/pdf/1706.01383/).
- Implementation of very recent Multi-Armed Bandits algorithms, e.g., [`kl-UCB++`](http://banditslilian.gforge.inria.fr/docs/Policies.klUCBPlusPlus.html) (from [this article](https://hal.inria.fr/hal-01475078)), [`UCB-dagger`](http://banditslilian.gforge.inria.fr/docs/Policies.UCBdagger.html) (from [this article](https://arxiv.org/pdf/1507.07880)),  or [`MOSS-anytime`](http://banditslilian.gforge.inria.fr/docs/Policies.MOSSAnytime.html) (from [this article](http://proceedings.mlr.press/v48/degenne16.pdf)).
- Experimental policies: [`BlackBoxOpt`](http://banditslilian.gforge.inria.fr/docs/Policies.BlackBoxOpt.html) or [`UnsupervisedLearning`](http://banditslilian.gforge.inria.fr/docs/Policies.UnsupervisedLearning.html) (using Gaussian processes to learn the arms distributions).

### Arms and problems
- My framework mainly target stochastic bandits, with arms following [`Bernoulli`](Arms/Bernoulli.py), bounded (truncated) or unbounded [`Gaussian`](Arms/Gaussian.py), [`Exponential`](Arms/Exponential.py), [`Gamma`](Arms/Gamma.py) or [`Poisson`](Arms/Poisson.py) distributions.
- The default configuration is to use a fixed problem for N repetitions (e.g. 1000 repetitions, use [`MAB.MAB`](Environment/MAB.py)), but there is also a perfect support for "Bayesian" problems where the mean vector µ1,…,µK change *at every repetition* (see [`MAB.DynamicMAB`](Environment/MAB.py)).
- There is also a good support for Markovian problems, see [`MAB.MarkovianMAB`](Environment/MAB.py), even though I didn't implement any.

----

### :scroll: License ? [![GitHub license](https://img.shields.io/github/license/Naereen/AlgoBandits.svg)](https://github.com/Naereen/AlgoBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/AlgoBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/Naereen/AlgoBandits/README.md?pixel)](https://GitHub.com/Naereen/AlgoBandits/)
![PyPI implementation](https://img.shields.io/pypi/implementation/ansicolortags.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)
[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)
