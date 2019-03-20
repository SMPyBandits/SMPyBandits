# List of research publications using Lilian Besson's SMPyBandits project

[I (Lilian Besson)](https://perso.crans.org/besson/) have [started my PhD](https://perso.crans.org/besson/phd/) in October 2016, and [this project](https://github.com/SMPyBandits/SMPyBandits/) is a part of my **on going** research since December 2016.

----

## 1st article, about [**policy aggregation algorithm (aka model selection)**](Aggregation.md)
I designed and added the [`Aggregator`](https://smpybandits.github.io/docs/Policies.Aggregator.html) policy, in order to test its validity and performance.

It is a "simple" **voting algorithm to combine multiple bandit algorithms into one**.
Basically, it behaves like a simple MAB bandit just based on empirical means (even simpler than UCB), where *arms* are the child algorithms `A_1 .. A_N`, each running in "parallel".

> **For more details**, refer to this file: [`Aggregation.md`](Aggregation.md) and [this research article](https://hal.inria.fr/hal-01705292).

> PDF : [BKM_IEEEWCNC_2018.pdf](https://hal.inria.fr/hal-01705292/document) | HAL notice : [BKM_IEEEWCNC_2018](https://hal.inria.fr/hal-01705292/) | BibTeX : [BKM_IEEEWCNC_2018.bib](https://hal.inria.fr/hal-01705292/bibtex) | [Source code and documentation](Aggregation.md)
> [![Published](https://img.shields.io/badge/Published%3F-accepted-green.svg)](https://hal.inria.fr/hal-01705292)  [![Maintenance](https://img.shields.io/badge/Maintained%3F-finished-green.svg)](https://bitbucket.org/lbesson/aggregation-of-multi-armed-bandits-learning-algorithms-for/commits/)  [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://bitbucket.org/lbesson/ama)

----

## 2nd article, about [**Multi-players Multi-Armed Bandits**](MultiPlayers.md)
There is another point of view: instead of comparing different single-player policies on the same problem, we can make them play against each other, in a multi-player setting.
The basic difference is about **collisions** : at each time `t`, if two or more user chose to sense the same channel, there is a *collision*. Collisions can be handled in different way from the base station point of view, and from each player point of view.

> **For more details**, refer to this file: [`MultiPlayers.md`](MultiPlayers.md) and [this research article](https://hal.inria.fr/hal-01629733).

> PDF : [BK__ALT_2018.pdf](https://hal.inria.fr/hal-01629733/document) | HAL notice : [BK__ALT_2018](https://hal.inria.fr/hal-01629733/) | BibTeX : [BK__ALT_2018.bib](https://hal.inria.fr/hal-01629733/bibtex) | [Source code and documentation](MultiPlayers.md)
> [![Published](https://img.shields.io/badge/Published%3F-accepted-green.svg)](http://www.cs.cornell.edu/conferences/alt2018/index.html#accepted)  [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://bitbucket.org/lbesson/multi-player-bandits-revisited/commits/)  [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://bitbucket.org/lbesson/ama)

----

## 3rd article, using [**Doubling Trick for Multi-Armed Bandits**](DoublingTrick.md)
I studied what Doubling Trick can and can't do to obtain efficient anytime version of non-anytime optimal Multi-Armed Bandits algorithms.

> **For more details**, refer to this file: [`DoublingTrick.md`](DoublingTrick.md) and [this research article](https://hal.inria.fr/hal-01736357).

> PDF : [BK__DoublingTricks_2018.pdf](https://hal.inria.fr/hal-01736357/document) | HAL notice : [BK__DoublingTricks_2018](https://hal.inria.fr/hal-01736357/) | BibTeX : [BK__DoublingTricks_2018.bib](https://hal.inria.fr/hal-01736357/bibtex) | [Source code and documentation](DoublingTrick.md)
> [![Published](https://img.shields.io/badge/Published%3F-waiting-orange.svg)](https://hal.inria.fr/hal-01736357) [![Maintenance](https://img.shields.io/badge/Maintained%3F-almost%20finished-orange.svg)](https://bitbucket.org/lbesson/what-doubling-tricks-can-and-cant-do-for-multi-armed-bandits/commits/) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://bitbucket.org/lbesson/ama)

----

## 4th article, about [**Piece-Wise Stationary Multi-Armed Bandits**](NonStationaryBandits.md)
With Emilie Kaufmann, we studied the Generalized Likelihood Ratio Test (GLRT) for sub-Bernoulli distributions, and proposed the B-GLRT algorithm for change-point detection for piece-wise stationary one-armed bandit problems. We combined the B-GLRT with the kl-UCB multi-armed bandit algorithm and proposed the GLR-klUCB algorithm for piece-wise stationary multi-armed bandit problems. We prove finite-time guarantees for the B-GLRT and the GLR-klUCB algorithm, and we illustrate its performance with extensive numerical experiments.

> **For more details**, refer to this file: [`NonStationaryBandits.md`](NonStationaryBandits.md) and [this research article](https://hal.inria.fr/hal-02006471).

> PDF : [BK__COLT_2019.pdf](https://hal.inria.fr/hal-02006471/document) | HAL notice : [BK__COLT_2019](https://hal.inria.fr/hal-02006471/) | BibTeX : [BK__COLT_2019.bib](https://hal.inria.fr/hal-02006471/bibtex) | [Source code and documentation](NonStationaryBandits.html)
> [![Published](https://img.shields.io/badge/Published%3F-waiting-orange.svg)](https://hal.inria.fr/hal-02006471) [![Maintenance](https://img.shields.io/badge/Maintained%3F-almost%20finished-orange.svg)](https://bitbucket.org/lbesson/combining-the-generalized-likelihood-ratio-test-and-kl-ucb-for/commits/) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://bitbucket.org/lbesson/ama)

----

## Other interesting things
### [Single-player Policies](https://smpybandits.github.io/docs/Policies.html)
- More than 65 algorithms, including all known variants of the [`UCB`](https://smpybandits.github.io/docs/Policies.UCB.html), [kl-UCB](https://smpybandits.github.io/docs/Policies.klUCB.html), [`MOSS`](https://smpybandits.github.io/docs/Policies.MOSS.html) and [Thompson Sampling](https://smpybandits.github.io/docs/Policies.Thompson.html) algorithms, as well as other less known algorithms (https://smpybandits.github.io/docs/[`OCUCB`](Policies.OCUCB.html), [`BESA`](https://smpybandits.github.io/docs/Policies.OCUCB.html), [`OSSB`](https://smpybandits.github.io/docs/Policies.OSSB.html) etc).
- [`SparseWrapper`](https://smpybandits.github.io/docs/Policies.SparseWrapper.html#module-Policies.SparseWrapper) is a generalization of [the SparseUCB from this article](https://arxiv.org/pdf/1706.01383/).
- Implementation of very recent Multi-Armed Bandits algorithms, e.g., [`kl-UCB++`](https://smpybandits.github.io/docs/Policies.klUCBPlusPlus.html) (from [this article](https://hal.inria.fr/hal-01475078)), [`UCB-dagger`](https://smpybandits.github.io/docs/Policies.UCBdagger.html) (from [this article](https://arxiv.org/pdf/1507.07880)),  or [`MOSS-anytime`](https://smpybandits.github.io/docs/Policies.MOSSAnytime.html) (from [this article](http://proceedings.mlr.press/v48/degenne16.pdf)).
- Experimental policies: [`BlackBoxOpt`](https://smpybandits.github.io/docs/Policies.BlackBoxOpt.html) or [`UnsupervisedLearning`](https://smpybandits.github.io/docs/Policies.UnsupervisedLearning.html) (using Gaussian processes to learn the arms distributions).

### Arms and problems
- My framework mainly targets stochastic bandits, with arms following [`Bernoulli`](https://smpybandits.github.io/docs/Arms.Bernoulli.html), bounded (truncated) or unbounded [`Gaussian`](https://smpybandits.github.io/docs/Arms.Gaussian.html), [`Exponential`](https://smpybandits.github.io/docs/Arms.Exponential.html), [`Gamma`](https://smpybandits.github.io/docs/Arms.Gamma.html) or [`Poisson`](https://smpybandits.github.io/docs/Arms.Poisson.html) distributions.
- The default configuration is to use a fixed problem for N repetitions (e.g. 1000 repetitions, use [`MAB.MAB`](https://smpybandits.github.io/docs/Environment.MAB.html)), but there is also a perfect support for "Bayesian" problems where the mean vector µ1,…,µK change *at every repetition* (see [`MAB.DynamicMAB`](https://smpybandits.github.io/docs/Environment.MAB.html)).
- There is also a good support for Markovian problems, see [`MAB.MarkovianMAB`](https://smpybandits.github.io/docs/Environment.MAB.html#Environment.MAB.MarkovianMAB), even though I didn't implement any policies tailored for Markovian problems.
- I'm actively working on adding a very clean support for non-stationary MAB problems, and [`MAB.PieceWiseStationaryMAB`](https://smpybandits.github.io/docs/Environment.MAB.html#Environment.MAB.PieceWiseStationaryMAB) is already working well. Use it with policies designed for piece-wise stationary problems, like [Discounted-Thompson](https://smpybandits.github.io/docs/Policies.DiscountedThompson.html), [CD-UCB](https://smpybandits.github.io/docs/Policies.CD_UCB.html), [M-UCB](https://smpybandits.github.io/docs/Policies.Monitored_UCB.html), [SW-UCB#](https://smpybandits.github.io/docs/Policies.SWHash_UCB.html).

----

### :scroll: License ? [![GitHub license](https://img.shields.io/github/license/SMPyBandits/SMPyBandits.svg)](https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

Note: I have worked on other topics during [my PhD](https://perso.crans.org/besson/phd/), you can find my research articles [on my website](https://perso.crans.org/besson/articles/), or have a look to [my Google Scholar profile](https://scholar.google.com/citations?hl=en&user=bt3upq8AAAAJ) or [résumé on HAL](https://cv.archives-ouvertes.fr/lilian-besson).

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
