# *SMPyBandits*
**Open-Source Python package for Single- and Multi-Players multi-armed Bandits algorithms**.

<img width="50%" src="logo_large.png" align="right" alt="Logo, logo_large.png"/>

This repository contains the code of [Lilian Besson's](https://perso.crans.org/besson/) numerical environment, written in [Python (2 or 3)](https://www.python.org/), for numerical simulations on :slot_machine: *single*-player and *multi*-players [Multi-Armed Bandits (MAB)](https://en.wikipedia.org/wiki/Multi-armed_bandit) algorithms.

A complete Sphinx-generated documentation is on [SMPyBandits.GitHub.io](https://smpybandits.github.io/).

## Quick presentation

It contains the most complete collection of single-player (classical) bandit algorithms on the Internet ([over 65!](https://smpybandits.github.io/docs/Policies/)), as well as implementation of all the state-of-the-art [multi-player algorithms](https://smpybandits.github.io/docs/PoliciesMultiPlayers/).

I follow very actively the latest publications related to Multi-Armed Bandits (MAB) research, and usually implement quite quickly the new algorithms (see for instance, [Exp3++](https://smpybandits.github.io/docs/Policies.Exp3PlusPlus.html), [CORRAL](https://smpybandits.github.io/docs/Policies.CORRAL.html) and [SparseUCB](https://smpybandits.github.io/docs/Policies.SparseUCB.html) were each introduced by articles ([for Exp3++](https://arxiv.org/pdf/1702.06103), [for CORRAL](https://arxiv.org/abs/1612.06246v2), [for SparseUCB](https://arxiv.org/abs/1706.01383)) presented at COLT in July 2017, [LearnExp](https://smpybandits.github.io/docs/Policies.LearnExp.html) comes from a [NIPS 2017 paper](https://arxiv.org/abs/1702.04825), and [kl-UCB++](https://smpybandits.github.io/docs/Policies.klUCBPlusPlus.html) from an [ALT 2017 paper](https://hal.inria.fr/hal-01475078).).
More recent examples are [klUCBswitch](https://smpybandits.github.io/docs/Policies.klUCBswitch.html) from [a paper from May 2018](https://arxiv.org/abs/1805.05071), and also [MusicalChairNoSensing](https://smpybandits.github.io/docs/Policies.MusicalChairNoSensing.html) from [a paper from August 2018](https://arxiv.org/abs/1808.08416).

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/SMPyBandits/SMPyBandits/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel)](https://GitHub.com/SMPyBandits/SMPyBandits/)
[![![PyPI version](https://img.shields.io/pypi/v/smpybandits.svg)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![![PyPI implementation](https://img.shields.io/pypi/implementation/smpybandits.svg)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![![PyPI pyversions](https://img.shields.io/pypi/pyversions/smpybandits.svg?logo=python)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![![PyPI download](https://img.shields.io/pypi/dm/smpybandits.svg)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![![PyPI status](https://img.shields.io/pypi/status/smpybandits.svg)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![Documentation Status](https://readthedocs.org/projects/smpybandits/badge/?version=latest)](https://SMPyBandits.ReadTheDocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/SMPyBandits/SMPyBandits.svg?branch=master)](https://travis-ci.org/SMPyBandits/SMPyBandits)
[![Stars of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/stars/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/stargazers)
[![Releases of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/release/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/releases)

- Classical MAB have a lot of applications, from clinical trials, A/B testing, game tree exploration, and online content recommendation (my framework does *not* implement contextual bandit - yet).
- [Multi-player MAB](MultiPlayers.md) have applications in Cognitive Radio, and my framework implements [all the collision models](https://smpybandits.github.io/docs/Environment.CollisionModels.html) found in the literature, as well as all the algorithms from the last 10 years or so ([`rhoRand`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.rhoRand.html) from 2009, [`MEGA`](https://smpybandits.github.io/docs/Policies.MEGA.html) from 2015, [`MusicalChair`](https://smpybandits.github.io/docs/Policies.MusicalChair.html), and our state-of-the-art algorithms [`RandTopM`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.RandTopM.html) and [`MCTopM`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.MCTopM.html), along with very recent algorithms [`SIC-MMAB`](https://smpybandits.github.io/docs/Policies.SIC_MMAB.html) from [arXiv:1809.08151](https://arxiv.org/abs/1809.08151) and [`MusicalChairNoSensing`](https://smpybandits.github.io/docs/Policies.MusicalChairNoSensing.html) from [arXiv:1808.08416](https://arxiv.org/abs/1808.08416)).
- I'm working on adding a clean support for non-stationary MAB problem, and I will soon implement all state-of-the-art algorithms for these problems.

With this numerical framework, simulations can run on a single CPU or a multi-core machine, and summary plots are automatically saved as high-quality PNG, PDF and EPS (ready for being used in research article).
Making new simulations is very easy, one only needs to write a configuration script and basically no code! See [these examples](https://github.com/SMPyBandits/SMPyBandits/search?l=Python&q=configuration&type=&utf8=%E2%9C%93) (files named `configuration_*.py`).

A complete [Sphinx](http://sphinx-doc.org/) documentation for each algorithms and every piece of code (included constants in the configurations!) is available here: [SMPyBandits.GitHub.io](https://smpybandits.github.io/). (I will use [ReadTheDocs](https://readthedocs.org/) for this project, but I won't use any *continuous integration*, don't even think of it!)


> [I (Lilian Besson)](https://perso.crans.org/besson/) have [started my PhD](https://perso.crans.org/besson/phd/) in October 2016, and this is a part of my **on going** research since December 2016.
>
> I launched the [documentation](https://smpybandits.github.io/) on March 2017, I wrote my first research articles using this framework in 2017 and decided to (finally) open-source my project in February 2018.
> [![Commits of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/commits/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/commits/master) / [![Date of last commit of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/last-commit/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/commits/master)
> [![Issues of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/issues/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/issues) : [![Open issues of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/open-issues/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/issues?q=is%3Aopen+is%3Aissue) / [![Closed issues of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/closed-issues/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/issues?q=is%3Aclosed+is%3Aissue)

----

## How to cite this work?
If you use this package for your own work, please consider citing it with [this piece of BibTeX](SMPyBandits.bib):


```bibtex
@misc{SMPyBandits,
    title =   {{SMPyBandits: an Open-Source Research Framework for Single and Multi-Players Multi-Arms Bandits (MAB) Algorithms in Python}},
    author =  {Lilian Besson},
    year =    {2018},
    url =     {https://github.com/SMPyBandits/SMPyBandits/},
    howpublished = {Online at: \url{github.com/SMPyBandits/SMPyBandits}},
    note =    {Code at https://github.com/SMPyBandits/SMPyBandits/, documentation at https://smpybandits.github.io/}
}
```

I also wrote a small paper to present *SMPyBandits*, and I will send it to [JMLR MLOSS](http://jmlr.org/mloss/).
The paper can be consulted [here on my website](https://perso.crans.org/besson/articles/SMPyBandits.pdf).

> A DOI will arrive as soon as possible! I tried to publish [a paper](docs/paper/paper.md) on both [JOSS](http://joss.theoj.org/) and [MLOSS](http://mloss.org/software/).

## [List of research publications using SMPyBandits](PublicationsWithSMPyBandits.md)

### 1st article, about [**policy aggregation algorithm (aka model selection)**](Aggregation.md)
I designed and added the [`Aggregator`](https://smpybandits.github.io/docs/Policies.Aggregator.html) policy, in order to test its validity and performance.

It is a "simple" **voting algorithm to combine multiple bandit algorithms into one**.
Basically, it behaves like a simple MAB bandit just based on empirical means (even simpler than UCB), where *arms* are the child algorithms `A_1 .. A_N`, each running in "parallel".

> **For more details**, refer to this file: [`Aggregation.md`](Aggregation.md) and [this research article](https://hal.inria.fr/hal-01705292).

> PDF : [BKM_IEEEWCNC_2018.pdf](https://hal.inria.fr/hal-01705292/document) | HAL notice : [BKM_IEEEWCNC_2018](https://hal.inria.fr/hal-01705292/) | BibTeX : [BKM_IEEEWCNC_2018.bib](https://hal.inria.fr/hal-01705292/bibtex) | [Source code and documentation](Aggregation.md)
> [![Published](https://img.shields.io/badge/Published%3F-accepted-green.svg)](https://hal.inria.fr/hal-01705292)  [![Maintenance](https://img.shields.io/badge/Maintained%3F-finished-green.svg)](https://bitbucket.org/lbesson/aggregation-of-multi-armed-bandits-learning-algorithms-for/commits/)  [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://bitbucket.org/lbesson/ama)

### 2nd article, about [**Multi-players Multi-Armed Bandits**](MultiPlayers.md)
There is another point of view: instead of comparing different single-player policies on the same problem, we can make them play against each other, in a multi-player setting.
The basic difference is about **collisions** : at each time `t`, if two or more user chose to sense the same channel, there is a *collision*. Collisions can be handled in different way from the base station point of view, and from each player point of view.

> **For more details**, refer to this file: [`MultiPlayers.md`](MultiPlayers.md) and [this research article](https://hal.inria.fr/hal-01629733).

> PDF : [BK__ALT_2018.pdf](https://hal.inria.fr/hal-01629733/document) | HAL notice : [BK__ALT_2018](https://hal.inria.fr/hal-01629733/) | BibTeX : [BK__ALT_2018.bib](https://hal.inria.fr/hal-01629733/bibtex) | [Source code and documentation](MultiPlayers.md)
> [![Published](https://img.shields.io/badge/Published%3F-accepted-green.svg)](http://www.cs.cornell.edu/conferences/alt2018/index.html#accepted)  [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://bitbucket.org/lbesson/multi-player-bandits-revisited/commits/)  [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://bitbucket.org/lbesson/ama)

### 3rd article, using [**Doubling Trick for Multi-Armed Bandits**](DoublingTrick.md)
I studied what Doubling Trick can and can't do to obtain efficient anytime version of non-anytime optimal Multi-Armed Bandits algorithms.

> **For more details**, refer to this file: [`DoublingTrick.md`](DoublingTrick.md) and [this research article](https://hal.inria.fr/hal-01736357).

> PDF : [BK__DoublingTricks_2018.pdf](https://hal.inria.fr/hal-01736357/document) | HAL notice : [BK__DoublingTricks_2018](https://hal.inria.fr/hal-01736357/) | BibTeX : [BK__DoublingTricks_2018.bib](https://hal.inria.fr/hal-01736357/bibtex) | [Source code and documentation](DoublingTrick.md)
> [![Published](https://img.shields.io/badge/Published%3F-waiting-orange.svg)](https://hal.inria.fr/hal-01736357) [![Maintenance](https://img.shields.io/badge/Maintained%3F-almost%20finished-orange.svg)](https://bitbucket.org/lbesson/what-doubling-tricks-can-and-cant-do-for-multi-armed-bandits/commits/) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://bitbucket.org/lbesson/ama)

### 4th article, about [**Piece-Wise Stationary Multi-Armed Bandits**](NonStationaryBandits.md)
With Emilie Kaufmann, we studied the Generalized Likelihood Ratio Test (GLRT) for sub-Bernoulli distributions, and proposed the B-GLRT algorithm for change-point detection for piece-wise stationary one-armed bandit problems. We combined the B-GLRT with the kl-UCB multi-armed bandit algorithm and proposed the GLR-klUCB algorithm for piece-wise stationary multi-armed bandit problems. We prove finite-time guarantees for the B-GLRT and the GLR-klUCB algorithm, and we illustrate its performance with extensive numerical experiments.

> **For more details**, refer to this file: [`NonStationaryBandits.md`](NonStationaryBandits.md) and [this research article](https://hal.inria.fr/hal-02006471).

> PDF : [BK__COLT_2019.pdf](https://hal.inria.fr/hal-02006471/document) | HAL notice : [BK__COLT_2019](https://hal.inria.fr/hal-02006471/) | BibTeX : [BK__COLT_2019.bib](https://hal.inria.fr/hal-02006471/bibtex) | [Source code and documentation](NonStationaryBandits.html)
> [![Published](https://img.shields.io/badge/Published%3F-waiting-orange.svg)](https://hal.inria.fr/hal-02006471) [![Maintenance](https://img.shields.io/badge/Maintained%3F-almost%20finished-orange.svg)](https://bitbucket.org/lbesson/combining-the-generalized-likelihood-ratio-test-and-kl-ucb-for/commits/) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://bitbucket.org/lbesson/ama)

----

## Other interesting things
### [Single-player Policies](https://smpybandits.github.io/docs/Policies.html)
- More than 65 algorithms, including all known variants of the [`UCB`](https://smpybandits.github.io/docs/Policies.UCB.html), [kl-UCB](https://smpybandits.github.io/docs//Policies.klUCB.html), [`MOSS`](https://smpybandits.github.io/docs/Policies.MOSS.html) and [Thompson Sampling](https://smpybandits.github.io/docs/Policies.Thompson.html) algorithms, as well as other less known algorithms ([`OCUCB`](https://smpybandits.github.io/docs/Policies.OCUCB.html), [`BESA`](https://smpybandits.github.io/docs/Policies.OCUCB.html), [`OSSB`](https://smpybandits.github.io/docs/Policies.OSSB.html) etc).
- For instance, [`SparseWrapper`](https://smpybandits.github.io/docs/Policies.SparseWrapper.html#module-Policies.SparseWrapper) is a generalization of [the SparseUCB from this article](https://arxiv.org/pdf/1706.01383/).
- Implementation of very recent Multi-Armed Bandits algorithms, e.g., [`kl-UCB++`](https://smpybandits.github.io/docs/Policies.klUCBPlusPlus.html) (from [this article](https://hal.inria.fr/hal-01475078)), [`UCB-dagger`](https://smpybandits.github.io/docs/Policies.UCBdagger.html) (from [this article](https://arxiv.org/pdf/1507.07880)),  or [`MOSS-anytime`](https://smpybandits.github.io/docs/Policies.MOSSAnytime.html) (from [this article](http://proceedings.mlr.press/v48/degenne16.pdf)).
- Experimental policies: [`BlackBoxOpt`](https://smpybandits.github.io/docs/Policies.BlackBoxOpt.html) or [`UnsupervisedLearning`](https://smpybandits.github.io/docs/Policies.UnsupervisedLearning.html) (using Gaussian processes to learn the arms distributions).

### Arms and problems
- My framework mainly targets stochastic bandits, with arms following [`Bernoulli`](https://smpybandits.github.io/docs/Arms.Bernoulli.html), bounded (truncated) or unbounded [`Gaussian`](https://smpybandits.github.io/docs/Arms.Gaussian.html), [`Exponential`](https://smpybandits.github.io/docs/Arms.Exponential.html), [`Gamma`](https://smpybandits.github.io/docs/Arms.Gamma.html) or [`Poisson`](https://smpybandits.github.io/docs/Arms.Poisson.html) distributions, and more.
- The default configuration is to use a fixed problem for N repetitions (e.g. 1000 repetitions, use [`MAB.MAB`](https://smpybandits.github.io/docs/Environment.MAB.html#Environment.MAB.MAB)), but there is also a perfect support for "Bayesian" problems where the mean vector µ1,…,µK change *at every repetition* (see [`MAB.DynamicMAB`](https://smpybandits.github.io/docs/Environment.MAB.html#Environment.MAB.DynamicMAB)).
- There is also a good support for Markovian problems, see [`MAB.MarkovianMAB`](https://smpybandits.github.io/docs/Environment.MAB.html#Environment.MAB.MarkovianMAB), even though I didn't implement any policies tailored for Markovian problems.
- I'm actively working on adding a very clean support for non-stationary MAB problems, and [`MAB.PieceWiseStationaryMAB`](https://smpybandits.github.io/docs/Environment.MAB.html#Environment.MAB.PieceWiseStationaryMAB) is already working well. Use it with policies designed for piece-wise stationary problems, like [Discounted-Thompson](https://smpybandits.github.io/docs/Policies.DiscountedThompson.html), one of the [CD-UCB](https://smpybandits.github.io/docs/Policies.CD_UCB.html) algorithms, [M-UCB](https://smpybandits.github.io/docs/Policies.Monitored_UCB.html), [SlidingWindowUCB](https://smpybandits.github.io/docs/Policies.SlidingWindowUCB.html) or [Discounted-UCB](https://smpybandits.github.io/docs/Policies.DiscountedUCB.html), or [SW-UCB#](https://smpybandits.github.io/docs/Policies.SWHash_UCB.html).

----

## Other remarks
- Everything here is done in an imperative, object oriented style. The API of the Arms, Policy and MultiPlayersPolicy classes is documented [in this file (`API.md`)](API.md).
- The code is [clean](logs/main_pylint_log.txt), valid for both [Python 2](logs/main_pylint_log.txt) and [Python 3](logs/main_pylint3_log.txt).
- Some piece of code come from the [pymaBandits](http://mloss.org/software/view/415/) project, but most of them were refactored. Thanks to the initial project!
- [G.Varoquaux](http://gael-varoquaux.info/)'s [joblib](https://joblib.readthedocs.io/) is used for the [`Evaluator`](https://smpybandits.github.io/docs/Environment.Evaluator.html) and [`EvaluatorMultiPlayers`](https://smpybandits.github.io/docs/Environment.EvaluatorMultiPlayers.html) classes, so the simulations are easily parallelized on multi-core machines. (Put `n_jobs = -1` or `PARALLEL = True` in the config file to use all your CPU cores, as it is by default).

## [How to run the experiments ?](How_to_run_the_code.md)
> See this document: [`How_to_run_the_code.md`](How_to_run_the_code.md) for more details (or [this documentation page](How_to_run_the_code.html)).

TL;DR: this short bash snippet shows how to clone the code, install the requirements for Python 3 (in a [virtualenv](https://virtualenv.pypa.io/en/stable/), and starts some simulation for N=100 repetitions of the default non-Bayesian Bernoulli-distributed problem, for K=9 arms, an horizon of T=10000 and on 4 CPUs (it should take about 20 minutes for each simulations):

```bash
cd /tmp/  # or wherever you want
git clone -c core.symlinks=true https://GitHub.com/SMPyBandits/SMPyBandits.git
cd SMPyBandits
# just be sure you have the latest virtualenv from Python 3
sudo pip3 install --upgrade --force-reinstall virtualenv
# create and active the virtualenv
virtualenv venv
. venv/bin/activate
type pip  # check it is /tmp/SMPyBandits/venv/bin/pip
type python  # check it is /tmp/SMPyBandits/venv/bin/python
# install the requirements in the virtualenv
pip install -r requirements_full.txt
# run a single-player simulation!
N=100 T=10000 K=9 N_JOBS=4 make single
# run a multi-player simulation!
N=100 T=10000 M=3 K=9 N_JOBS=4 make more
```

You can also install it directly with [`pip`](https://pip.pypa.io/) and from GitHub:

```bash
cd /tmp/ ; mkdir SMPyBandits ; cd SMPyBandits/
virtualenv venv
. venv/bin/activate
type pip  # check it is /tmp/SMPyBandits/venv/bin/pip
type python  # check it is /tmp/SMPyBandits/venv/bin/python
pip install git+https://github.com/SMPyBandits/SMPyBandits.git#egg=SMPyBandits[full]
```

> - If speed matters to you and you want to use algorithms based on [kl-UCB](https://smpybandits.github.io/docs/Policies.klUCB.html), you should take the time to build and install the fast C implementation of the utilities KL functions. Default is to use [kullback.py](https://smpybandits.github.io/docs/Policies.kullback.html), but using [the C version from Policies/C/](Policies/C/) really speeds up the computations. Just follow the instructions, it should work well (you need `gcc` to be installed).
> - And if speed matters, be sure that you have a working version of [Numba](https://numba.pydata.org/), it is used by many small functions to (try to automatically) speed up the computations.

----

### :boom: Warning
- This work is still **experimental** even if [it is well tested and stable](https://travis-ci.org/SMPyBandits/SMPyBandits)! It's [active research](https://github.com/SMPyBandits/SMPyBandits/graphs/contributors). It should be completely bug free and every single module/file should work perfectly (as [this pylint log](main_pylint_log.txt) and [this other one](main_pylint3_log.txt) says), but bugs are sometimes hard to spot so if you encounter any issue, [please fill a bug ticket](https://github.com/SMPyBandits/SMPyBandits/issues/new).
- Whenever I add a new feature, I run experiments [to check that nothing is broken](https://travis-ci.org/SMPyBandits/SMPyBandits) (and [Travis CI](https://travis-ci.org/SMPyBandits/SMPyBandits) helps too). But *there is no unittest* (I don't have time). You would have to trust me :sunglasses:!
- This project is NOT meant to be a library that you can use elsewhere, but a research tool.

## Contributing?
> I don't except issues or pull requests on this project, but you are welcome to.

Contributions (issues, questions, pull requests) are of course welcome, but this project is and will stay a personal environment designed for quick research experiments, and will never try to be an industry-ready module for applications of Multi-Armed Bandits algorithms.
If you want to contribute, please have a look to the [CONTRIBUTING.md](.github/CONTRIBUTING.md) file, and if you want to be more seriously involved, read the [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) file.

- You are welcome to [submit an issue](https://github.com/SMPyBandits/SMPyBandits/issues/new), if it was not previously answered,
- If you have interesting example of use of SMPyBandits, please share it! ([Jupyter Notebooks](https://www.jupyter.org/) are preferred). And fill a pull request to [add it to the notebooks examples](notebooks/).

## :boom: [TODO](TODO.md)
> See this file [`TODO.md`](TODO.md), and [the issues on GitHub](https://github.com/SMPyBandits/SMPyBandits/issues).

----

## :scroll: License ? [![GitHub license](https://img.shields.io/github/license/SMPyBandits/SMPyBandits.svg)](https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen), with help [from contributors](https://github.com/SMPyBandits/SMPyBandits/graphs/contributors).

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel)](https://GitHub.com/SMPyBandits/SMPyBandits/)
[![![PyPI version](https://img.shields.io/pypi/v/smpybandits.svg)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![![PyPI implementation](https://img.shields.io/pypi/implementation/smpybandits.svg)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![![PyPI pyversions](https://img.shields.io/pypi/pyversions/smpybandits.svg?logo=python)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![![PyPI download](https://img.shields.io/pypi/dm/smpybandits.svg)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![![PyPI status](https://img.shields.io/pypi/status/smpybandits.svg)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![Documentation Status](https://readthedocs.org/projects/smpybandits/badge/?version=latest)](https://SMPyBandits.ReadTheDocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/SMPyBandits/SMPyBandits.svg?branch=master)](https://travis-ci.org/SMPyBandits/SMPyBandits)

[![Stars of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/stars/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/stargazers) [![Contributors of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/contributors/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/graphs/contributors) [![Watchers of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/watchers/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/watchers) [![Forks of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/forks/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/network/members)

[![Releases of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/release/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/releases)
[![Commits of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/commits/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/commits/master) / [![Date of last commit of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/last-commit/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/commits/master)

[![Issues of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/issues/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/issues) : [![Open issues of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/open-issues/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/issues?q=is%3Aopen+is%3Aissue) / [![Closed issues of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/closed-issues/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/issues?q=is%3Aclosed+is%3Aissue)

[![Pull requests of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/prs/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/pulls) : [![Open pull requests of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/open-prs/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/issues?q=is%3Aopen+is%3Apr) / [![Closed pull requests of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/closed-prs/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/issues?q=is%3Aclose+is%3Apr)

[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://GitHub.com/Naereen/)
