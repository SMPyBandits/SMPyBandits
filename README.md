# *SMPyBandits*
**Open-Source Python package for Single- and Multi-Players multi-armed Bandits algorithms**.

<img width="50%" src="logo_large.png" align="right"/>

This repository contains the code of [Lilian Besson's](http://perso.crans.org/besson/) numerical environment, written in [Python (2 or 3)](https://www.python.org/), for numerical simulations on :slot_machine: *single*-player and *multi*-players [Multi-Armed Bandits (MAB)](https://en.wikipedia.org/wiki/Multi-armed_bandit) algorithms.

A complete Sphinx-generated documentation is on [SMPyBandits.GitHub.io](https://smpybandits.github.io/).

## Quick presentation

It contains the most complete collection of single-player (classical) bandit algorithms on the Internet ([over 65!](SMPyBandits/Policies/)), as well as implementation of all the state-of-the-art [multi-player algorithms](SMPyBandits/PoliciesMultiPlayers/).

I follow very actively the latest publications related to Multi-Armed Bandits (MAB) research, and usually implement quite quickly the new algorithms (see for instance, [Exp3++](https://smpybandits.github.io/docs/Policies.Exp3PlusPlus.html), [CORRAL](https://smpybandits.github.io/docs/Policies.CORRAL.html) and [SparseUCB](https://smpybandits.github.io/docs/Policies.SparseUCB.html) were each introduced by articles ([for Exp3++](https://arxiv.org/pdf/1702.06103), [for CORRAL](https://arxiv.org/abs/1612.06246v2), [for SparseUCB](https://arxiv.org/abs/1706.01383)) presented at COLT in July 2017, [LearnExp](https://smpybandits.github.io/docs/Policies.LearnExp.html) comes from a [NIPS 2017 paper](https://arxiv.org/abs/1702.04825), and [kl-UCB++](https://smpybandits.github.io/docs/Policies.klUCBPlusPlus.html) from an [ALT 2017 paper](https://hal.inria.fr/hal-01475078).).

- Classical MAB have a lot of applications, from clinical trials, A/B testing, game tree exploration, and online content recommendation (my framework does *not* implement contextual bandit - yet).
- [Multi-player MAB](MultiPlayers.md) have applications in Cognitive Radio, and my framework implements [all the collision models](SMPyBandits/Environment/CollisionModels.py) found in the literature, as well as all the algorithms from the last 10 years or so ([`rhoRand`](SMPyBandits/PoliciesMultiPlayers/rhoRand.py) from 2009, [`MEGA`](SMPyBandits/Policies/MEGA.py) from 2015, [`MusicalChair`](SMPyBandits/Policies/MusicalChair.py), and our state-of-the-art algorithms [`RandTopM`](SMPyBandits/PoliciesMultiPlayers/RandTopM.py) and [`MCTopM`](SMPyBandits/PoliciesMultiPlayers/MCTopM.py)).

With this numerical framework, simulations can run on a single CPU or a multi-core machine, and summary plots are automatically saved as high-quality PNG, PDF and EPS (ready for being used in research article).
Making new simulations is very easy, one only needs to write a configuration script and basically no code! See [these examples](https://github.com/SMPyBandits/SMPyBandits/search?l=Python&q=configuration&type=&utf8=%E2%9C%93) (files named `configuration_...py`).

A complete [Sphinx](http://sphinx-doc.org/) documentation for each algorithms and every piece of code (included constants in the configurations!) is available here: [SMPyBandits.GitHub.io](https://smpybandits.github.io/). (I will use [ReadTheDocs](https://readthedocs.org/) for this project, but I won't use any *continuous integration*, don't even think of it!)

![PyPI version](https://img.shields.io/pypi/v/smpybandits.svg)
![PyPI implementation](https://img.shields.io/pypi/implementation/smpybandits.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/smpybandits.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Documentation Status](https://readthedocs.org/projects/smpybandits/badge/?version=latest)](https://SMPyBandits.ReadTheDocs.io/en/latest/?badge=latest)

> [I (Lilian Besson)](http://perso.crans.org/besson/) have [started my PhD](http://perso.crans.org/besson/phd/) in October 2016, and this is a part of my **on going** research since December 2016.
>
> I launched the [documentation](https://smpybandits.github.io/) on March 2017, I wrote my first research articles using this framework in 2017 and decided to (finally) open-source my project in February 2018.

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

> A DOI will arrive as soon as possible! I will try to publish [a paper](paper.md) on both [JOSS](http://joss.theoj.org/) and [MLOSS](http://mloss.org/software/).

## [List of research publications using SMPyBandits](PublicationsWithSMPyBandits.md)

### 1st article, using the [**policy aggregation algorithm**](Aggregation.md)
I designed and added the [`Aggregator`](SMPyBandits/Policies/Aggregator.py) policy, in order to test its validity and performance.

It is a "simple" **voting algorithm to combine multiple bandit algorithms into one**.
Basically, it behaves like a simple MAB bandit just based on empirical means (even simpler than UCB), where *arms* are the child algorithms `A_1 .. A_N`, each running in "parallel".

> **For more details**, refer to this file: [`Aggregation.md`](Aggregation.md) and [this research article](https://hal.inria.fr/hal-01705292).

----

### 2nd article, using [**Multi-players simulation environment**](MultiPlayers.md)
There is another point of view: instead of comparing different single-player policies on the same problem, we can make them play against each other, in a multi-player setting.
The basic difference is about **collisions** : at each time `t`, if two or more user chose to sense the same channel, there is a *collision*. Collisions can be handled in different way from the base station point of view, and from each player point of view.

> **For more details**, refer to this file: [`MultiPlayers.md`](MultiPlayers.md) and [this research article](https://hal.inria.fr/hal-01629733).

----

### 3rd article, using [**Doubling Trick for Multi-Armed Bandits**](DoublingTrick.md)
I studied what Doubling Trick can and can't do to obtain efficient anytime version of non-anytime optimal Multi-Armed Bandits algorithms.

> **For more details**, refer to this file: [`DoublingTrick.md`](DoublingTrick.md) and [this research article](https://hal.inria.fr/hal-XXX).

----

## Other interesting things
### [Single-player Policies](https://smpybandits.github.io/docs/Policies.html)
- More than 65 algorithms, including all known variants of the [`UCB`](SMPyBandits/Policies/UCB.py), [kl-UCB](SMPyBandits/Policies/klUCB.py), [`MOSS`](SMPyBandits/Policies/MOSS.py) and [Thompson Sampling](SMPyBandits/Policies/Thompson.py) algorithms, as well as other less known algorithms ([`OCUCB`](SMPyBandits/Policies/OCUCB.py), [`BESA`](SMPyBandits/Policies/OCUCB.py), [`OSSB`](SMPyBandits/Policies/OSSB.py) etc).
- [`SparseWrapper`](https://smpybandits.github.io/docs/Policies.SparseWrapper.html#module-Policies.SparseWrapper) is a generalization of [the SparseUCB from this article](https://arxiv.org/pdf/1706.01383/).
- Implementation of very recent Multi-Armed Bandits algorithms, e.g., [`kl-UCB++`](https://smpybandits.github.io/docs/Policies.klUCBPlusPlus.html) (from [this article](https://hal.inria.fr/hal-01475078)), [`UCB-dagger`](https://smpybandits.github.io/docs/Policies.UCBdagger.html) (from [this article](https://arxiv.org/pdf/1507.07880)),  or [`MOSS-anytime`](https://smpybandits.github.io/docs/Policies.MOSSAnytime.html) (from [this article](http://proceedings.mlr.press/v48/degenne16.pdf)).
- Experimental policies: [`BlackBoxOpt`](https://smpybandits.github.io/docs/Policies.BlackBoxOpt.html) or [`UnsupervisedLearning`](https://smpybandits.github.io/docs/Policies.UnsupervisedLearning.html) (using Gaussian processes to learn the arms distributions).

### Arms and problems
- My framework mainly targets stochastic bandits, with arms following [`Bernoulli`](SMPyBandits/Arms/Bernoulli.py), bounded (SMPyBandits/truncated) or unbounded [`Gaussian`](Arms/Gaussian.py), [`Exponential`](SMPyBandits/Arms/Exponential.py), [`Gamma`](SMPyBandits/Arms/Gamma.py) or [`Poisson`](SMPyBandits/Arms/Poisson.py) distributions.
- The default configuration is to use a fixed problem for N repetitions (e.g. 1000 repetitions, use [`MAB.MAB`](SMPyBandits/Environment/MAB.py)), but there is also a perfect support for "Bayesian" problems where the mean vector µ1,…,µK change *at every repetition* (see [`MAB.DynamicMAB`](SMPyBandits/Environment/MAB.py)).
- There is also a good support for Markovian problems, see [`MAB.MarkovianMAB`](SMPyBandits/Environment/MAB.py), even though I didn't implement any policies tailored for Markovian problems.

----

## Other remarks
- Everything here is done in an imperative, object oriented style. The API of the Arms, Policy and MultiPlayersPolicy classes is documented [in this file (`API.md`)](API.md).
- The code is [clean](logs/main_pylint_log.txt), valid for both [Python 2](logs/main_pylint_log.txt) and [Python 3](logs/main_pylint3_log.txt).
- Some piece of code come from the [pymaBandits](http://mloss.org/software/view/415/) project, but most of them were refactored. Thanks to the initial project!
- [G.Varoquaux](http://gael-varoquaux.info/)'s [joblib](https://pythonhosted.org/joblib/) is used for the [`Evaluator`](SMPyBandits/Environment/Evaluator.py) and [`EvaluatorMultiPlayers`](SMPyBandits/Environment/EvaluatorMultiPlayers.py) classes, so the simulations are easily parallelized on multi-core machines. (Put `n_jobs = -1` or `PARALLEL = True` in the config file to use all your CPU cores, as it is by default).

## [How to run the experiments ?](How_to_run_the_code.md)
> See this document: [`How_to_run_the_code.md`](How_to_run_the_code.md) for more details (or [this documentation page](How_to_run_the_code.html)).

TL;DR: this short bash snippet shows how to clone the code, install the requirements for Python 3 (in a [virtualenv](https://virtualenv.pypa.io/en/stable/), and starts some simulation for N=100 repetitions of the default non-Bayesian Bernoulli-distributed problem, for K=9 arms, an horizon of T=10000 and on 4 CPUs (it should take about 20 minutes for each simulations):

```bash
cd /tmp/  # or wherever you want
git clone https://GitHub.com/SMPyBandits/SMPyBandits.git
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

> - If speed matters to you and you want to use algorithms based on [kl-UCB](SMPyBandits/Policies/klUCB.py), you should take the time to build and install the fast C implementation of the utilities KL functions. Default is to use [kullback.py](SMPyBandits/Policies/kullback.py), but using [the C version from Policies/C/](Policies/C/) really speeds up the computations. Just follow the instructions, it should work well (you need `gcc` to be installed).
> - And if speed matters, be sure that you have a working version of [Numba](https://numba.pydata.org/), it is used by many small functions to (try to automatically) speed up the computations.

----

### :boom: Warning
- This work is still **experimental**! It's [active research](https://github.com/SMPyBandits/SMPyBandits/graphs/contributors). It should be completely bug free and every single module/file should work perfectly(as [this pylint log](main_pylint_log.txt) and [this other one](main_pylint3_log.txt) says), but bugs are sometimes hard to spot so if you encounter any issue, [please fill a bug ticket](https://github.com/SMPyBandits/SMPyBandits/issues/new).
- Whenever I add a new feature, I run experiments to check that nothing is broken. But *there is no unittest* (I don't have time). You would have to trust me :sunglasses:!
- This project is NOT meant to be a library that you can use elsewhere, but a research tool. In particular, I don't take ensure that any of the Python modules can be imported from another directory than the main directory.

## Contributing?
> I don't except issues or pull requests on this project, but you are welcome to.

Contributions (issues, questions, pull requests) are of course welcome, but this project is and will stay a personal environment designed for quick research experiments, and will never try to be an industry-ready module for applications of Multi-Armed Bandits algorithms.

If you want to contribute, please have a look to the [CONTRIBUTING.md](CONTRIBUTING.md) file, and if you want to be more seriously involved, read the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) file.

- You are welcome to [submit an issue](https://github.com/SMPyBandits/SMPyBandits/issues/new), if it was not previously answered,
- If you have interesting example of use of SMPyBandits, please share it! ([Jupyter Notebooks](https://www.jupyter.org/) are preferred). And fill a pull request to [add it to the notebooks examples](notebooks/).

## :boom: [TODO](TODO.md)
> See this file [`TODO.md`](TODO.md), and [the issues on GitHub](https://github.com/SMPyBandits/SMPyBandits/issues).

----

## :scroll: License ? [![GitHub license](https://img.shields.io/github/license/SMPyBandits/SMPyBandits.svg)](https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel)](https://GitHub.com/SMPyBandits/SMPyBandits/)
![PyPI version](https://img.shields.io/pypi/v/smpybandits.svg)
![PyPI implementation](https://img.shields.io/pypi/implementation/smpybandits.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/smpybandits.svg)
[![Documentation Status](https://readthedocs.org/projects/smpybandits/badge/?version=latest)](https://SMPyBandits.ReadTheDocs.io/en/latest/?badge=latest)
[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://GitHub.com/Naereen/)
