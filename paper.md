---
title: "AlgoBandits: an Open-Source Research Framework for Single and Multi-Players Multi-Arms Bandits (MAB) Algorithms in Python"
tags:
- sequential learning
- multi-arm bandits
- multi-player multi-arm bandits
- aggregation of sequential learning algorithms
- learning theory
authors:
- name: Lilian Besson
  orcid: 0000-0003-2767-2563
  affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
- name: CentraleSupélec, campus de Rennes, équipe SCEE
  index: 1
- name: Inria Lille Nord Europe, équipe SequeL
  index: 2
date: 22 February 2018
bibliography: paper.bib
---

# Summary

This article presents [my](http://perso.crans.org/besson/) numerical environment *AlgoBandits*, written in [Python (2 or 3)](https://www.python.org/) [@Python], for numerical simulations on *single*-player and *multi*-players [Multi-Armed Bandits (MAB)](https://en.wikipedia.org/wiki/Multi-armed_bandit) algorithms [@Bubeck12].

*AlgoBandits* is the most complete open-source implementation of state-of-the-art algorithms tackling various kinds of sequential learning problems referred to as Multi-Armed Bandits.
It aims at being extensive, simple to use and maintain, with a clean and perfectly documented codebase. But most of all it allows fast prototyping of simulations and experiments, with an easy configuration system and command-line options to customize experiments while starting them (see below for an example).

*AlgoBandits* does not aim at being blazing fast or perfectly memory efficient, and comes with a pure Python implementation with no dependency except standard open-source Python packages.
Even if critical parts are coded in C or use Numba [@numba], if speed matters one should rather refer to less exhaustive but faster implementations, like [@tor_libbandit] in `C++` or [@vish_MAB_jl] in Julia.

---


## Quick presentation

### Single-Player MAB
Multi-Armed Bandit (MAB) problems are well-studied sequential decision making problems in which an agent repeatedly chooses an action (the "arm" of a one-armed bandit) in order to maximize some total reward [@Robbins52,LaiRobbins85]. Initial motivation for their study came from the modeling of clinical trials, as early as 1933 with the seminal work of [@Thompson33]. In this example, arms correspond to different treatments with unknown, random effect. Since then, MAB models have been proved useful for many more applications, that range from cognitive radio [@Jouini09] to online content optimization (news article recommendation [@Li10contextual], online advertising [@LiChapelle11] or A/B Testing [@Kaufmann14,Jamieson17ABTest]), or portfolio optimization [@Sani2012risk].

This Python package contains the most complete collection of single-player (classical) bandit algorithms on the Internet ([over 65!](http://banditslilian.gforge.inria.fr/docs/Policies.html)).
We use a well-designed hierarchical structure and class inheritance scheme to minimize redundancy in the codebase, and for instance the code specific to the UCB algorithm [@LaiRobbins85,@Auer02] is as short as this (and fully documented), by inheriting from a generic [`IndexPolicy`](http://banditslilian.gforge.inria.fr/docs/Policies.IndexPolicy.html) class:

```python
import numpy as np
from .IndexPolicy import IndexPolicy

class UCB(IndexPolicy):
  """ The UCB policy for bounded bandits. Reference: [Lai & Robbins, 1985]. """

  def computeIndex(self, arm):
    r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

    .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{2 \log(t)}{N_k(t)}}.
    """
    if self.pulls[arm] < 1:
      return float('+inf')
    else:
      estimated_mean = (self.rewards[arm] / self.pulls[arm])
      observation_bias = np.sqrt((2 * np.log(self.t)) / self.pulls[arm])
      return estimated_mean + observation_bias

```

### Multi-Players MAB

For Cognitive Radio applications, a well-studied extension is to consider $M\geq2$ players, interacting on the same $K$ arms. Whenever two or more players select the same arm at the same time, they all suffer from a collision.
Different collision models has been proposed, and the simplest one consist in giving a $0$ reward to each colliding players.
Without any centralized supervision or coordination between players, they must learn to access the $M$ best resources (*i.e.*, arms with highest means) without collisions.

This package implements [all the collision models](http://banditslilian.gforge.inria.fr/docs/Environment.CollisionModels.py) found in the literature, as well as all the algorithms from the last 10 years or so ([`rhoRand`](http://banditslilian.gforge.inria.fr/docs/PoliciesMultiPlayers.rhoRand.py) from 2009, [`MEGA`](http://banditslilian.gforge.inria.fr/docs/Policies.MEGA.py) from 2015, [`MusicalChair`](http://banditslilian.gforge.inria.fr/docs/Policies.MusicalChair.py), and our state-of-the-art algorithms [`RandTopM`](http://banditslilian.gforge.inria.fr/docs/PoliciesMultiPlayers.RandTopM.py) and [`MCTopM`](http://banditslilian.gforge.inria.fr/docs/PoliciesMultiPlayers.MCTopM.py)) from [@besson:hal-01629733].

---


## Purpose

The main goal of this package is to implement [with the same API](http://banditslilian.gforge.inria.fr/API.html) most of the existing single- and multi-player multi-armed bandit algorithms.
Each algorithm comes with a clean documentation page, containing a reference to the research article(s) that introduced it, and with remarks on its numerical efficiency.

It is neither the first nor the only open-source implementation of multi-armed bandits algorithms, although one can notice the absence of any well-maintained reference implementation.
I built AlgoBandits from a framework called "pymaBandits" [@pymaBandits], which implemented a few algorithms and three kinds of arms

Since November $2016$, I follow actively the latest publications related to Multi-Armed Bandits (MAB) research, and usually I implement quickly any new algorithms. For instance, [Exp3++](http://banditslilian.gforge.inria.fr/docs/Policies.Exp3PlusPlus.html), [CORRAL](http://banditslilian.gforge.inria.fr/docs/Policies.CORRAL.html) and [SparseUCB](http://banditslilian.gforge.inria.fr/docs/Policies.SparseUCB.html) were each introduced by articles ([for Exp3++](https://arxiv.org/pdf/1702.06103), [for CORRAL](https://arxiv.org/abs/1612.06246v2), [for SparseUCB](https://arxiv.org/abs/1706.01383)) presented at COLT in July 2017, [LearnExp](http://banditslilian.gforge.inria.fr/docs/Policies.LearnExp.html) comes from a [NIPS 2017 paper](https://arxiv.org/abs/1702.04825), and [kl-UCB++](http://banditslilian.gforge.inria.fr/docs/Policies.klUCBPlusPlus.html) from an [ALT 2017 paper](https://hal.inria.fr/hal-01475078).

---

## Features

With this numerical framework, simulations can run on a single CPU or a multi-core machine, and summary plots are automatically saved as high-quality PNG, PDF and EPS (ready for being used in research article).
Making new simulations is very easy, one only needs to write a configuration script and basically no code.

### Example of configuration for a simulation

See for example, a simple python file, [`configuration_comparing_doubling_algorithms.py`](http://banditslilian.gforge.inria.fr/docs/configuration_comparing_doubling_algorithms.html), is used to import the [arm classes](http://banditslilian.gforge.inria.fr/docs/Arms.html), the [policy classes](http://banditslilian.gforge.inria.fr/docs/Policies.html) and define the problems and the experiments.
We can compare the standard anytime [`klUCB`](http://banditslilian.gforge.inria.fr/docs/Policies.klUCB.py) algorithm against the non-anytime variant [`klUCBPlusPlus`](http://banditslilian.gforge.inria.fr/docs/Policies.klUCBPlusPlus.py) algorithm, as well as [`UCB`](http://banditslilian.gforge.inria.fr/docs/Policies.UCB.py) and [`Thompson`](http://banditslilian.gforge.inria.fr/docs/Policies.Thompson.py).

```python
configuration = {
  "horizon": 10000,    # Finite horizon of the simulation
  "repetitions": 100,  # Number of repetitions
  "n_jobs": -1,        # Max number of cores for parallelization
  # Environment configuration, you can set up more than one.
  "environment": [
    {
      "arm_type": Bernoulli,
      "probabilities": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
  ],
  # Policies that should be simulated, and their parameters.
  "policies": [
    {"archtype": UCB, "params": {} },
    {"archtype": klUCB, "params": {} },
    {"archtype": Thompson, "params": {} },
    {"archtype": klUCBPlusPlus, "params": {
      "horizon": 10000
    }},
  ]
}
```

### Documentation

A complete Sphinx [@Sphinx] documentation for each algorithms and every piece of code, included the constants in the different configuration files, is available here: [`banditslilian.gforge.inria.fr`](http://banditslilian.gforge.inria.fr/).

### Other noticeable features

#### [Single-player Policies](http://banditslilian.gforge.inria.fr/docs/Policies.html)

- More than 65 algorithms, including all known variants of the [`UCB`](http://banditslilian.gforge.inria.fr/docs/Policies.UCB.py), [kl-UCB](http://banditslilian.gforge.inria.fr/docs/Policies.klUCB.py), [`MOSS`](http://banditslilian.gforge.inria.fr/docs/Policies.MOSS.py) and [Thompson Sampling](http://banditslilian.gforge.inria.fr/docs/Policies.Thompson.py) algorithms, as well as other less known algorithms ([`OCUCB`](http://banditslilian.gforge.inria.fr/docs/Policies.OCUCB.py), [`BESA`](http://banditslilian.gforge.inria.fr/docs/Policies.OCUCB.py), [`OSSB`](http://banditslilian.gforge.inria.fr/docs/Policies.OSSB.py) etc).
- Implementation of very recent Multi-Armed Bandits algorithms, e.g., [`kl-UCB++`](http://banditslilian.gforge.inria.fr/docs/Policies.klUCBPlusPlus.html) (from [this article](https://hal.inria.fr/hal-01475078)), [`UCB-dagger`](http://banditslilian.gforge.inria.fr/docs/Policies.UCBdagger.html),  or [`MOSS-anytime`](http://banditslilian.gforge.inria.fr/docs/Policies.MOSSAnytime.html) (from [this article](http://proceedings.mlr.press/v48/degenne16.pdf)).
- Experimental policies: [`BlackBoxOpt`](http://banditslilian.gforge.inria.fr/docs/Policies.BlackBoxOpt.html) or [`UnsupervisedLearning`](http://banditslilian.gforge.inria.fr/docs/Policies.UnsupervisedLearning.html) (using Gaussian processes to learn the arms distributions).

#### Arms and problems
- My framework mainly target stochastic bandits, with arms following [`Bernoulli`](Arms/Bernoulli.py), bounded (truncated) or unbounded [`Gaussian`](Arms/Gaussian.py), [`Exponential`](Arms/Exponential.py), [`Gamma`](Arms/Gamma.py) or [`Poisson`](Arms/Poisson.py) distributions.
- The default configuration is to use a fixed problem for N repetitions (e.g. 1000 repetitions, use [`MAB.MAB`](Environment/MAB.py)), but there is also a perfect support for "Bayesian" problems where the mean vector µ1,…,µK change *at every repetition* (see [`MAB.DynamicMAB`](Environment/MAB.py)).
- There is also a good support for Markovian problems, see [`MAB.MarkovianMAB`](Environment/MAB.py), even though I didn't implement any policies tailored for Markovian problems.

---

## Other remarks
- Everything here is done in an imperative, object oriented style. The API of the Arms, Policy and MultiPlayersPolicy classes is documented [in this file (`API.md`)](API.md).
- The code is [clean](http://banditslilian.gforge.inria.fr/logs/main_pylint_log.txt), valid for both [Python 2](http://banditslilian.gforge.inria.fr/logs/main_pylint_log.txt) and [Python 3](http://banditslilian.gforge.inria.fr/logs/main_pylint3_log.txt).
- The joblib [@joblib] is used for the [`Evaluator`](http://banditslilian.gforge.inria.fr/docs/Environment.Evaluator.py) and [`EvaluatorMultiPlayers`](http://banditslilian.gforge.inria.fr/docs/Environment.EvaluatorMultiPlayers.py) classes, so the simulations are easily parallelized on multi-core machines and servers. *AlgoBandits* does no use of any optimization using a GPU or for a cluster.

### How to run the experiments ?
> See [this page of the documentation](http://banditslilian.gforge.inria.fr/How_to_run_the_code.html) for more details.

For example, this short bash snippet shows how to clone the code, install the requirements for Python 3 (in a [virtualenv](https://virtualenv.pypa.io/en/stable/), and starts some simulation for $N=100$ repetitions of the default non-Bayesian Bernoulli-distributed problem, for $K=9 arms$, an horizon of $T=10000$ and on $4$ CPUs (it should take about $20$ minutes for each simulations):

```bash
cd /tmp/  # or wherever you want
git clone https://GitHub.com/Naereen/AlgoBandits.git
cd AlgoBandits.git
# just be sure you have the latest virtualenv from Python 3
sudo pip3 install --upgrade virtualenv
# create and active the virtualenv
virtualenv3 venv || virtualenv venv
. venv/bin/activate
# install the requirements in the virtualenv
pip3 install -r requirements.txt
# run a single-player simulation!
N=100 T=10000 K=9 N_JOBS=4 make single
# run a multi-player simulation!
N=100 T=10000 M=3 K=9 N_JOBS=4 make more
```

---

## Research using AlgoBandits

[I (Lilian Besson)](http://perso.crans.org/besson/) have [started my PhD](http://perso.crans.org/besson/phd/) in October 2016, and this is a part of my **on going** research since December 2016.
I launched the [documentation](http://banditslilian.gforge.inria.fr/) on March 2017, I wrote my first research articles using this framework in 2017 and I was finally able to open-source my project in February 2018.

This project was used for the following research articles written in 2017 and 2018:

- [@Bonnefoi17] not to generate the main figures, but to explore on a smaller scale many other approaches (using [`EvaluatorSparseMultiPlayers`](http://banditslilian.gforge.inria.fr/docs/Environment.EvaluatorSparseMultiPlayers.html)).

- [@besson:hal-01629733] for all the simulations for multi-player bandit algorithms (more details on this documentation page, [`MultiPlayers`](http://banditslilian.gforge.inria.fr/MultiPlayers.html)). We designed the two [`RandTopM`](http://banditslilian.gforge.inria.fr/docs/PoliciesMultiPlayers.RandTopM.html) and [`MCTopM`](http://banditslilian.gforge.inria.fr/docs/PoliciesMultiPlayers.MCTopM.html) algorithms and proved than they enjoy logarithmic regret in the usual setting, and always beat the previous state-of-the-art work ([`rhoRand`](http://banditslilian.gforge.inria.fr/docs/PoliciesMultiPlayers.rhoRand.html), [`MEGA`](http://banditslilian.gforge.inria.fr/docs/Policies.MEGA.html) and [`MusicalChair`](http://banditslilian.gforge.inria.fr/docs/Policies.MusicalChair.html)).

- [@besson:hal-01705292] for all the simulations (more details on this documentation page, [`Aggregation`](http://banditslilian.gforge.inria.fr/Aggregation.html)). We designed a variant of the Exp3 algorithm [@Bubeck12], called [`Aggregator`]() and showed that it can be used in practice to select on the run the best bandit algorithm for a certain problem from a fixed pool of experts. This idea and algorithm can have interesting impact for Opportunistic Spectrum Access applications [@Jouini09] that use multi-armed bandits algorithms for sequential learning and network efficiency optimization.

- [@besson:hal-XXX] for all the simulations (more details on this documentation page, [`DoublingTrick`](http://banditslilian.gforge.inria.fr/DoublingTrick.html)).


## Dependencies
The framework is written in [@Python], using [@matplotlib] for 2D plotting, [@numpy] for data storing, random number generations and and operations on arrays, [@scipy] for statistical and special functions, [@seaborn] for pretty default plot configuration, [@joblib] for parallel simulations, [@numba] for automatic speed-up on some small functions, as well as [@Sphinx] for generating the documentations.

---

# References
