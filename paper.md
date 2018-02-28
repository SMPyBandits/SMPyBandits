---
title: "*SMPyBandits*: a Research Framework for Single and Multi-Players Multi-Arms Bandits Algorithms in Python"
authors:
- name: Lilian Besson
  orcid: 0000-0003-2767-2563
  affiliation:
    "1, 2"
  email: Lilian.Besson[AT]CentraleSupelec[.]fr
affiliations:
  - name: CentraleSupélec, campus of Rennes, SCEE team
    index: 1
  - name: Inria Lille Nord Europe, SequeL team.
    index: 2
tags:
- sequential learning
- multi-arm bandits
- multi-player multi-arm bandits
- aggregation of sequential learning algorithms
- learning theory
date: 22 February 2018
bibliography: paper.bib
---

# Summary

*SMPyBandits* is a package for numerical simulations on *single*-player and *multi*-players [Multi-Armed Bandits (MAB)](https://en.wikipedia.org/wiki/Multi-armed_bandit) algorithms [@Bubeck12], written in [Python (2 or 3)](https://www.python.org/) [@python].

*SMPyBandits* is the most complete open-source implementation of state-of-the-art algorithms tackling various kinds of sequential learning problems referred to as Multi-Armed Bandits.
It aims at being extensive, simple to use and maintain, with a clean and perfectly documented codebase.
It allows fast prototyping of simulations and experiments, with an easy configuration system and command-line options to customize experiments while starting them (see below for an example).

---


## Presentation

### Single-Player MAB
Multi-Armed Bandit (MAB) problems are well-studied sequential decision making problems in which an agent repeatedly chooses an action (the "*arm*" of a one-armed bandit) in order to maximize some total reward [@Robbins52,LaiRobbins85]. Initial motivation for their study came from the modeling of clinical trials, as early as 1933 with the seminal work of Thompson  [@Thompson33], where arms correspond to different treatments with unknown, random effect. Since then, MAB models have been proved useful for many more applications, that range from cognitive radio [@Jouini09] to online content optimization (news article recommendation [@Li10], online advertising [@LiChapelle11] or A/B Testing [@Kaufmann14;Jamieson17]), or portfolio optimization [@Sani12].

*SMPyBandits* is the most complete open-source implementation of single-player (classical) bandit algorithms ([over 65!](https://smpybandits.github.io/docs/Policies.html)).
We use a well-designed hierarchical structure and [class inheritance scheme](https://smpybandits.github.io/uml_diagrams/README.html) to minimize redundancy in the codebase.
Most existing algorithms are index-based, and can be written very shortly by inheriting from the [`IndexPolicy`](https://smpybandits.github.io/docs/Policies.IndexPolicy.html) class.

### Multi-Players MAB

For Cognitive Radio applications, a well-studied extension is to consider $M\geq2$ players, interacting on the *same* $K$ arms. Whenever two or more players select the same arm at the same time, they all suffer from a collision.
Different collision models has been proposed, and the simplest one consist in giving a $0$ reward to each colliding players.
Without any centralized supervision or coordination between players, they must learn to access the $M$ best resources (*i.e.*, arms with highest means) without collisions.

*SMPyBandits* implements [all the collision models](https://smpybandits.github.io/docs/Environment.CollisionModels.py) found in the literature, as well as all the algorithms from the last 10 years or so (including [`rhoRand`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.rhoRand.py) from 2009, [`MEGA`](https://smpybandits.github.io/docs/Policies.MEGA.py) from 2015, [`MusicalChair`](https://smpybandits.github.io/docs/Policies.MusicalChair.py) from 2016, and our state-of-the-art algorithms [`RandTopM`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.RandTopM.py) and [`MCTopM`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.MCTopM.py)) from [@BessonALT2018].

---

## Features

With this numerical framework, simulations can run on a single CPU or a multi-core machine using joblib [@joblib],
and summary plots are automatically saved as high-quality PNG, PDF and EPS (ready for being used in research article), using matplotlib [@matplotlib] and seaborn [@seaborn].
Making new simulations is very easy, one only needs to write a configuration script and no knowledge of the internal code architecture.

### Documentation

A complete sphinx [@sphinx] documentation for each algorithms and every piece of code, included the constants in the different configuration files, is available here: [`https://smpybandits.github.io`](https://smpybandits.github.io/).

### How to run the experiments ?

For example, this short bash snippet [^docforconf] shows how to clone the code, install the requirements for Python 3 (in a virtualenv [@virtualenv]), and starts some simulation for $N=1000$ repetitions of the default non-Bayesian Bernoulli-distributed problem, for $K=9$ arms, an horizon of $T=10000$ and on $4$ CPUs [^speedofsimu].
Using environment variables (`N=1000`) when launching the simulation is not required but it is convenient.

[^docforconf]:  See [this page of the documentation](https://smpybandits.github.io/How_to_run_the_code.html) for more details.
[^speedofsimu]:  It takes about $20$ to $40$ minutes for each simulation, on a standard $4$-cores $64$ bits GNU/Linux laptop.

```bash
# 1. get the code in /tmp/, or wherever you want
cd /tmp/
git clone https://GitHub.com/SMPyBandits/SMPyBandits.git
cd SMPyBandits.git
# 2. just be sure you have the latest virtualenv from Python 3
sudo pip3 install --upgrade virtualenv
# 3. create and active the virtualenv
virtualenv3 venv || virtualenv venv
. venv/bin/activate
# 4. install the requirements in the virtualenv
pip3 install -r requirements.txt
# 5. run a single-player simulation!
N=1000 T=10000 K=9 N_JOBS=4 make single
```

### Example of simulation and illustration

A small script [`configuration.py`](https://smpybandits.github.io/docs/configuration.html) is used to import the [arm classes](https://smpybandits.github.io/docs/Arms.html), the [policy classes](https://smpybandits.github.io/docs/Policies.html) and define the problems and the experiments.
For instance, we can compare the standard anytime [`klUCB`](https://smpybandits.github.io/docs/Policies.klUCB.py) algorithm against the non-anytime variant [`klUCBPlusPlus`](https://smpybandits.github.io/docs/Policies.klUCBPlusPlus.py) algorithm, as well as [`UCB`](https://smpybandits.github.io/docs/Policies.UCBalpha.py) (with $\alpha=1$) and [`Thompson`](https://smpybandits.github.io/docs/Policies.Thompson.py) (with [Beta posterior](https://smpybandits.github.io/docs/Policies.Posterior.Beta.html)).
See below in Figure \ref{fig:plot1} for the result showing the average regret [^regret] for these $4$ algorithms.

[^regret]:  The regret is the difference between the cumulated rewards of the best fixed-armed strategy (which is the oracle strategy for stationary bandits) and the cumulated rewards of the considered algorithms.

![Single-player simulation showing the regret of $4$ algorithms, and the asymptotic lower-bound from [@LaiRobbins85]. They all perform very well, and at finite time they are empirically *below* the asymptotic lower-bound. Each algorithm is known to be order-optimal (*i.e.*, its regret is proved to match the lower-bound up-to a constant), and each but UCB is known to be optimal (*i.e.* with the constant matching the lower-bound).\label{fig:plot1}](plots/paper/1.png){ width=95% }

---

## Research using *SMPyBandits*

*SMPyBandits* was used for the following research articles since $2017$:

- For [@BessonALT2018], we used *SMPyBandits* for all the simulations for multi-player bandit algorithms [^article1]. We designed the two [`RandTopM`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.RandTopM.html) and [`MCTopM`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.MCTopM.html) algorithms and proved than they enjoy logarithmic regret in the usual setting, and outperform significantly the previous state-of-the-art solutions (*i.e.*, [`rhoRand`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.rhoRand.html), [`MEGA`](https://smpybandits.github.io/docs/Policies.MEGA.html) and [`MusicalChair`](https://smpybandits.github.io/docs/Policies.MusicalChair.html)).

[^article1]:  See [`MultiPlayers`](https://smpybandits.github.io/MultiPlayers.html) on the documentation.

- In [@BessonWCNC2018], we used *SMPyBandits* to illustrate and compare different aggregation algorithms [^article2]. We designed a variant of the Exp3 algorithm for online aggregation of experts [@Bubeck12], called [`Aggregator`](https://smpybandits.github.io/docs/Policies.Aggregator.html). Aggregating experts is a well-studied idea in sequential learning and in machine learning in general. We showed that it can be used in practice to select on the run the best bandit algorithm for a certain problem from a fixed pool of experts. This idea and algorithm can have interesting impact for Opportunistic Spectrum Access applications [@Jouini09] that use multi-armed bandits algorithms for sequential learning and network efficiency optimization.

[^article2]:  See [`Aggregation`](https://smpybandits.github.io/Aggregation.html) on the documentation.

- In [@Besson2018c], we used *SMPyBandits* to illustrate and compare different "doubling trick" schemes [^article3]. In sequential learning, an algorithm is *anytime* if it does not need to know the horizon $T$ of the experiments. A well-known trick for transforming any non-anytime algorithm to an anytime variant is the "Doubling Trick": start with an horizon $T_0\in\mathbb{N}$, and when $t > T_i$, use $T_{i+1} = 2 T_i$. We studied two generic sequences of growing horizons (geometric and exponential), and we proved two theorems that generalized previous results. A geometric sequence suffices to minimax regret bounds (in $R_T = \mathcal{O}(\sqrt(T))$), with a constant multiplicative loss $\ell \leq 4$, but cannot be used to conserve a logarithmic regret bound (in $R_T = \mathcal{O}(\log(T))$). And an exponential sequence can be used to conserve logarithmic bounds, with a constant multiplicative loss also $\ell \leq 4$ in the usual setting. It is still an open question to know if a well-tuned exponential sequence can conserve minimax bounds or weak minimax bounds (in $R_T = \mathcal{O}(\sqrt{T \log(T)})$).

[^article3]:  See [`DoublingTrick`](https://smpybandits.github.io/DoublingTrick.html) on the documentation.


## Dependencies
Written in Python [@python], using matplotlib [@matplotlib] for 2D plotting, numpy [@numpy] for data storing, random number generations and and operations on arrays, scipy [@scipy] for statistical and special functions, and seaborn [@seaborn] for pretty plotting and colorblind-aware colormaps.
Optional dependencies include joblib [@joblib] for parallel simulations, numba [@numba] for automatic speed-up on small functions, sphinx [@sphinx] for generating the documentations, virtualenv [@virtualenv] for launching simulations in isolated environments, and jupyter [@jupyter] used with ipython [@ipython] to experiment with the code.

# References
