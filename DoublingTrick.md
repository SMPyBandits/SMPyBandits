# **Doubling Trick for Multi-Armed Bandits**
I studied what [Doubling Trick](Policies/DoublingTrickWrapper.py) can and can't do for multi-armed bandits, to obtain efficient anytime version of non-anytime optimal Multi-Armed Bandits algorithms.

The [Doubling Trick](Policies/DoublingTrickWrapper.py) algorithm, denoted DT(A, (Ti)) for a diverging increasing sequence Ti, is the following algorithm:

![Policies/DoublingTrick.py](plots/DoublingTrick_algo1.png)

Long story short, we proved the two following theorems.

### For geometric sequences
> It works for minimax regret bounds (in RT = O(sqrt(T))), with a constant multiplicative loss <= 4, but not for logarithmic regret bounds (in RT = O(log(T))).

![https://hal.inria.fr/hal-XXX](plots/DoublingTrick_theorem1.png)

### For exponential sequences
> It works for logarithmic regret bounds (in RT = O(log(T))), but not for minimax regret bounds (in RT = O(sqrt(T))).

![https://hal.inria.fr/hal-XXX](plots/DoublingTrick_theorem2.png)

----

## Article
I wrote a research article on that topic, it will be a better introduction as a self-contained document to explain this idea and the algorithms. Reference: [[What the Doubling Trick Can or Can't Do for Multi-Armed Bandits, Lilian Besson and Emilie Kaufmann, 2018]](https://hal.inria.fr/hal-XXX), to be presented soon.

----

## Configuration
A simple python file, [`configuration_comparing_doubling_algorithms.py`](configuration_comparing_doubling_algorithms.py), is used to import the [arm classes](Arms/), the [policy classes](Policies/) and define the problems and the experiments.

For example, we can compare the standard anytime [`klUCB`](Policies/klUCB.py) algorithm against the non-anytime [`klUCBPlusPlus`](Policies/klUCBPlusPlus.py) algorithm, as well as 3 versions of [`DoublingTrickWrapper`](Policies/DoublingTrickWrapper.py) applied to [`klUCBPlusPlus`](Policies/klUCBPlusPlus.py).

```python
configuration = {
    "horizon": 10000,    # Finite horizon of the simulation
    "repetitions": 100,  # number of repetitions
    "n_jobs": -1,        # Maximum number of cores for parallelization: use ALL your CPU
    "verbosity": 5,      # Verbosity for the joblib calls
    # Environment configuration, you can set up more than one.
    "environment": [
        {
            "arm_type": Bernoulli,
            "probabilities": 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        }
    ],
    # Policies that should be simulated, and their parameters.
    "policies": [
        {"archtype": UCB, "params": {} },
        {"archtype": klUCB, "params": {} },
        {"archtype": klUCBPlusPlus, "params": { "horizon": 10000 } },
    ]
}
```

Then add a [Doubling-Trick](Policies/DoublingTrickWrapper.py) bandit algorithm ([`DoublingTrickWrapper` class](Policies/DoublingTrickWrapper.py)), you can use this piece of code:

```python
configuration["policies"] += [
    {
        "archtype": DoublingTrickWrapper,
        "params": {
            "next_horizon": next_horizon,
            "full_restart": full_restart,
            "policy": klUCBPlusPlus,
        }
    }
    for full_restart in [ True, False ]
    for next_horizon in [
        next_horizon__arithmetic,
        next_horizon__geometric,
        next_horizon__exponential_fast,
        next_horizon__exponential_slow,
        next_horizon__exponential_generic
    ]
]
```

----

## [How to run the experiments ?](How_to_run_the_code.md)

You should use the provided [`Makefile`](Makefile) file to do this simply:
```bash
make install  # install the requirements ONLY ONCE
make comparing_doubling_algorithms   # run and log the main.py script
```

----

## Some illustrations
Here are some plots illustrating the performances of the different [policies](Policies/) implemented in this project, against various problems (with [`Bernoulli`](Arms/Bernoulli.py) and [`UnboundedGaussian`](Arms/Gaussian.py) arms only):

### [Doubling-Trick](Policies/DoublingTrickWrapper.py) with restart, on a "simple" Bernoulli problem
![Doubling-Trick with restart, on a "simple" Bernoulli problem](plots/main____env1-1_1217677871459230631.png)

Regret for [Doubling-Trick](Policies/DoublingTrickWrapper.py), for K=9 [Bernoulli arms](Arms/Bernoulli.py), horizon T=45678, n=1000 repetitions and µ1,…,µK taken uniformly in [0,1]^K.
Geometric doubling (b=2) and slow exponential doubling (b=1.1) are too slow, and short first sequences make the regret blow up in the beginning of the experiment.
At t=40000 we see clearly the effect of a new sequence for the best [doubling trick](Policies/DoublingTrickWrapper.py) (T_i = 200 x 2^i).
As expected, [kl-UCB++](Policies/klUCBPlusPlus.py) outperforms [kl-UCB](Policies/klUCB.py), and if the doubling sequence is growing fast enough then [Doubling-Trick](Policies/DoublingTrickWrapper.py)([kl-UCB++](Policies/klUCBPlusPlus.py)) can perform as well as [kl-UCB++](Policies/klUCBPlusPlus.py) (see for t < 40000).

### [Doubling-Trick](Policies/DoublingTrickWrapper.py) with restart, on randomly taken Bernoulli problems
![Doubling-Trick with restart, on randomly taken Bernoulli problems](plots/main____env1-1_3633169128724378553.png)

Similarly but for µ1,…,µK evenly spaced in [0,1]^K ({0.1,…,0.9}).
Both [kl-UCB](Policies/klUCB.py) and [kl-UCB++](Policies/klUCBPlusPlus.py) are very efficient on "easy" problems like this one, and we can check visually that they match the lower bound from Lai & Robbins (1985).
As before we check that slow doubling are too slow to give reasonable performance.


### [Doubling-Trick](Policies/DoublingTrickWrapper.py) with restart, on randomly taken Gaussian problems with variance V=1
![Doubling-Trick with restart, on randomly taken Gaussian problems with variance V=1](plots/main____env1-1_2223860464453456415.png)

Regret for K=9 [Gaussian arms](Arms/Gaussian.py) N(mu, 1), horizon T=45678, n=1000 repetitions and µ1,…,µK taken uniformly in [-5,5]^K and variance V=1.
On "hard" problems like this one, both [UCB](Policies/UCB.py) and [AFHG](Policies/ApproximatedFHGittins.py) perform similarly and poorly w.r.t. to the lower bound from Lai & Robbins (1985).
As before we check that geometric doubling (b=2) and slow exponential doubling (b=1.1) are too slow, but a fast enough doubling sequence does give reasonable performance for the anytime [AFHG](Policies/ApproximatedFHGittins.py) obtained by [Doubling-Trick](Policies/DoublingTrickWrapper.py).

### [Doubling-Trick](Policies/DoublingTrickWrapper.py) with restart, on an easy Gaussian problems with variance V=1
![Doubling-Trick with restart, on an easy Gaussian problems with variance V=1](plots/main____env1-1_6979515539977716717.png)

Regret for [Doubling-Trick](Policies/DoublingTrickWrapper.py), for K=9 [Gaussian arms](Arms/Gaussian.py) N(mu, 1), horizon T=45678, n=1000 repetitions and µ1,…,µK uniformly spaced in [-5,5]^K.
On "easy" problems like this one, both [UCB](Policies/UCB.py) and [AFHG](Policies/ApproximatedFHGittins.py) perform similarly and attain near constant regret (identifying the best [Gaussian arm](Arms/Gaussian.py) is very easy here as they are sufficiently distinct).
Each [doubling trick](Policies/DoublingTrickWrapper.py) also appear to attain near constant regret, but geometric doubling (b=2) and slow exponential doubling (b=1.1) are slower to converge and thus less efficient.


### [Doubling-Trick](Policies/DoublingTrickWrapper.py) with no restart, on randomly taken Bernoulli problems
![Doubling-Trick with no restart, on randomly taken Bernoulli problems](plots/main____env1-1_5964629015089571121.png)

Regret for K=9 [Bernoulli arms](Arms/Bernoulli.py), horizon T=45678, n=1000 repetitions and µ1,…,µK taken uniformly in [0,1]^K, for [Doubling-Trick](Policies/DoublingTrickWrapper.py) no-restart.
Geometric doubling (\eg, b=2) and slow exponential doubling (\eg, b=1.1) are too slow, and short first sequences make the regret blow up in the beginning of the experiment.
At t=40000 we see clearly the effect of a new sequence for the best [doubling trick](Policies/DoublingTrickWrapper.py) (T_i = 200 x 2^i).
As expected, [kl-UCB++](Policies/klUCBPlusPlus.py) outperforms [kl-UCB](Policies/klUCB.py), and if the doubling sequence is growing fast enough then [Doubling-Trick](Policies/DoublingTrickWrapper.py) no-restart for [kl-UCB++](Policies/klUCBPlusPlus.py) can perform as well as [kl-UCB++](Policies/klUCBPlusPlus.py).

### [Doubling-Trick](Policies/DoublingTrickWrapper.py) with no restart, on an "simple" Bernoulli problems
![Doubling-Trick with no restart, on an "simple" Bernoulli problems](plots/main____env1-1_5972568793654673752.png)

K=9 [Bernoulli arms](Arms/Bernoulli.py) with µ1,…,µK evenly spaced in [0,1]^K.
On easy problems like this one, both [kl-UCB](Policies/klUCB.py) and [kl-UCB++](Policies/klUCBPlusPlus.py) are very efficient, and here the geometric allows the [Doubling-Trick](Policies/DoublingTrickWrapper.py) no-restart anytime version of [kl-UCB++](Policies/klUCBPlusPlus.py) to outperform both [kl-UCB](Policies/klUCB.py) and [kl-UCB++](Policies/klUCBPlusPlus.py).


> These illustrations come from my article, [[What the Doubling Trick Can or Can't Do for Multi-Armed Bandits, Lilian Besson and Emilie Kaufmann, 2018]](https://hal.inria.fr/hal-XXX), to be presented soon.


----

## :scroll: License ? [![GitHub license](https://img.shields.io/github/license/Naereen/AlgoBandits.svg)](https://github.com/Naereen/AlgoBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/AlgoBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/Naereen/AlgoBandits/README.md?pixel)](https://GitHub.com/Naereen/AlgoBandits/)
![PyPI implementation](https://img.shields.io/pypi/implementation/ansicolortags.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)
[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)
