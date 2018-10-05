# **Non-Stationary Stochastic Multi-Armed Bandits**

A well-known and well-studied variant of the [stochastic Multi-Armed Bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit) is the so-called **Non-Stationary Stochastic Multi-Armed Bandits**.
I give here a short introduction, with references below. If you are in a hurry, please read the first two pages of [this article instead](https://arxiv.org/pdf/1802.08380).

- The first studied variant considers *piece-wise* stationary problems, also referred to as **abruptly changing**, where the distributions of the `$K$` arms are stationary on some intervals `$[T_i,\ldots,T_{i+1}]$` with some abrupt change points `$(T_i)$`.
    + It is always assumed that the location of the change points are unknown to the user, otherwise the problem is not harder: just play your [favorite algorithm](docs/Policies.html), and restart it at each change point.
    + The change points can be fixed or randomly generated, but it is assumed that they are generated with a random source being oblivious of the user's actions, so we can always consider that they were already generated before the game starts.
    + For instance, [`Arms.geometricChangePoints()`](docs/Arms.html#Arms.geometricChangePoints) generate some change point if we assume that at every time step `$t=1,\ldots,T]$`, there is a (small) probability p to have a change point.
    + The number of change points is usually denoted `$L$` or `$\Upsilon_T$`, and should not be a constant w.r.t. `$T$` (otherwise when `$T\to\infty$` only the last section counts and give a stationary problem so it is not harder). Some algorithms require to know the value of `$\Upsilon_T$`, or at least an upper-bound, and some algorithms try to be efficient without knowing it (this is what we want!).
    + The goal is to have an efficient algorithm, but of course if `$\Upsilon_T = \mathcal{O}(T)$` the problem is too hard to hope to be efficient and any algorithm will suffer a linear regret (i.e., be as efficient as a naive random strategy).

- Another variant is the **slowly varying** problem, where the rewards `$r(t) = r_{A(t),t}$` is sampled at each time from a parametric distribution, and the parameter(s) change at each time (usually parametrized by its mean). If we focus on 1D exponential families, or any family of distributions parametrized by their mean `$\mu$`, we denote this by having `$r(t) \sim D(\mu_{A(t)}(t))$` where `$\mu_k(t)$` can be varying with the time. The slowly varying hypothesis is that every time step can be a break point, and that the speed of change `$|\mu_k(t+1) - \mu_k(t)|$` is bounded.

- Other variants include harder settings.
    + For instance, we can consider that an adversarial is deciding the change points, by being adaptative to the user's actions. I consider it harder, as always with adversarial problems, and not very useful to model real-world problems.
    + Another harder setting is a "pseudo-Markovian rested" point-of-view: the mean (or parameters) of an arm's distribution can change only when it is sampled, either from time to time or at each time step. It makes sense for some applications (see [Julien's work](https://www.linkedin.com/in/julien-seznec-29364a104/)), but for others it doesn't really make sense (e.g., cognitive radio applications).

TODO fix notations more!

## Applications

TL;DR: the world is non stationary, so it makes sense to study this!

TODO write more justifications about applications, mainly for IoT networks (like when I studied [multi-player bandits](MultiPlayers)).

## References

Here is a partial list of references on this topic. For more, a good starting point is to read the references given in the mentioned article, as always.

### Main references
1. [["The Non-Stochastic Multi-Armed Bandit Problem". P. Auer, N. Cesa-Bianchi, Y. Freund and R. Schapire. SIAM journal on computing, 32(1), 48-77, 2002](https://epubs.siam.org/doi/pdf/10.1137/S0097539701398375)] is apparently the first reference (historically).

2. The Sliding-Window and Discounted UCB algorithms were given in [["On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems". Aurélien Garivier and Éric Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)].
    + They are implemented in [`Policies.SlidingWindowUCB.SWUCB`](docs/Policies.SlidingWindowUCB.html#Policies.SlidingWindowUCB.SWUCB) and [`Policies.DiscountedUCB`](docs/Policies.DiscountedUCB.html).
    + Note that I also implemented the non-anytime heuristic given by the author, [`Policies.SlidingWindowUCB.SWUCBPlus`](docs/Policies.SlidingWindowUCB.html#Policies.SlidingWindowUCB.SWUCBPlus) which uses the knowledge of the horizon `$T$` to *try to* guess a correct value for `$\tau$` the sliding window size.
    + I implemented this sliding window idea in a generic way, and [`Policies.SlidingWindowRestart`](docs/Policies.SlidingWindowRestart.html) is a generic wrapper that can work with (almost) any algorithm: it is an experimental policy, using a sliding window (of for instance `$\tau=100$` draws of each arm), and reset the underlying algorithm as soon as the small empirical average is too far away from the long history empirical average (or just restart for one arm, if possible).

3. [["Thompson sampling for dynamic multi-armed bandits". N Gupta,. OC Granmo, A. Agrawala, 10th International Conference on Machine Learning and Applications Workshops. IEEE, 2011](https://www.researchgate.net/profile/Ole-Christoffer_Granmo/publication/232616670_Thompson_Sampling_for_Dynamic_Multi-armed_Bandits/links/56a7d8e808ae0fd8b3fe3dc6.pdf)]
    + TODO read it!

4. [["Stochastic multi-armed-bandit problem with non-stationary rewards", O. Besbes, Y. Gur, A. Zeevi. Advances in Neural Information Processing Systems (pp. 199-207), 2014](http://papers.nips.cc/paper/5378-stochastic-multi-armed-bandit-problem-with-non-stationary-rewards.pdf)]
    + TODO read it!


### Recent references
More recent articles include the following:

1. [["On Abruptly-Changing and Slowly-Varying Multiarmed Bandit Problems". L. Wei and V. Srivastav. arXiv preprint arXiv:1802.08380, 2018](https://arxiv.org/pdf/1802.08380)], introduced the first algorithms that can (try to) tackle the two problems simultaneously, [`LM-DSEE`](XXX) and [`SW-UCB#`](XXX).
    + TODO code it!

2. [["Adaptively Tracking the Best Arm with an Unknown Number of Distribution Changes". Peter Auer, Pratik Gajane and Ronald Ortner. EWRL 2018, Lille](https://ewrl.files.wordpress.com/2018/09/ewrl_14_2018_paper_28.pdf)], introduced the [`AdSwitch`](XXX) algorithm, which does not require to know the number `$\Upsilon_T$` of change points.
    + TODO code it!
    + TODO adapt it to unknown horizon (using [doubling tricks?](DoublingTrick.html)!

3. [["Memory Bandits: a Bayesian approach for the Switching Bandit Problem". Réda Alami, Odalric Maillard, Raphaël Féraud. 31st Conference on Neural Information Processing Systems (NIPS 2017), hal-01811697](https://hal.archives-ouvertes.fr/hal-01811697/document)], introduced the [`MemoryBandit`](XXX) algorithm, which does not require to know the number `$\Upsilon_T$` of change points.
    + TODO code it!

- FIXME give more!


References that seem interesting but I haven't read them yet:

1. [["The non-stationary stochastic multi-armed bandit problem". R. Allesiardo, Raphaël Féraud and Odalric-Ambrym Maillard. International Journal of Data Science and Analytics, 3(4), 267-283. 2017](https://hal.archives-ouvertes.fr/hal-01575000/document)]

2. [["A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem". F. Liu, J. Lee and N. Shroff. arXiv preprint arXiv:1711.03539, 2017](https://arxiv.org/pdf/1711.03539)]

3. [["Taming non-stationary bandits: A Bayesian approach". V. Raj and S. Kalyani. arXiv preprint arXiv:1707.09727, 2017](https://arxiv.org/pdf/1707.09727)]

----

## Example of simulation configuration

FIXME finish implementation!

> See [issue #71](https://github.com/SMPyBandits/SMPyBandits/issues/71) and [issue #146](https://github.com/SMPyBandits/SMPyBandits/issues/146).

A simple python file, [`configuration_sparse.py`](SMPyBandits/configuration_sparse.py), is used to import the [arm classes](Arms/), the [policy classes](Policies/) and define the problems and the experiments.

For example, we can compare the standard [`UCB`](SMPyBandits/Policies/UCB.py) and  [`BayesUCB`](SMPyBandits/Policies/BayesUCB.py) algorithms, non aware of the sparsity, against the sparsity-aware [`SparseUCB`](SMPyBandits/Policies/SparseUCB.py) algorithm, as well as 4 versions of [`SparseWrapper`](SMPyBandits/Policies/SparseWrapper.py) applied to [`BayesUCB`](SMPyBandits/Policies/BayesUCB.py).

```python
horizon = 10000
nb_random_events = 10
configuration = {
    "horizon": horizon,    # Finite horizon of the simulation
    "repetitions": 100,  # number of repetitions
    "n_jobs": -1,        # Maximum number of cores for parallelization: use ALL your CPU
    "verbosity": 5,      # Verbosity for the joblib calls
    # Environment configuration, you can set up more than one.
    "environment": [     # Bernoulli arms with non-stationarity
        {   # A non stationary problem: every step of the same repetition use a different mean vector!
            "arm_type": Bernoulli,
            "params": {
                "newMeans": randomContinuouslyVaryingMeans,
                "changePoints": geometricChangePoints(horizon=horizon, proba=nb_random_events/horizon),
                "args": {
                    "nbArms": 9,
                    "lower": 0, "amplitude": 1,
                    "mingap": None, "isSorted": True,
                }
            }
        },
    ]
    ],
    # Policies that should be simulated, and their parameters.
    "policies": [
        {"archtype": UCB, "params": {} },
        {"archtype": SWUCB, "params": { "tau": 100 } },
        {"archtype": SWUCB, "params": { "tau": 500 } },
        {"archtype": SWUCB, "params": { "tau": 1000 } },
        {"archtype": SWUCB, "params": { "tau":  # formula from [GarivierMoulines2011]
            2 * np.sqrt(horizon * np.log(horizon) / (1 + nb_random_events))
        } },
        {"archtype": DiscountedUCB, "params": { "alpha": 1, "gamma": 0.99 } },
        {"archtype": DiscountedUCB, "params": { "alpha": 1, "gamma": 0.85 } },
        {"archtype": DiscountedUCB, "params": { "alpha": 1, "gamma": 0.5 } },
    ]
}
```

----

## [How to run the experiments ?](How_to_run_the_code.md)

You should use the provided [`Makefile`](Makefile) file to do this simply:
```bash
make install         # install the requirements ONLY ONCE
make nonstationary   # run and log the main.py script FIXME
```

----

## Some illustrations

Here are some plots illustrating the performances of the different [policies](SMPyBandits/Policies/) implemented in this project, against various sparse problems (with [`Bernoulli`](Arms/Bernoulli.py) or [`UnboundedGaussian`](SMPyBandits/Arms/Gaussian.py) arms only):

FIXME do some plots!

----

## Article?

> Not yet! We are working on this! TODO

----

### :scroll: License ? [![GitHub license](https://img.shields.io/github/license/SMPyBandits/SMPyBandits.svg)](https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel)](https://GitHub.com/SMPyBandits/SMPyBandits/)
![PyPI version](https://img.shields.io/pypi/v/smpybandits.svg)
![PyPI implementation](https://img.shields.io/pypi/implementation/SMPyBandits.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/SMPyBandits.svg)
[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)
