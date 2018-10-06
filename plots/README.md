# Some illustrations for [this project](https://github.com/SMPyBandits/SMPyBandits)

Here are some plots illustrating the performances of the different [policies](../SMPyBandits/Policies/) implemented in this project, against various problems (with [`Bernoulli`](../SMPyBandits/Arms/Bernoulli.py) arms only):

## Histogram of regrets at the end of some simulations
On a simple Bernoulli problem, we can compare 16 different algorithms (on a short horizon and a small number of repetitions, just as an example).
If we plot the distribution of the regret at the end of each experiment, `R_T`, we can see this kind of plot:

![Histogramme_regret_monoplayer_2.png](Histogramme_regret_monoplayer_2.png)

It helps a lot to see both the mean value (in solid black) of the regret, and its distribution of a few runs (100 here).
It can be used to detect algorithms that perform well in average, but sometimes with really bad runs.
Here, the [Exp3++](../SMPyBandits/Policies/Exp3PlusPlus.py) seems to had one bad run.

---

## Demonstration of different [Aggregation policies](../Aggregation.md)
On a fixed Gaussian problem, aggregating some algorithms tuned for this exponential family (ie, they know the variance but not the means).
Our algorithm, [Aggregator](../SMPyBandits/Policies/Aggregator.py), outperforms its ancestor [Exp4](../SMPyBandits/Policies/Aggregator.py) as well as the other state-of-the-art experts aggregation algorithms, [CORRAL](../SMPyBandits/Policies/CORRAL.py) and [LearnExp](../SMPyBandits/Policies/LearnExp.py).

![main____env3-4_932221613383548446.png](main____env3-4_932221613383548446.png)

---

## Demonstration of [multi-player algorithms](../MultiPlayers.md)
Regret plot on a random Bernoulli problem, with `M=6` players accessing independently and in a decentralized way `K=9` arms.
Our algorithms ([RandTopM](../SMPyBandits/PoliciesMultiPlayers/RandTopM.py) and [MCTopM](../SMPyBandits/PoliciesMultiPlayers/RandTopM.py), as well as [Selfish](../SMPyBandits/Policie/Selfish.py)) outperform the state-of-the-art [rhoRand](../SMPyBandits/PoliciesMultiPlayers/rhoRand.py):

![MP__K9_M6_T5000_N500__4_algos__all_RegretCentralized____env1-1_8318947830261751207.png](MP__K9_M6_T5000_N500__4_algos__all_RegretCentralized____env1-1_8318947830261751207.png)


Histogram on the same random Bernoulli problems.
We see that some all algorithms have a non-negligible variance on their regrets.

![MP__K9_M6_T10000_N1000__4_algos__all_HistogramsRegret____env1-1_8200873569864822246.png](MP__K9_M6_T10000_N1000__4_algos__all_HistogramsRegret____env1-1_8200873569864822246.png)


Comparison with two other "state-of-the-art" algorithms ([MusicalChair](../SMPyBandits/Policies/MusicalChair.py) and [MEGA](../SMPyBandits/Policies/MEGA.py), in semilogy scale to really see the different scale of regret between efficient and sub-optimal algorithms):

![MP__K9_M3_T123456_N100__8_algos__all_RegretCentralized_semilogy____env1-1_7803645526012310577.png](MP__K9_M3_T123456_N100__8_algos__all_RegretCentralized_semilogy____env1-1_7803645526012310577.png)

---

## Other illustrations
### Piece-wise stationary problems
Comparing [Sliding-Window UCB](../SMPyBandits/Policies/SlidingWindowUCB.py) and [Discounted UCB](../SMPyBandits/Policies/DiscountedUCB.py) and [UCB](../SMPyBandits/Policies/UCB.py), on a simple Bernoulli problem which regular random shuffling of the arm.

![Demo_of_DiscountedUCB2.png](Demo_of_DiscountedUCB2.png)

### Sparse problem and Sparsity-aware algorithms
Comparing regular [UCB](../SMPyBandits/Policies/UCB.py), [klUCB](../SMPyBandits/Policies/klUCB.py) and [Thompson sampling](../SMPyBandits/Policies/Thompson.py) against ["sparse-aware" versions](../SMPyBandits/Policies/SparseWrapper.py), on a simple Gaussian problem with `K=10` arms but only `s=4` with non-zero mean.

![Demo_of_SparseWrapper_regret.png](Demo_of_SparseWrapper_regret.png)

---

## Demonstration of the [Doubling Trick policy](../DoublingTrick.md)
- On a fixed problem with full restart:
  ![main____env1-1_3633169128724378553.png](main____env1-1_3633169128724378553.png)

- On a fixed problem with no restart:
  ![main____env1-1_5972568793654673752.png](main____env1-1_5972568793654673752.png)

- On random problems with full restart:
  ![main____env1-1_1217677871459230631.png](main____env1-1_1217677871459230631.png)

- On random problems with no restart:
  ![main____env1-1_5964629015089571121.png](main____env1-1_5964629015089571121.png)

---

## Plots for the [JMLR MLOSS](http://jmlr.org/mloss/) paper

In [the JMLR MLOSS paper](../paper/paper.md) I wrote to present SMPyBandits,
an example of a simulation is presented, where we compare the standard anytime [`klUCB`](https://SMPyBandits.GitHub.io/docs/Policies.klUCB.html) algorithm against the non-anytime variant [`klUCBPlusPlus`](https://SMPyBandits.GitHub.io/docs/Policies.klUCBPlusPlus.html) algorithm, and also [`UCB`](https://SMPyBandits.GitHub.io/docs/Policies.UCBalpha.html) (with \(\alpha=1\)) and [`Thompson`](https://SMPyBandits.GitHub.io/docs/Policies.Thompson.html) (with [Beta posterior](https://SMPyBandits.GitHub.io/docs/Policies.Posterior.Beta.html)).

```python
configuration["policies"] = [
  { "archtype": klUCB, "params": { "klucb": klucbBern } },
  { "archtype": klUCBPlusPlus, "params": { "horizon": HORIZON, "klucb": klucbBern } },
  { "archtype": UCBalpha, "params": { "alpha": 1 } },
  { "archtype": Thompson, "params": { "posterior": Beta } }
]
```

Running this simulation as shown below will save figures in a sub-folder, as well as save data (pulls, rewards and regret) in [HDF5 files](http://docs.h5py.org/en/stable/high/file.html).

```bash
# 3. run a single-player simulation
$ BAYES=False ARM_TYPE=Bernoulli N=1000 T=10000 K=9 N_JOBS=4 \
  MEANS=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] python3 main.py configuration.py
```

The two plots below shows the average regret for these 4 algorithms.
The regret is the difference between the cumulated rewards of the best fixed-armed strategy (which is the oracle strategy for stationary bandits), and the cumulated rewards of the considered algorithms.

- Average regret:
  ![paper/3.png](paper/3.png)

- Histogram of regrets:
  ![paper/3_hist.png](paper/3_hist.png)

> Example of a single-player simulation showing the average regret and histogram of regrets of 4 algorithms. They all perform very well: each algorithm is known to be order-optimal (*i.e.*, its regret is proved to match the lower-bound up-to a constant), and each but UCB is known to be optimal (*i.e.* with the constant matching the lower-bound). For instance, Thomson sampling is very efficient in average (in yellow), and UCB shows a larger variance (in red).

### Saving simulation data to HDF5 file

This simulation produces this example HDF5 file,
which contains attributes (*e.g.*, `horizon=10000`, `repetitions=1000`, `nbPolicies=4`),
and a collection of different datasets for each environment.
Only one environment was tested, and for `env_0` the HDF5 stores some attributes (*e.g.*, `nbArms=9` and `means=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]`)
and datasets (*e.g.*, `bestArmPulls` of shape `(4, 10000)`, `cumulatedRegret` of shape `(4, 10000)`, `lastRegrets` of shape `(4, 1000)`, `averageRewards` of shape `(4, 10000)`).
See the example:
[GitHub.com/SMPyBandits/SMPyBandits/blob/master/plots/paper/example.hdf5](https://github.com/SMPyBandits/SMPyBandits/blob/master/plots/paper/example.hdf5).

> Note: [HDFCompass](https://github.com/HDFGroup/hdf-compass) is recommended to explore the file from a nice and easy to use GUI. Or use it from a Python script with [h5py](http://docs.h5py.org/en/stable/index.html) or a Julia script with [HDF5.jl](https://github.com/JuliaIO/HDF5.jl).
> ![Example of exploring this 'example.hdf5' file using HDFCompass](paper/example_HDF5_exploration_with_HDFCompass.png)

---

## Graph of time and memory consumptions
### Time consumption
Note that [I had added a very clean support](https://github.com/SMPyBandits/SMPyBandits/issues/94) for time consumption measures, every simulation script will output (as the end) some lines looking like this:

```
Giving the mean and std running times ...
For policy #0 called 'UCB($\alpha=1$)' ...
    84.3 ms ± 7.54 ms per loop (mean ± std. dev. of 10 runs)
For policy #1 called 'Thompson' ...
    89.6 ms ± 17.7 ms per loop (mean ± std. dev. of 10 runs)
For policy #3 called 'kl-UCB$^{++}$($T=1000$)' ...
    2.52 s ± 29.3 ms per loop (mean ± std. dev. of 10 runs)
For policy #2 called 'kl-UCB' ...
    2.59 s ± 284 ms per loop (mean ± std. dev. of 10 runs)
```

![Demo_of_automatic_time_consumption_measure_between_algorithms](../plots/Demo_of_automatic_time_consumption_measure_between_algorithms.png)

### Memory consumption
Note that [I had added an experimental support](https://github.com/SMPyBandits/SMPyBandits/issues/129) for time consumption measures, every simulation script will output (as the end) some lines looking like this:

```
Giving the mean and std memory consumption ...
For players called '3 x RhoRand-kl-UCB, rank:1' ...
    23.6 KiB ± 52 B (mean ± std. dev. of 10 runs)
For players called '3 x RandTopM-kl-UCB' ...
    1.1 KiB ± 0 B (mean ± std. dev. of 10 runs)
For players called '3 x Selfish-kl-UCB' ...
    12 B ± 0 B (mean ± std. dev. of 10 runs)
For players called '3 x MCTopM-kl-UCB' ...
    4.9 KiB ± 86 B (mean ± std. dev. of 10 runs)
For players called '3 x MCNoSensing($M=3$, $T=1000$)' ...
    12 B ± 0 B (mean ± std. dev. of 10 runs)
```

![Demo_of_automatic_memory_consumption_measure_between_algorithms](../plots/Demo_of_automatic_memory_consumption_measure_between_algorithms.png)

> It is still experimental!

----

### :scroll: License ? [![GitHub license](https://img.shields.io/github/license/SMPyBandits/SMPyBandits.svg)](https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/SMPyBandits/SMPyBandits/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel)](https://GitHub.com/SMPyBandits/SMPyBandits/)
![PyPI version](https://img.shields.io/pypi/v/smpybandits.svg)
![PyPI implementation](https://img.shields.io/pypi/implementation/smpybandits.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/smpybandits.svg)
[![Documentation Status](https://readthedocs.org/projects/smpybandits/badge/?version=latest)](https://SMPyBandits.ReadTheDocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/SMPyBandits/SMPyBandits.svg?branch=master)](https://travis-ci.org/SMPyBandits/SMPyBandits)
[![Stars of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/stars/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/stargazers)
[![Releases of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/release/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/releases)
