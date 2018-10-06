# `logs` files

This folder keeps some examples of log files to show the output of the simulation scripts.

## Single player simulations

### Example of output of the [`main.py`](../docs/main.html) program
- [See `main_py3_log.txt`](main_py3_log.txt).

---

## Multi players simulations

### Example of output of the [`main_multiplayers.py`](../docs/main_multiplayers.html) program
- [See `main_multiplayers_py3_log.txt`](main_multiplayers_py3_log.txt).

## Example of output of the [`main_multiplayers_more.py`](../docs/main_multiplayers_more.html) program
- [See `main_multiplayers_more_py3_log.txt`](main_multiplayers_more_py3_log.txt).

---

## Linters
### Pylint
- [See `main_pylint_log.txt`](main_pylint_log.txt) for Python 2 (generic) linting report.
- [See `main_pylint3_log.txt`](main_pylint3_log.txt) for Python 3 (specific) linting report.

## Profilers
- [See `main_py3_kernprof_log.txt`](main_py3_kernprof_log.txt) from [`kernprof`](https://github.com/rkern/line_profiler#kernprof) profiling.

- [See `main_py3_profile_log.txt`](main_py3_profile_log.txt) for an example of a line-by-line time profiler.

- [See `main_py3_memory_profiler_log.txt`](main_py3_memory_profiler_log.txt) for an example of a line-by-line time profiler.

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

---

## Other examples
### Example of output of a script
For the [`complete_tree_exploration_for_MP_bandits`](../docs/complete_tree_exploration_for_MP_bandits.html) script, see the file [`complete_tree_exploration_for_MP_bandits_py3_log.txt`](complete_tree_exploration_for_MP_bandits_py3_log.txt).

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
