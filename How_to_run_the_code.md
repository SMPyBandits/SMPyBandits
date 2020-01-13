# How to run the code ?

> This short page explains quickly how to install the requirements for this project, and then how to use the code to run simulations.

## Required modules

### Virtualenv
*First*, install the requirements, globally (or with a virtualenv, [see below](#In-a-virtualenv-?)):
```bash
pip install -r requirements.txt
```

> Some requirements are only needed for one policy (mostly the experimental ones), and for the documentation.

### Nix

A pinned [Nix](https://nixos.org) environment is available:
```bash 
nix-shell
``` 

## Running some simulations
*Then*, it should be very straight forward to run some experiment.
This will run the simulation, average them (by `repetitions`) and plot the results.

### Single player
#### [Single player](https://smpybandits.github.io/docs/main.html)
```bash
python main.py
# or
make main
```

#### [Single player, aggregating algorithms](https://smpybandits.github.io/docs/configuration_comparing_aggregation_algorithms.html)
```bash
python main.py configuration_comparing_aggregation_algorithms
# or
make comparing_aggregation_algorithms
```
> See these explainations: [Aggregation.md](Aggregation.html)

#### [Single player, doubling-trick algorithms](https://smpybandits.github.io/docs/configuration_comparing_doubling_algorithms.html)
```bash
python main.py configuration_comparing_doubling_algorithms
# or
make comparing_doubling_algorithms
```
> See these explainations: [DoublingTrick.md](DoublingTrick.html)

#### [Single player, with Sparse Stochastic Bandit](https://smpybandits.github.io/docs/configuration_sparse.html)
```bash
python main.py configuration_sparse
# or
make sparse
```
> See these explainations: [SparseBandits.md](SparseBandits.html)

#### [Single player, with Markovian problem](https://smpybandits.github.io/docs/configuration_markovian.html)
```bash
python main.py configuration_markovian
# or
make markovian
```

#### [Single player, with non-stationary problem](https://smpybandits.github.io/docs/configuration_nonstationary.html)
```bash
python main.py configuration_nonstationary
# or
make nonstationary
```
> See these explainations: [NonStationaryBandits.md](NonStationaryBandits.html)

### Multi-Player
#### [Multi-Player, one algorithm](https://smpybandits.github.io/docs/main_multiplayers.html)
```bash
python main_multiplayers.py
# or
make multi
```

#### [Multi-Player, comparing different algorithms](https://smpybandits.github.io/docs/main_multiplayers_more.html)
```bash
python main_multiplayers_more.py
# or
make moremulti
```
> See these explainations: [MultiPlayers.md](MultiPlayers.html)

----

### Using `env` variables ?

For all simulations, I recently added the support for *environment variable*, to ease the customization of the main parameters of every simulations.

For instance, if the [`configuration_multiplayers_more.py`](https://smpybandits.github.io/docs/configuration_multiplayers_more.html) file is correct,
then you can customize to use `N=4` repetitions, for horizon `T=1000` and `M=3` players, parallelized with `N_JOBS=4` jobs (use the number of cores of your CPU for optimal performance):

```bash
N=4 T=1000 M=3 DEBUG=True SAVEALL=False N_JOBS=4 make moremulti
```

----

## In a [`virtualenv`](https://virtualenv.pypa.io/en/stable/) ?
If you prefer to not install the requirements globally on your system-wide Python setup, you can (and should) use [`virtualenv`](https://virtualenv.pypa.io/en/stable/).

```bash
$ virtualenv .
Using base prefix '/usr'
New python executable in /your/path/to/SMPyBandits/bin/python3
Also creating executable in /your/path/to/SMPyBandits/bin/python
Installing setuptools, pip, wheel...done.
$ source bin/activate  # in bash, use activate.csh or activate.fish if needed
$ type pip  # just to check
pip is /your/path/to/SMPyBandits/bin/pip
$ pip install -r requirements.txt
Collecting numpy (from -r requirements.txt (line 5))
...
Installing collected packages: numpy, scipy, cycler, pytz, python-dateutil, matplotlib, joblib, pandas, seaborn, tqdm, sphinx-rtd-theme, commonmark, docutils, recommonmark
Successfully installed commonmark-0.5.4 cycler-0.10.0 docutils-0.13.1 joblib-0.11 matplotlib-2.0.0 numpy-1.12.1 pandas-0.19.2 python-dateutil-2.6.0 pytz-2016.10 recommonmark-0.4.0 scipy-0.19.0 seaborn-0.7.1 sphinx-rtd-theme-0.2.4 tqdm-4.11.2
```

And then be sure to use the virtualenv binary for Python, `bin/python`, instead of the system-wide one, to launch the experiments (the Makefile should use it by default, if `source bin/activate` was executed).

---

## Or with a [`Makefile`](Makefile) ?
You can also use the provided [`Makefile`](Makefile) file to do this simply:
```bash
make install       # install the requirements
make multiplayers  # run and log the main.py script
```

It can be used to check [the quality of the code](logs/main_pylint_log.txt) with [pylint](https://www.pylint.org/):
```bash
make lint lint3  # check the code with pylint
```

It is also used to clean the code, build the doc, send the doc, etc. (This should not be used by others)

---

## Or within a [Jupyter notebook](https://jupyter.org/) ?
> I am writing some [Jupyter notebooks](https://jupyter.org/), in [this folder (`notebooks/`)](notebooks/), so if you want to do the same for your small experiments, you can be inspired by the few notebooks already written.

----

### :scroll: License ? [![GitHub license](https://img.shields.io/github/license/SMPyBandits/SMPyBandits.svg)](https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

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
