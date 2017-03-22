# How to run the code ?
## Required modules
*First*, install the requirements, globally (or with a virtualenv, [see below](#In-a-virtualenv-?)):
```bash
pip install -r requirements.txt
```

## Running some simulations
*Then*, it should be very straight forward to run some experiment.
This will run the simulation, average them (by `repetitions`) and plot the results.

### [Single player](main.py)
```bash
python main.py
# or
make main
```

### [Single player, aggregating algorithms](main.py)
```bash
python main.py configuration_comparing_KLUCB_aggregation
# or
make comparing_KLUCB_aggregation
```

### [Multi-Player, one algorithm](main_multiplayers.py)
```bash
python main_multiplayers.py
# or
make multi
```

### [Multi-Player, comparing different algorithms](main_multiplayers_more.py)
```bash
python main_multiplayers_more.py
# or
make moremulti
```

----

## In a [`virtualenv`](https://virtualenv.pypa.io/en/stable/) ?
If you prefer to not install the requirements globally on your system-wide Python setup, you can (and should) use [`virtualenv`](https://virtualenv.pypa.io/en/stable/).

```bash
$ virtualenv .
Using base prefix '/usr'
New python executable in /your/path/to/AlgoBandits/bin/python3
Also creating executable in /your/path/to/AlgoBandits/bin/python
Installing setuptools, pip, wheel...done.
$ source bin/activate  # in bash, use activate.csh or activate.fish if needed
$ type pip  # just to check
pip is /tmp/test/AlgoBandits/bin/pip
$ pip install -r requirements.txt
Collecting numpy (from -r requirements.txt (line 5))
...
Installing collected packages: numpy, scipy, cycler, pytz, python-dateutil, matplotlib, joblib, pandas, seaborn, tqdm, sphinx-rtd-theme, commonmark, docutils, recommonmark
Successfully installed commonmark-0.5.4 cycler-0.10.0 docutils-0.13.1 joblib-0.11 matplotlib-2.0.0 numpy-1.12.1 pandas-0.19.2 python-dateutil-2.6.0 pytz-2016.10 recommonmark-0.4.0 scipy-0.19.0 seaborn-0.7.1 sphinx-rtd-theme-0.2.4 tqdm-4.11.2
```

And then be sure to the virtualenv binary, `bin/python`, to launch the experiments (the Makefile should use it by default, if `source bin/activate` was executed).

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

It is also used to clean the code, build the doc, send the doc, etc.

---

## Or within a [Jupyter notebook](https://jupyter.org/) ?
> I am writing some [Jupyter notebooks](https://jupyter.org/), in [this folder (`notebooks/`)](notebooks/), so if you want to do the same for your small experiments, you can be inspired by the few notebooks already written.

----

## :scroll: License ? [![GitHub license](https://img.shields.io/github/license/Naereen/AlgoBandits.svg)](https://github.com/Naereen/AlgoBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2012 [Olivier Cappé](http://perso.telecom-paristech.fr/%7Ecappe/), [Aurélien Garivier](https://www.math.univ-toulouse.fr/%7Eagarivie/), [Émilie Kaufmann](http://chercheurs.lille.inria.fr/ekaufman/) and for the initial [pymaBandits v1.0](http://mloss.org/software/view/415/) project, and © 2016-2017 [Lilian Besson](https://GitHub.com/Naereen) for the rest.

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/AlgoBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/Naereen/AlgoBandits/README.md?pixel)](https://GitHub.com/Naereen/AlgoBandits/)
![PyPI implementation](https://img.shields.io/pypi/implementation/ansicolortags.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)
[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)
