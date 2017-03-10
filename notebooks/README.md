# [Jupyter Notebooks](https://www.jupyter.org/) :notebook: by [Naereen @ GitHub](https://naereen.github.io/)

This folder hosts some [Jupyter Notebooks](http://jupyter.org/), to present in a nice format some numerical experiments for [my AlgoBandits project](https://naereen.github.io/AlgoBandits/).

> [The wonderful Jupyter tools](http://jupyter.org/)  is awesome to write interactive and nicely presented :snake: Python simulations!
>
> [![made-with-jupyter](https://img.shields.io/badge/Made%20with-Jupyter-1f425f.svg)](http://jupyter.org/) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

----

## 1. List of experiments presented with notebooks
- [Easily creating various Multi-Armed Bandit problems](Easily_creating_MAB_problems.ipynb), explains the interface of the [`Environment.MAB`](../Environment/MAB.py) module.
- [A simple example of Single-Player simulation](Example_of_a_small_Single-Player_Simulation.ipynb), comparing UCB1 (for two values of $\alpha$, 1 and 1/2), Thompson Sampling, BayesUCB and kl-UCB.

----

## 2. Question: *How to read these documents*?

### 2.a. View the _notebooks_ statically :memo:
- Either directly in GitHub: [see the list of notebooks](https://github.com/Naereen/AlgoBandits/search?l=jupyter-notebook);
- Or on [nbviewer.jupiter.org](http://nbviewer.jupiter.org/): [list of notebooks](http://nbviewer.jupyter.org/github/Naereen/AlgoBandits/).

### 2.b. FIXME not yet - Play with the _notebooks_ dynamically :boom:
[![MyBinder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/Naereen/AlgoBandits)

Anyone can use the [mybinder.org](http://mybinder.org/) website (by [clicking](http://mybinder.org/repo/Naereen/AlgoBandits) on the icon above) to run the notebook in her/his web-browser.
You can then play with it as long as you like, for instance by modifying the values or experimenting with the code.

----

## 3. Question: *Requirements to run the notebooks locally*?
All [the requirements](requirements.txt) can be installed with [``pip``](https://pip.readthedocs.io/).

> Note: if you use [Python 3](https://docs.python.org/3/) instead of [Python 2](https://docs.python.org/2/), you *might* have to *replace* ``pip`` and ``python`` by ``pip3`` and ``python3`` in the next commands (if both `pip` and `pip3` are installed).

### 3.a. [Jupyter Notebook](http://jupyter.readthedocs.org/en/latest/install.html) and [IPython](http://ipython.org/)

```bash
sudo pip install jupyter ipython
```

It will also install all the dependencies, afterward you should have a ``jupyter-notebook`` command (or a ``jupyter`` command, to be ran as ``jupyter notebook``) available in your ``PATH``:

```bash
$ whereis jupyter-notebook
jupyter-notebook: /usr/local/bin/jupyter-notebook
$ jupyter-notebook --version  # version >= 4 is recommended
4.4.1
```

### 3.b. My numerical environment, [`AlgoBandits`](https://naereen.github.io/AlgoBandits/)

- First, install its dependencies (`pip install -r requirements`).
- Then, either install it (*not yet*), or be sure to work in the main folder.

> *Note:* it's probably better to use *virtualenv*, if you like it.

----

### :information_desk_person: More information?
> - More information about [notebooks (on the documentation of IPython)](http://nbviewer.jupiter.org/github/ipython/ipython/blob/3.x/examples/Notebook/Index.ipynb) or [on the FAQ on Jupyter's website](http://nbviewer.jupyter.org/faq).
> - More information about [mybinder.org](http://mybinder.org/): on [this example repository](https://github.com/binder-project/example-requirements).


## :scroll: License ? [![GitHub license](https://img.shields.io/github/license/Naereen/notebooks.svg)](https://github.com/Naereen/AlgoBandits/blob/master/LICENSE)
All the notebooks in this folder are published under the terms of the [MIT License](https://lbesson.mit-license.org/) (file [LICENSE.txt](../LICENSE.txt)).
© [Lilian Besson](https://GitHub.com/Naereen), 2016-17.

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/AlgoBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/Naereen/AlgoBandits/README.md?pixel)](https://GitHub.com/Naereen/AlgoBandits/)
[![made-with-jupyter](https://img.shields.io/badge/Made%20with-Jupyter-1f425f.svg)](http://jupyter.org/) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)

[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://GitHub.com/Naereen/)
