# [Jupyter Notebooks](https://www.jupyter.org/) :notebook: by [Naereen @ GitHub](https://naereen.github.io/)

This folder hosts some [Jupyter Notebooks](http://jupyter.org/), to present in a nice format some numerical experiments for [my SMPyBandits project](https://github.com/SMPyBandits/SMPyBandits/).

> [The wonderful Jupyter tools](http://jupyter.org/)  is awesome to write interactive and nicely presented :snake: Python simulations!
>
> [![made-with-jupyter](https://img.shields.io/badge/Made%20with-Jupyter-1f425f.svg)](http://jupyter.org/) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

----

## 1. List of experiments presented with notebooks
### MAB problems
- [Easily creating various Multi-Armed Bandit problems](Easily_creating_MAB_problems.ipynb), explains the interface of the [`Environment.MAB`](../Environment/MAB.py) module.

### Single-Player simulations
- [A simple example of Single-Player simulation](Example_of_a_small_Single-Player_Simulation.ipynb), comparing [`UCB1`](https://smpybandits.github.io/docs/Policies.UCBalpha.html) (for two values of $\alpha$, 1 and 1/2), [`Thompson Sampling`](https://smpybandits.github.io/docs/Policies.Thompson.html), [`BayesUCB`](https://smpybandits.github.io/docs/Policies.BayesUCB.html) and [`kl-UCB`](https://smpybandits.github.io/docs/Policies.klUCB.html).
- [*Do we even need UCB?*](Do_we_even_need_UCB.ipynb) demonstrates the need for an algorithm smarter than the naive [`EmpiricalMeans`](https://smpybandits.github.io/docs/Policies.EmpiricalMeans.html).
- [Lai-Robbins lower-bound for doubling-tricks algorithms with full restart](Lai_Robbins_Lower_Bound_for_Doubling_Trick_with_Restarting_Algorithms.ipynb).

### Active research on Single-Player MAB
- [Exploring different doubling tricks for different kinds of regret bounds](Exploring_different_doubling_tricks_for_different_kinds_of_regret_bounds.ipynb).
- [Experiments of statistical tests for piecewise stationary bandits](Experiments_of_statistical_tests_for_piecewise_stationary_bandit.ipynb)
- [Demonstrations of Single-Player Simulations for Non-Stationary Bandits](Demonstrations_of_Single-Player_Simulations_for_Non-Stationary-Bandits.ipynb).

### Multi-Player simulations
- [A simple example of Multi-Player simulation with 4 Centralized Algorithms](Example_of_a_small_Multi-Player_Simulation__with_Centralized_Algorithms.ipynb), comparing [`CentralizedMultiplePlay`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.CentralizedMultiplePlay.html) and [`CentralizedIMP`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.CentralizedIMP.html) with [`UCB`](https://smpybandits.github.io/docs/Policies.UCB.html) and [`Thompson Sampling`](https://smpybandits.github.io/docs/Policies.Thompson.html).
- [A simple example of Multi-Player simulation with 2 Decentralized Algorithms](Example_of_a_small_Multi-Player_Simulation__with_rhoRand_and_Selfish_Algorithms.ipynb), comparing [`rhoRand`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.rhoRand.html) and [`Selfish`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.Selfish.html) (for the "collision avoidance" part) combined with [`UCB`](https://smpybandits.github.io/docs/Policies.UCB.html) and [`Thompson Sampling`](https://smpybandits.github.io/docs/Policies.Thompson.html) for learning the arms. Spoiler: `Selfish` beats `rhoRand`!

## (Old) Experiments
- [Can we use a (non-online) Unsupervised Learning algorithm for (online) Bandit problem ?](Unsupervised_Learning_for_Bandit_problem.ipynb)
- [Can we use a computationally expensive Black-Box Bayesian optimization algorithm for (online) Bandit problem ?](BlackBox_Bayesian_Optimization_for_Bandit_problems.ipynb)

----

## 2. Question: *How to read these documents*?

### 2.a. View the _notebooks_ statically :memo:
- Either directly in GitHub: [see the list of notebooks](https://github.com/SMPyBandits/SMPyBandits/search?l=jupyter-notebook);
- Or on [nbviewer.jupiter.org](https://nbviewer.jupiter.org/): [list of notebooks](https://nbviewer.jupyter.org/github/SMPyBandits/SMPyBandits/).

### 2.b. Play with the _notebooks_ dynamically (on MyBinder) :boom:
[![Binder](https://mybinder.org/badge_logo.svg)](http://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks)

Anyone can use the [mybinder.org](http://mybinder.org/) website (by [clicking](http://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks) on the icon above) to run the notebook in her/his web-browser.
You can then play with it as long as you like, for instance by modifying the values or experimenting with the code.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FDo_we_even_need_UCB.ipynb)

- [Do_we_even_need_UCB.ipynb](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FDo_we_even_need_UCB.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FDo_we_even_need_UCB.ipynb)
- [Easily_creating_MAB_problems.ipynb](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FEasily_creating_MAB_problems.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FEasily_creating_MAB_problems.ipynb)
- [Example_of_a_small_Single-Player_Simulation.ipynb](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FExample_of_a_small_Single-Player_Simulation.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FExample_of_a_small_Single-Player_Simulation.ipynb)
- [Example_of_a_small_Multi-Player_Simulation__with_Centralized_Algorithms.ipynb](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FExample_of_a_small_Multi-Player_Simulation__with_Centralized_Algorithms.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FExample_of_a_small_Multi-Player_Simulation__with_Centralized_Algorithms.ipynb)
- [Example_of_a_small_Multi-Player_Simulation__with_rhoRand_and_Selfish_Algorithms.ipynb](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FExample_of_a_small_Multi-Player_Simulation__with_rhoRand_and_Selfish_Algorithms.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FExample_of_a_small_Multi-Player_Simulation__with_rhoRand_and_Selfish_Algorithms.ipynb)
- [Lai_Robbins_Lower_Bound_for_Doubling_Trick_with_Restarting_Algorithms.ipynb](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FLai_Robbins_Lower_Bound_for_Doubling_Trick_with_Restarting_Algorithms.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FLai_Robbins_Lower_Bound_for_Doubling_Trick_with_Restarting_Algorithms.ipynb)
- [Exploring different doubling tricks for different kinds of regret bounds.ipynb](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FExploring_different_doubling_tricks_for_different_kinds_of_regret_bounds.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FExploring_different_doubling_tricks_for_different_kinds_of_regret_bounds.ipynb)
- [Experiments of statistical tests for piecewise stationary bandits.ipynb](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FExperiments_of_statistical_tests_for_piecewise_stationary_bandit.ipynb.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FExperiments_of_statistical_tests_for_piecewise_stationary_bandit.ipynb.ipynb)
- [Demonstrations of Single-Player Simulations for Non-Stationary-Bandits.ipynb](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FDemonstrations_of_Single-Player_Simulations_for_Non-Stationary-Bandits.ipynb)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SMPyBandits/SMPyBandits/master?filepath=notebooks%2FDemonstrations_of_Single-Player_Simulations_for_Non-Stationary-Bandits.ipynb)

### 2.c. Play with the _notebooks_ dynamically (on Google Colab) :boom:

Anyone can use the [colab.research.google.com/notebook](https://colab.research.google.com/notebook) website (by [clicking](XXX) on the icon above) to run the notebook in her/his web-browser.
You can then play with it as long as you like, for instance by modifying the values or experimenting with the code.

- [Do_we_even_need_UCB.ipynb](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Do_we_even_need_UCB.ipynb)  [![Google Colab](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Do_we_even_need_UCB.ipynb)
- [Easily_creating_MAB_problems.ipynb](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Easily_creating_MAB_problems.ipynb)  [![Google Colab](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Easily_creating_MAB_problems.ipynb)
- [Example_of_a_small_Single-Player_Simulation.ipynb](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb)  [![Google Colab](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb)
- [Example_of_a_small_Multi-Player_Simulation__with_Centralized_Algorithms.ipynb](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Multi-Player_Simulation__with_Centralized_Algorithms.ipynb)  [![Google Colab](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Multi-Player_Simulation__with_Centralized_Algorithms.ipynb)
- [Example_of_a_small_Multi-Player_Simulation__with_rhoRand_and_Selfish_Algorithms.ipynb](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Multi-Player_Simulation__with_rhoRand_and_Selfish_Algorithms.ipynb)  [![Google Colab](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Multi-Player_Simulation__with_rhoRand_and_Selfish_Algorithms.ipynb)
- [Lai_Robbins_Lower_Bound_for_Doubling_Trick_with_Restarting_Algorithms.ipynb](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Lai_Robbins_Lower_Bound_for_Doubling_Trick_with_Restarting_Algorithms.ipynb)  [![Google Colab](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Lai_Robbins_Lower_Bound_for_Doubling_Trick_with_Restarting_Algorithms.ipynb)
- [Exploring different doubling tricks for different kinds of regret bounds.ipynb](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Exploring_different_doubling_tricks_for_different_kinds_of_regret_bounds.ipynb)  [![Google Colab](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Exploring_different_doubling_tricks_for_different_kinds_of_regret_bounds.ipynb)
- [Experiments of statistical tests for piecewise stationary bandits.ipynb](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Experiments_of_statistical_tests_for_piecewise_stationary_bandit.ipynb.ipynb)  [![Google Colab](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Experiments_of_statistical_tests_for_piecewise_stationary_bandit.ipynb.ipynb)
- [Demonstrations of Single-Player Simulations for Non-Stationary-Bandits.ipynb](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Demonstrations_of_Single-Player_Simulations_for_Non-Stationary-Bandits.ipynb)  [![Google Colab](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/SMPyBandits/SMPyBandits/blob/master/notebooks/Demonstrations_of_Single-Player_Simulations_for_Non-Stationary-Bandits.ipynb)


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

### 3.b. My numerical environment, [`SMPyBandits`](https://github.com/SMPyBandits/SMPyBandits/)

- First, install its dependencies (`pip install -r requirements`).
- Then, either install it (*not yet*), or be sure to work in the main folder.

> *Note:* it's probably better to use [*virtualenv*](https://virtualenv.pypa.io/), if you like it.
> I never really understood how and why virtualenv are useful, but if you know why, you should know how to use it.

----

### :information_desk_person: More information?
> - More information about [notebooks (on the documentation of IPython)](https://nbviewer.jupiter.org/github/ipython/ipython/blob/3.x/examples/Notebook/Index.ipynb) or [on the FAQ on Jupyter's website](https://nbviewer.jupyter.org/faq).
> - More information about [mybinder.org](http://mybinder.org/): on [this example repository](https://github.com/binder-project/example-requirements).


## :scroll: License ? [![GitHub license](https://img.shields.io/github/license/Naereen/notebooks.svg)](https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE)
All the notebooks in this folder are published under the terms of the [MIT License](https://lbesson.mit-license.org/) (file [LICENSE.txt](../LICENSE.txt)).
© [Lilian Besson](https://GitHub.com/Naereen), 2016-18.

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel)](https://GitHub.com/SMPyBandits/SMPyBandits/)
[![made-with-jupyter](https://img.shields.io/badge/Made%20with-Jupyter-1f425f.svg)](http://jupyter.org/) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)

[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://GitHub.com/Naereen/)
