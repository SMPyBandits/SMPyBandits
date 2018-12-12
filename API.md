# Short documentation of the API
> This short document aim at documenting the API used in [my SMPyBandits environment](https://github.com/SMPyBandits/SMPyBandits/), and closing [this issue #3](https://github.com/SMPyBandits/SMPyBandits/issues/3).

----

## Code organization
### Layout of the code:
- Arms are defined in [this folder (`Arms/`)](Arms/), see for example [`Arms.Bernoulli`](SMPyBandits/Arms/Bernoulli.py)
- MAB algorithms (also called policies) are defined in [this folder (`Policies/`)](Policies/), see for example [`Policies.Dummy`](SMPyBandits/Policies/Dummy.py) for a fully random policy, [`Policies.EpsilonGreedy`](SMPyBandits/Policies/EpsilonGreedy.py) for the epsilon-greedy random policy, [`Policies.UCB`](SMPyBandits/Policies/UCB.py) for the "simple" UCB algorithm, or also [`Policies.BayesUCB`](SMPyBandits/Policies/BayesUCB.py), [`Policies.klUCB`](SMPyBandits/Policies/klUCB.py) for two UCB-like algorithms, [`Policies.AdBandits`](SMPyBandits/Policies/AdBandits.py) for the [AdBandits](https://github.com/flaviotruzzi/AdBandits/) algorithm, and [`Policies.Aggregator`](SMPyBandits/Policies/Aggregator.py) for my *aggregated bandits* algorithms.
- Environments to encapsulate date are defined in [this folder (`Environment/`)](Environment/): MAB problem use the class [`Environment.MAB`](SMPyBandits/Environment/MAB.py), simulation results are stored in a [`Environment.Result`](SMPyBandits/Environment/Result.py), and the class to evaluate multi-policy single-player multi-env is [`Environment.Evaluator`](SMPyBandits/Environment/Evaluator.py).
- [very_simple_configuration.py`](SMPyBandits/configuration.py) imports all the classes, and define the simulation parameters as a dictionary (JSON-like).
- [`main.py`](SMPyBandits/main.py) runs the simulations, then display the final ranking of the different policies and plots the results (saved to [this folder (`plots/`)](plots/)).

----

### UML diagrams
> For more details, see [these UML diagrams](uml_diagrams/).

----

## Question: *How to change the simulations*?
### To customize the plots
1. Change the default settings defined in [`Environment/plotsettings.py`](SMPyBandits/Environment/plotsettings.py).

### To change the configuration of the simulations
1. Change the config file, i.e., [`configuration.py`](SMPyBandits/configuration.py) for single-player simulations, or [`configuration_multiplayers.py`](SMPyBandits/configuration_multiplayers.py) for multi-players simulations.
2. A good example of a very simple configuration file is given in [very_simple_configuration.py`](SMPyBandits/very_simple_configuration.py)

### To change how to results are exploited
1. Change the main script, i.e., [`main.py`](SMPyBandits/main.py) for single-player simulations, [`main_multiplayers.py`](SMPyBandits/main_multiplayers.py) for multi-players simulations. Some plots can be disabled or enabled by commenting a few lines, and some options are given as flags (constants in the beginning of the file).
2. If needed, change, improve or add some methods to the simulation environment class, i.e., [`Environment.Evaluator`](SMPyBandits/Environment/Evaluator.py) for single-player simulations, and [`Environment.EvaluatorMultiPlayers`](SMPyBandits/Environment/EvaluatorMultiPlayers.py) for multi-players simulations. They use a class to store their simulation result, [`Environment.Result`](SMPyBandits/Environment/Result.py) and [`Environment.ResultMultiPlayers`](SMPyBandits/Environment/ResultMultiPlayers.py).

----

## Question: *How to add something to this project?*
> In other words, *what's the API of this project*?

### For a **new arm**
1. Make a new file, e.g., `MyArm.py`
2. Save it in [`Arms/`](Arms/)
3. The file should contain a class of the same name, inheriting from [`Arms/Arm`](SMPyBandits/Arms/Arm.py), e.g., like this `class MyArm(Arm): ...` (no need for any [`super`](https://stackoverflow.com/questions/576169/ddg#576183) call)
4. This class `MyArm` has to **have at least** an `__init__(...)` method to create the arm object (with or without arguments - named or not); a `__str__` method to print it as a string; a `draw(t)` method to draw a reward from this arm (`t` is the time, which can be used or not); and **should have** a `mean()` method that gives/computes the mean of the arm
5. Finally, add it to the [`Arms/__init__.py`](SMPyBandits/Arms/__init__.py) file: `from .MyArm import MyArm`

> - For examples, see [`Arms.Bernoulli`](SMPyBandits/Arms/Bernoulli.py), [`Arms.Gaussian`](SMPyBandits/Arms/Gaussian.py), [`Arms.Exponential`](SMPyBandits/Arms/Exponential.py), [`Arms.Poisson`](SMPyBandits/Arms/Poisson.py).

> - For example, use this template:

```python
from .Arm import Arm

class MyArm(Arm):
    def __init__(self, *args, **kwargs):
        # TODO Finish this method that initialize the arm MyArm

    def __str__(self):
        return "MyArm(...)".format('...')  # TODO

    def draw(self, t=None):
        # TODO Simulates a pull of this arm. t might be used, but not necessarily

    def mean(self):
        # TODO Returns the mean of this arm
```

----

### For a **new (single-user) policy**
1. Make a new file, e.g., `MyPolicy.py`
2. Save it in [`Policies/`](Policies/)
3. The file should contain a class of the same name, it can inherit from [`Policies/IndexPolicy`](SMPyBandits/Policies/IndexPolicy.py) if it is a simple [index policy](SMPyBandits/Policies/IndexPolicy.py), e.g., like this, `class MyPolicy(IndexPolicy): ...` (no need for any [`super`](https://stackoverflow.com/questions/576169/ddg#576183) call), or simply like `class MyPolicy(object): ...`
4. This class `MyPolicy` has to **have at least** an `__init__(nbArms, ...)` method to create the policy object (with or without arguments - named or not), with **at least** the parameter `nbArms` (number of arms); a `__str__` method to print it as a string; a `choice()` method to choose an arm (index among `0, ..., nbArms - 1`, e.g., at random, or based on a maximum index if it is an [index policy](SMPyBandits/Policies/IndexPolicy.py)); and a `getReward(arm, reward)` method called when the arm `arm` gave the reward `reward`, and finally a `startGame()` method (possibly empty) which is called when a new simulation is ran.
5. Optionally, a policy class can have a `handleCollision(arm)` method to handle a collision after choosing the arm `arm` (eg. update an internal index, change a fixed offset etc).
6. Finally, add it to the [`Policies/__init__.py`](SMPyBandits/Policies/__init__.py) file: `from .MyPolicy import MyPolicy`

> - For examples, see [`Arms.Uniform`](SMPyBandits/Arms/Uniform.py) for a fully randomized policy, [`Arms.EpsilonGreedy`](SMPyBandits/Arms/EpsilonGreedy.py) for a simple exploratory policy, [`Arms.Softmax`](SMPyBandits/Arms/Softmax.py) for another simple approach, [`Arms.UCB`](SMPyBandits/Arms/UCB.py) for the class Upper Confidence-Bounds policy (SMPyBandits/based on indexes, so inheriting from [`Policies/IndexPolicy`](Policies/IndexPolicy.py)). There is also [`Arms.Thompson`](SMPyBandits/Arms/Thompson.py) and [`Arms.BayesUCB`](SMPyBandits/Arms/BayesUCB.py) for Bayesian policies (using a posterior, e.g., like [`Arms.Beta`](SMPyBandits/Arms/Beta.py)), [`Arms.klUCB`](SMPyBandits/Arms/klUCB.py) for a policy based on the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
> - For less classical [`Arms.AdBandit`](SMPyBandits/Arms/AdBandit.py) is an approach combining Bayesian and frequentist point of view, and [`Arms.Aggregator`](SMPyBandits/Arms/Aggregator.py) is [my aggregating policy](Aggregation.md).

> - For example, use this template:

```python
class MyPolicy(object):
    def __init__(self, nbArms, *args, **kwargs):
        self.nbArms = nbArms
        # TODO Finish this method that initialize the arm MyArm

    def __str__(self):
        return "MyArm(...)".format('...')  # TODO

    def startGame(self):
        pass  # Can be non-trivial, TODO if needed

    def getReward(self, arm, reward):
        # TODO After the arm 'arm' has been pulled, it gave the reward 'reward'
        pass  # Can be non-trivial, TODO if needed

    def choice(self):
        # TODO Do a smart choice of arm
        return random.randint(self.nbArms)

    def handleCollision(self, arm):
        pass  # Can be non-trivial, TODO if needed
```

> Other `choice...()` methods can be added, if this policy `MyPolicy` has to be used for multiple play, ranked play, etc.


----

### For a **new multi-users policy**
1. Make a new file, e.g., `MyPoliciesMultiPlayers.py`
2. Save it in [`PoliciesMultiPlayers/`](PoliciesMultiPlayers/)
3. The file should contain a class, of the same name, e.g., like this, `class MyPoliciesMultiPlayers(object):`
4. This class `MyPoliciesMultiPlayers` has to **have at least** an `__init__` method to create the arm; a `__str__` method to print it as a string; and a `children` **attribute** that gives a list of *players* ([single-player policies](#for-a-new-single-user-policy)).
5. Finally, add it to the [`PoliciesMultiPlayers/__init__.py`](SMPyBandits/PoliciesMultiPlayers/__init__.py) file: `from .MyPoliciesMultiPlayers import MyPoliciesMultiPlayers`

> For examples, see [`PoliciesMultiPlayers.OracleNotFair`](SMPyBandits/PoliiesyMultiPlayers/OracleNotFair.py) and [`PoliciesMultiPlayers.OracleFair`](SMPyBandits/PoliiesyMultiPlayers/OracleFair.py) for full-knowledge centralized policies (fair or not), [`PoliciesMultiPlayers.CentralizedFixed`](SMPyBandits/PoliiesyMultiPlayers/CentralizedFixed.py) and [`PoliciesMultiPlayers.CentralizedCycling`](SMPyBandits/PoliiesyMultiPlayers/CentralizedCycling.py) for non-full-knowledge centralized policies (fair or not). There is also the [`PoliciesMultiPlayers.Selfish`](SMPyBandits/PoliiesyMultiPlayers/Selfish.py) decentralized policy, where all players runs in without any knowledge on the number of players, and no communication (decentralized).

> [`PoliciesMultiPlayers.Selfish`](SMPyBandits/PoliiesyMultiPlayers/Selfish.py) is the simplest possible example I could give as a template.

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
