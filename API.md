# Short documentation of the API
> This short document aim at documenting the API used in [my AlgoBandits environment](https://github.com/Naereen/AlgoBandits/), and closing [this issue #3](https://github.com/Naereen/AlgoBandits/issues/3).

----

## Question: *How to change the simulations*?
### To customize the plots
1. Change the default settings defined in [`Environment/plotsettings.py`](Environment/plotsettings.py).

### To change the configuration of the simulations
1. Change the config file, i.e., [`configuration.py`](configuration.py) for single-player simulations, or [`configuration_multiplayers.py`](configuration_multiplayers.py) for multi-players simulations.

### To change how to results are exploited
1. Change the main script, i.e., [`main.py`](main.py) for single-player simulations, [`main_multiplayers.py`](main_multiplayers.py) for multi-players simulations. Some plots can be disabled or enabled by commenting a few lines, and some options are given as flags (constants in the beginning of the file).
2. If needed, change, improve or add some methods to the simulation environment class, i.e., [`Environment.Evaluator`](Environment/Evaluator.py) for single-player simulations, and [`Environment.EvaluatorMultiPlayers`](Environment/EvaluatorMultiPlayers.py) for multi-players simulations. They use a class to store their simulation result, [`Environment.Result`](Environment/Result.py) and [`Environment.ResultMultiPlayers`](Environment/ResultMultiPlayers.py).

----

## Question: *How to add something to this project?*
> In other words, *what's the API of this project*?

### For a **new arm**
1. Make a new file, e.g., `MyArm.py`
2. Save it in [`Arms/`](Arms/)
3. The file should contain a class of the same name, inheriting from [`Arms/Arm`](Arms/Arm.py), e.g., like this `class MyArm(Arm): ...` (no need for any [`super`](https://stackoverflow.com/questions/576169/ddg#576183) call)
4. This class `MyArm` has to **have at least** an `__init__(...)` method to create the arm object (with or without arguments - named or not); a `__str__` method to print it as a string; a `draw(t)` method to draw a reward from this arm (`t` is the time, which can be used or not); and **should have** a `mean()` method that gives/computes the mean of the arm
5. Finally, add it to the [`Arms/__init__.py`](Arms/__init__.py) file: `from .MyArm import MyArm`

> - For examples, see [`Arms.Bernoulli`](Arms/Bernoulli.py), [`Arms.Gaussian`](Arms/Gaussian.py), [`Arms.Exponential`](Arms/Exponential.py), [`Arms.Poisson`](Arms/Poisson.py).

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
3. The file should contain a class of the same name, it can inherit from [`Policies/IndexPolicy`](Policies/IndexPolicy.py) if it is a simple [index policy](Policies/IndexPolicy.py), e.g., like this, `class MyPolicy(IndexPolicy): ...` (no need for any [`super`](https://stackoverflow.com/questions/576169/ddg#576183) call), or simply like `class MyPolicy(object): ...`
4. This class `MyPolicy` has to **have at least** an `__init__(nbArms, ...)` method to create the policy object (with or without arguments - named or not), with **at least** the parameter `nbArms` (number of arms); a `__str__` method to print it as a string; a `choice()` method to choose an arm (index among `0, ..., nbArms - 1`, e.g., at random, or based on a maximum index if it is an [index policy](Policies/IndexPolicy.py)); and a `getReward(arm, reward)` method called when the arm `arm` gave the reward `reward`, and finally a `startGame()` method (possibly empty) which is called when a new simulation is ran.
5. Optionally, a policy class can have a `handleCollision(arm)` method to handle a collision after choosing the arm `arm` (eg. update an internal index, change a fixed offset etc).
6. Finally, add it to the [`Policies/__init__.py`](Policies/__init__.py) file: `from .MyPolicy import MyPolicy`

> - For examples, see [`Arms.Uniform`](Arms/Uniform.py) for a fully randomized policy, [`Arms.EpsilonGreedy`](Arms/EpsilonGreedy.py) for a simple exploratory policy, [`Arms.Softmax`](Arms/Softmax.py) for another simple approach, [`Arms.UCB`](Arms/UCB.py) for the class Upper Confidence-Bounds policy (based on indexes, so inheriting from [`Policies/IndexPolicy`](Policies/IndexPolicy.py)). There is also [`Arms.Thompson`](Arms/Thompson.py) and [`Arms.BayesUCB`](Arms/BayesUCB.py) for Bayesian policies (using a posterior, e.g., like [`Arms.Beta`](Arms/Beta.py)), [`Arms.klUCB`](Arms/klUCB.py) for a policy based on the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
> - For less classical [`Arms.AdBandit`](Arms/AdBandit.py) is an approach combining Bayesian and frequentist point of view, and [`Arms.Aggragorn`](Arms/Aggragorn.py) is [my aggregating policy](Aggregation.md).

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

### For a **new multi-users policy** - FIXME finish this doc
1. Make a new file, e.g., `MyPolicyMultiPlayers.py`
2. Save it in [`PoliciesMultiPlayers/`](PoliciesMultiPlayers/)
3. The file should contain a class, of the same name, e.g., like this, `class MyPoliciesMultiPlayers(object):`
4. This class `MyPoliciesMultiPlayers` has to **have at least** an `__init__` method to create the arm; a `__str__` method to print it as a string; and a `children` **attribute** that gives a list of *players* ([single-player policies](#for-a-new-single-user-policy)).
5. Finally, add it to the [`PoliciesMultiPlayers/__init__.py`](PoliciesMultiPlayers/__init__.py) file: `from .MyPoliciesMultiPlayers import MyPoliciesMultiPlayers`

> For examples, see [`PolicyMultiPlayers.OracleNotFair`](PolicyMultiPlayers/OracleNotFair.py) and [`PolicyMultiPlayers.OracleFair`](PolicyMultiPlayers/OracleFair.py) for full-knowledge centralized policies (fair or not), [`PolicyMultiPlayers.CentralizedFixed`](PolicyMultiPlayers/CentralizedFixed.py) and [`PolicyMultiPlayers.CentralizedCycling`](PolicyMultiPlayers/CentralizedCycling.py) for non-full-knowledge centralized policies (fair or not). There is also the [`PolicyMultiPlayers.Selfish`](PolicyMultiPlayers/Selfish.py) decentralized policy, where all players runs in without any knowledge on the number of players, and no communication (decentralized).

> [`PolicyMultiPlayers.Selfish`](PolicyMultiPlayers/Selfish.py) is the simplest possible example I could give as a template.

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
