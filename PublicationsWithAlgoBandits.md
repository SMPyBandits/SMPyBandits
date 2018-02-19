# List of research publications using Lilian Besson's AlgoBandits project

[I (Lilian Besson)](http://perso.crans.org/besson/) have [started my PhD](http://perso.crans.org/besson/phd/) in October 2016, and [this project](https://github.com/Naereen/AlgoBandits/) is a part of my **on going** research since December 2016.

----

## 1st article, using the [**policy aggregation algorithm**](Aggregation.md)
I designed and added the [`Aggregator`](Policies/Aggregator.py) policy, in order to test its validity and performance.

It is a "simple" **voting algorithm to combine multiple bandit algorithms into one**.
Basically, it behaves like a simple MAB bandit just based on empirical means (even simpler than UCB), where *arms* are the child algorithms `A_1 .. A_N`, each running in "parallel".

> **For more details**, refer to this file: [`Aggregation.md`](Aggregation.md) and [this research article](https://hal.inria.fr/hal-01705292).

----

## 2nd article, using [**Multi-players simulation environment**](MultiPlayers.md)
There is another point of view: instead of comparing different single-player policies on the same problem, we can make them play against each other, in a multi-player setting.
The basic difference is about **collisions** : at each time `t`, if two or more user chose to sense the same channel, there is a *collision*. Collisions can be handled in different way from the base station point of view, and from each player point of view.

> **For more details**, refer to this file: [`MultiPlayers.md`](MultiPlayers.md) and [this research article](https://hal.inria.fr/hal-01629733).

----

## 3rd article, using [**Doubling Trick for Multi-Armed Bandits**](DoublingTrick.md)
I studied what Doubling Trick can and can't do to obtain efficient anytime version of non-anytime optimal Multi-Armed Bandits algorithms.

> **For more details**, refer to this file: [`DoublingTrick.md`](DoublingTrick.md) and [this research article](https://hal.inria.fr/hal-XXX).

----

## Other interesting things
### [Single-player Policies](http://banditslilian.gforge.inria.fr/docs/Policies.html)
- [`SparseWrapper`](http://banditslilian.gforge.inria.fr/docs/Policies.SparseWrapper.html#module-Policies.SparseWrapper) is a generalization of [the SparseUCB from this article](https://arxiv.org/pdf/1706.01383/).
- Implementation of very recent Multi-Armed Bandits algorithms, e.g., [`kl-UCB++`](http://banditslilian.gforge.inria.fr/docs/Policies.klUCBPlusPlus.html) (from [this article](https://hal.inria.fr/hal-01475078)), [`UCB-dagger`](http://banditslilian.gforge.inria.fr/docs/Policies.UCBdagger.html) (from [this article](https://arxiv.org/pdf/1507.07880)),  or [`MOSS-anytime`](http://banditslilian.gforge.inria.fr/docs/Policies.MOSSAnytime.html) (from [this article](http://proceedings.mlr.press/v48/degenne16.pdf)).
- Experimental policies: [`BlackBoxOpt`](http://banditslilian.gforge.inria.fr/docs/Policies.BlackBoxOpt.html) or [`UnsupervisedLearning`](http://banditslilian.gforge.inria.fr/docs/Policies.UnsupervisedLearning.html) (using Gaussian processes to learn the arms distributions).