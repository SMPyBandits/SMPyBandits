# [Arms](https://smpybandits.github.io/docs/Arms.html)
> See here the documentation: [docs/Arms](https://smpybandits.github.io/docs/Arms.html)

Arms : contains different types of bandit arms:
[`Constant`](Constant.py), [`UniformArm`](UniformArm.py), [`Bernoulli`](Bernoulli.py), [`Binomial`](Binomial.py), [`Poisson`](Poisson.py), [`Gaussian`](Gaussian.py), [`Exponential`](Exponential.py), [`Gamma`](Gamma.py).

Each arm class follows the same interface:

```python
>>> my_arm = Arm(params)
>>> my_arm.mean
0.5
>>> my_arm.draw()  # one random draw
0.0
>>> my_arm.draw_nparray(20)  # or ((3, 10)), many draw
array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,
        1.,  0.,  0.,  0.,  1.,  1.,  1.])
```


Also the [`__init__.py`](__init__.py) file contains:

- `uniformMeans`, to generate uniformly spaced means of arms.
- `uniformMeansWithSparsity`, to generate uniformly spaced means of arms, with sparsity constraints.
- `randomMeans`, to generate randomly spaced means of arms.
- `randomMeansWithGapBetweenMbestMworst`, to generate randomly spaced means of arms, with a constraint on the gap between the M-best arms and the (K-M)-worst arms.
- `randomMeans`, to generate randomly spaced means of arms.
- `shuffled`, to return a shuffled version of a list.
- Utility functions `array_from_str` `list_from_str` and `tuple_from_str` to obtain a `numpy.ndarray`, a `list` or a `tuple` from a string (used for the CLI env variables interface).
- `optimal_selection_probabilities`.