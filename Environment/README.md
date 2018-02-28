# [Environments](https://smpybandits.github.io/docs/Environment.html)
> See here the documentation: [docs/Environment](https://smpybandits.github.io/docs/Environment.html)

- [`MAB`](MAB.py), [`MarkovianMAB`](MarkovianMAB.py), [`DynamicMAB`](DynamicMAB.py) and [`IncreasingMAB`](IncreasingMAB.py) objects, used to wrap the problems (list of arms).
- [`Result`](Result.py) and [`ResultMultiPlayers`](ResultMultiPlayers.py) objects, used to wrap simulation results (list of decisions and rewards).
- [`Evaluator`](Evaluator.py) environment, used to wrap simulation, for the single player case.
- [`EvaluatorMultiPlayers`](EvaluatorMultiPlayers.py) environment, used to wrap simulation, for the multi-players case.
- [`EvaluatorSparseMultiPlayers`](EvaluatorSparseMultiPlayers.py) environment, used to wrap simulation, for the multi-players case with sparse activated players.
- [`CollisionModels`](CollisionModels.py) implements different collision models.

And useful constants and functions for the plotting and stuff are in the [`__init__.py`](__init__.py) file:

- `DPI`, `signature`, `maximizeWindow`, `palette`, `makemarkers`, `wraptext`: for plotting
- `notify`: send a desktop notification
- `Parallel`, `delayed`: joblib related
- `tqdm`: pretty range() loops
- `sortedDistance`, `fairnessMeasures`: science related