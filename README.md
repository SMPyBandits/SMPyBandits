# TODO
- fork it
- clean up code
- pass to Python 3.5
- fully document it
- publish it on GitHub and readthedocs

# AdBandits
AdBandit: A New Algorithm For Multi-Armed Bandits

This repository contains the code for the algorithm described in the article: [AdBandit:
A New Algorithm For Multi-Armed Bandits] (http://sites.poli.usp.br/p/fabio.cozman/Publications/Article/truzzi-silva-costa-cozman-eniac2013.pdf) published in ENIAC2013.

Most of the code comes from the [pymabandits](http://mloss.org/software/view/415/) project, but some of them were refactored.

Unfortunately I didn't refactored everything, so you should encounter some javaish variable names all around.

I added the joblib in the Evaluator class, so the simulations can be parallelized.

## Configuration:

I use a simple python file: ```configuration.py```.

Example:

```python
configuration = {
    # Finite horizon of the simulation
    "horizon": 10000,
    # number of repetitions
    "repetitions": 100,
    # Number of cores for parallelization
    "n_jobs": 4,
    # Verbosity for the joblib
    "verbosity": 5,
    # Environment configuration, yeah you can set up more than one.
    # I striped some code that were not published yet, but you can implement
    # your own arms.
    "environment": [
        {
            "arm_type": Bernoulli,
            "probabilities": [0.02, 0.02, 0.02, 0.10, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01]
        }
    ],
    # Policies that should be simulated, and their parameters.
    "policies": [
        {
            "archtype": UCB,
            "params": {}
        },
        {
            "archtype": Thompson,
            "params": {}
        },
        {
            "archtype": klUCB,
            "params": {}
        },
        {
            "archtype": AdBandit,
            "params": {
                "alpha": 0.5,
                "horizon": 10000
            }
        }
    ]
}
```

## How to run

It should be very straight forward. It will plot the results.

```bash
pip install -r requirements.txt

python main.py
```

