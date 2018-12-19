# -*- coding: utf-8 -*-
"""
An very simple configuration file to run some basic simulations about stationary multi-armed bandits.
"""

from Arms import *

from Environment import MAB

from Policies import *

# --- Parameters of the experiments
HORIZON = 30

REPETITIONS = 1

NB_ARMS = 5

ARM_TYPE = Bernoulli

# Like http://localhost/publis/tiny-d3-bandit-animation.git/index.html?T=30&MU=0.1,0.2,0.3,0.4,0.9
MEANS = [0.1, 0.2, 0.3, 0.4, 0.9]


#: This dictionary configures the experiments
configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": 1,         # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Other parameters for the Evaluator
    "finalRanksOnAverage": True,  # Use an average instead of the last value for the final ranking of the tested players
    "averageOn": 1e-3,  # Average the final rank on the 1.% last time steps
    # --- Should we plot the lower-bounds or not?
    "plot_lowerbounds": False,  # XXX Default
    # --- Arms
    "environment": [
        {   # Use vector from command line
            "arm_type": ARM_TYPE,
            "params": MEANS
        },
    ],
}

configuration.update({
    "policies": [
        # --- Full or partial knowledge algorithms
        { "archtype": TakeFixedArm, "params": { "armIndex": 0 }},  # Take worse arm!
        { "archtype": TakeFixedArm, "params": { "armIndex": 1 }},  # Take second worse arm!
        { "archtype": TakeFixedArm, "params": { "armIndex": 2 }},  # Take third worse arm!
        { "archtype": TakeFixedArm, "params": { "armIndex": 3 }},  # Take forth worse arm!
        { "archtype": TakeFixedArm, "params": { "armIndex": 4 }},  # Take fifth worse arm!
        # --- Stupid algorithms
        {
            "archtype": Uniform,   # The stupidest policy, fully uniform
            "params": {}
        },
        # --- UCB algorithm
        {
            "archtype": UCB,   # UCB with alpha=1 parameter
            "params": {}
        },
        # --- Thompson algorithm
        {
            "archtype": Thompson,
            "params": {}
        },
        # --- KL UCB algorithm
        {
            "archtype": klUCB,
            "params": {}
        },
        # --- BESA algorithm
        {
            "archtype": BESA,
            "params": {
                "horizon": HORIZON,
            }
        },
        # --- MOSS algorithm
        {
            "archtype": MOSS,
            "params": {}
        },
        # --- Exp3++ algorithm
        {
            "archtype": Exp3PlusPlus,
            "params": {}
        },
    ]}
)

# DONE
print("Loaded experiments configuration from 'example_of_configuration_singleplayer.py' :")
print("configuration =", configuration)  # DEBUG
