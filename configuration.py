# -*- coding: utf-8 -*-
# Import arms
from Arms.Bernoulli import Bernoulli
# Import algorithms
from Policies import UCB
from Policies import Thompson
from Policies import klUCB
from Policies import AdBandit
from Policies import Aggr


HORIZON = 10000
REPETITIONS = 10

# DO_PARALLEL = False
DO_PARALLEL = True


configuration = {
    "horizon": HORIZON,
    "repetitions": REPETITIONS,
    "n_jobs": -1 if DO_PARALLEL else 1,    # = nb of CPU cores
    "verbosity": 5,  # Max joblib verbosity
    "environment": [
        {
            "arm_type": Bernoulli,
            # "probabilities": [0.02, 0.02, 0.02, 0.10, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01]
            "probabilities": [0.19, 0.2, 0.3, 0.4, 0.5, 0.6, 0.79, 0.8, 0.81]
        }
    ],
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
                "horizon": HORIZON
            }
        },
        {
            "archtype": Aggr,
            "params": {
                "learningRate": 0.1,  # FIXME improve
                "policies": [
                    {
                        "archtype": UCB,
                        "params": {}
                    },
                    {
                        "archtype": UCB,
                        "params": {}
                    },
                    # {
                    #     "archtype": Thompson,
                    #     "params": {}
                    # },
                    # {
                    #     "archtype": klUCB,
                    #     "params": {}
                    # },
                ]
            }
        }
    ]
}
