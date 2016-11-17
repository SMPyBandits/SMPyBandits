# -*- coding: utf-8 -*-
# Import arms
from Arms.Bernoulli import Bernoulli
# Import algorithms
from Policies import UCB
from Policies import Thompson
from Policies import klUCB
from Policies import AdBandit
from Policies import Aggr


HORIZON = 5000
DO_PARALLEL  = False


configuration = {
    "horizon": HORIZON,
    "repetitions": 10,
    "n_jobs": -1 if DO_PARALLEL else 1,    # = nb of CPU cores
    "verbosity": 5,
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
        # FIXME finish to implement my "aggregated bandit" in Policies.Aggr
        {
            "archtype": Aggr,
            "params": {
                "learningRate": 1,  # FIXME improve
                "policies": [
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
