# -*- coding: utf-8 -*-
"""
An example of a configuration file to launch some the simulations, for the single-player case.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


CPU_COUNT = 4

from os import getenv
import numpy as np

# Import arms
from Arms import *
# Import contained classes
from Environment import MAB
# Import single-player algorithms
from Policies import *

#: HORIZON : number of time steps of the experiments.
#: Warning Should be >= 10000 to be interesting "asymptotically".
HORIZON = 10000
HORIZON = int(getenv('T', HORIZON))

#: REPETITIONS : number of repetitions of the experiments.
#: Warning: Should be >= 10 to be statistically trustworthy.
REPETITIONS = 10
REPETITIONS = int(getenv('N', REPETITIONS))

#: To profile the code, turn down parallel computing
DO_PARALLEL = True

#: Number of jobs to use for the parallel computations. -1 means all the CPU cores, 1 means no parallelization.
N_JOBS = -1 if DO_PARALLEL else 1
N_JOBS = int(getenv('N_JOBS', N_JOBS))


#: Number of arms for non-hard-coded problems (Bayesian problems)
NB_ARMS = 3
NB_ARMS = int(getenv('K', NB_ARMS))
NB_ARMS = int(getenv('NB_ARMS', NB_ARMS))

#: Default value for the lower value of means
LOWER = 0.
#: Default value for the amplitude value of means
AMPLITUDE = 1.

#: Type of arms for non-hard-coded problems (Bayesian problems)
ARM_TYPE = "Bernoulli"
ARM_TYPE = str(getenv('ARM_TYPE', ARM_TYPE))
if ARM_TYPE in ["UnboundedGaussian"]:
    LOWER = -5
    AMPLITUDE = 10

LOWER = float(getenv('LOWER', LOWER))
AMPLITUDE = float(getenv('AMPLITUDE', AMPLITUDE))
assert AMPLITUDE > 0, "Error: invalid amplitude = {:.3g} but has to be > 0."  # DEBUG

ARM_TYPE_str = str(ARM_TYPE)
ARM_TYPE = mapping_ARM_TYPE[ARM_TYPE]

#: True to use bayesian problem
ENVIRONMENT_BAYESIAN = False
ENVIRONMENT_BAYESIAN = getenv('BAYES', str(ENVIRONMENT_BAYESIAN)) == 'True'

#: Means of arms for non-hard-coded problems (non Bayesian)
MEANS = uniformMeans(nbArms=NB_ARMS, delta=0.05, lower=LOWER, amplitude=AMPLITUDE, isSorted=True)

import numpy as np
# more parametric? Read from cli?
MEANS_STR = getenv('MEANS', '')
if MEANS_STR != '':
    MEANS = [ float(m) for m in MEANS_STR.replace('[', '').replace(']', '').split(',') ]
    print("Using cli env variable to use MEANS = {}.".format(MEANS))  # DEBUG


#: This dictionary configures the experiments
configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Other parameters for the Evaluator
    "finalRanksOnAverage": True,  # Use an average instead of the last value for the final ranking of the tested players
    "averageOn": 1e-3,  # Average the final rank on the 1.% last time steps
    # --- Should we plot the lower-bounds or not?
    "plot_lowerbounds": True,  # XXX Default
    # --- Arms
    "environment": [
        {   # Use vector from command line
            "arm_type": ARM_TYPE,
            "params": MEANS
        },
    ],
}

if ENVIRONMENT_BAYESIAN:
    configuration["environment"] = [  # XXX Bernoulli arms
        {   # A Bayesian problem: every repetition use a different mean vectors!
            "arm_type": ARM_TYPE,
            "params": {
                "function": randomMeans,
                "args": {
                    "nbArms": NB_ARMS,
                    "mingap": 1. / (3 * NB_ARMS),
                    "lower": LOWER,
                    "amplitude": AMPLITUDE,
                    "isSorted": True,
                }
            }
        },
    ]

try:
    #: Number of arms *in the first environment*
    nbArms = int(configuration['environment'][0]['params']['args']['nbArms'])
except (TypeError, KeyError):
    nbArms = len(configuration['environment'][0]['params'])

if len(configuration['environment']) > 1:
    print("WARNING do not use this hack if you try to use more than one environment.")

configuration.update({
    "policies": [
        # --- Stupid algorithms
        {
            "archtype": Uniform,   # The stupidest policy, fully uniform
            "params": {}
        },
        # # --- Full or partial knowledge algorithms
        { "archtype": TakeFixedArm, "params": { "armIndex": 0 }},  # Take worse arm!
        { "archtype": TakeFixedArm, "params": { "armIndex": 1 }},  # Take second worse arm!
        { "archtype": TakeFixedArm, "params": { "armIndex": min(2, nbArms - 1) }},  # Take third worse arm!
        {
            "archtype": UCB,   # UCB with alpha=1 parameter
            "params": {}
        },
        # --- Thompson algorithms
        {
            "archtype": Thompson,
            "params": {}
        },
        # --- KL UCB algorithms
        {
            "archtype": klUCB,
            "params": {}
        },
    ]}
)

# DONE
print("Loaded experiments configuration from 'example_of_configuration_singleplayer.py' :")
print("configuration =", configuration)  # DEBUG
