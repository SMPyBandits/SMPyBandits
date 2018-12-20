#!/usr/bin/env bash

set -euo pipefail

python --version

bash ./other_scripts/run_all_doctest.sh

cd SMPyBandits/
export N=10
export T=1000

# # Testing basic things
# N=$N T=$T ipython3 example_of_main_singleplayer.py very_simple_configuration
# N=$N T=$T ipython3 example_of_main_singleplayer.py
# N=$N T=$T ipython3 example_of_main_multiplayers_more.py

# Testing single player
NOPLOTS=True N=$N T=$T ipython3 main.py
NOPLOTS=True N=$N T=$T ipython3 main.py configuration_comparing_aggregation_algorithms
NOPLOTS=True N=$N T=$T ipython3 main.py configuration_comparing_doubling_algorithms
NOPLOTS=True N=$N T=$T ipython3 main.py configuration_nonstationary
NOPLOTS=True N=$N T=$T ipython3 main.py configuration_sparse
NOPLOTS=True N=$N T=$T ipython3 main.py configuration_markovian
NOPLOTS=True N=$N T=$T ipython3 main.py configuration_all_singleplayer

# Testing multi player
NOPLOTS=True N=$N T=$T ipython3 main_multiplayers.py
NOPLOTS=True N=$N T=$T ipython3 main_multiplayers_more.py
NOPLOTS=True N=$N T=$T ipython3 main_multiplayers_more.py configuration_multiplayers_with_aggregation
NOPLOTS=True N=$N T=$T ipython3 main_sparse_multiplayers.py

cd Policies/
python kullback.py