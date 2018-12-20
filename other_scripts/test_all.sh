#!/usr/bin/env bash

set -euo pipefail

python --version
cd SMPyBandits/

export N=10
export T=1000

# # Testing basic things
# DEBUG=False NOPLOTS=True N=$N T=$T ipython3 example_of_main_singleplayer.py very_simple_configuration
# DEBUG=False NOPLOTS=True N=$N T=$T ipython3 example_of_main_singleplayer.py
# DEBUG=False NOPLOTS=True N=$N T=$T ipython3 example_of_main_multiplayers_more.py

# Testing single player
DEBUG=False NOPLOTS=True N=$N T=$T ipython3 main.py
DEBUG=False NOPLOTS=True N=$N T=$T ipython3 main.py configuration_comparing_aggregation_algorithms
DEBUG=False NOPLOTS=True N=$N T=$T ipython3 main.py configuration_comparing_doubling_algorithms
DEBUG=False NOPLOTS=True N=$N T=$T ipython3 main.py configuration_nonstationary
DEBUG=False NOPLOTS=True N=$N T=$T ipython3 main.py configuration_sparse
DEBUG=False NOPLOTS=True N=$N T=$T ipython3 main.py configuration_markovian
DEBUG=False NOPLOTS=True N=$N T=$T ipython3 main.py configuration_all_singleplayer

# Testing multi player
DEBUG=False NOPLOTS=True N=$N T=$T ipython3 main_multiplayers.py
DEBUG=False NOPLOTS=True N=$N T=$T ipython3 main_multiplayers_more.py
DEBUG=False NOPLOTS=True N=$N T=$T ipython3 main_multiplayers_more.py configuration_multiplayers_with_aggregation
DEBUG=False NOPLOTS=True N=$N T=$T ipython3 main_sparse_multiplayers.py

bash ./other_scripts/run_all_doctest.sh

cd Policies/
python kullback.py