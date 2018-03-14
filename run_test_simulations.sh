#!/usr/bin/env bash
# Author: Lilian BESSON, (C) 2018
# Email: Lilian.BESSON[AT]centralesupelec[DOT]fr
# Date: 13/03/2018.
#
# Run some test simulations for SMPyBandits.
# Cf. https://github.com/SMPyBandits/SMPyBandits/
#
# Usage: ./run_test_simulations.sh
#

# More details at http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

export K=9
export N=4
export T=1000  # FIXME increase
export DEBUG=False
export PLOT_DIR="/tmp/SMPyBandits/plots/"
mkdir -p "$PLOT_DIR"  # XXX
export SAVEALL=False
export N_JOBS=4
export BAYES=False
export LOWER=0
export AMPLITUDE=1
export ARM_TYPE=Bernoulli
export M=3
export S=3

# allsingleplayer test
clear
echo -e "\n\n\nLaunching 'make allsingleplayer'"
N=4 T=1000 make allsingleplayer
# read  # DEBUG

# allsingleplayer test
clear
echo -e "\n\n\nLaunching 'make allsingleplayer'"
N=4 T=1000 BAYES=True make allsingleplayer
# read  # DEBUG

# single test
clear
echo -e "\n\n\nLaunching 'make single'"
make single
# read  # DEBUG

# single test
clear
echo -e "\n\n\nLaunching 'make single'"
ARM_TYPE=Gaussian make single
# read  # DEBUG

# multi test
clear
echo -e "\n\n\nLaunching 'make multi'"
make multi
# read  # DEBUG

# moremulti test
clear
echo -e "\n\n\nLaunching 'make moremulti'"
M=3 make moremulti
# read  # DEBUG

# moremulti test
clear
echo -e "\n\n\nLaunching 'make moremulti'"
M=6 make moremulti
# read  # DEBUG

# moremulti test
clear
echo -e "\n\n\nLaunching 'make moremulti'"
M=9 make moremulti
# read  # DEBUG

# moremulti test
clear
echo -e "\n\n\nLaunching 'make moremulti'"
LOWER=-10 AMPLITUDE=20 ARM_TYPE=Gaussian M=3 make moremulti
# read  # DEBUG

# sparsemulti test
clear
echo -e "\n\n\nLaunching 'make sparsemulti'"
M=12 make sparsemulti
# read  # DEBUG

# comparing_aggregation_algorithms test
clear
echo -e "\n\n\nLaunching 'make comparing_aggregation_algorithms'"
make comparing_aggregation_algorithms
# read  # DEBUG

# comparing_doubling_algorithms test
clear
echo -e "\n\n\nLaunching 'make comparing_doubling_algorithms'"
make comparing_doubling_algorithms
# read  # DEBUG

# sparse test
clear
echo -e "\n\n\nLaunching 'make sparse'"
LOWER=-10 AMPLITUDE=20 ARM_TYPE=Gaussian S=3 make sparse
# read  # DEBUG

# sparse test
clear
echo -e "\n\n\nLaunching 'make sparse'"
LOWER=-10 AMPLITUDE=20 ARM_TYPE=Gaussian S=10 K=50 make sparse
# read  # DEBUG

# markovian test
clear
echo -e "\n\n\nLaunching 'make markovian'"
N=4 make markovian
# read  # DEBUG

# treeexploration test
clear
echo -e "\n\n\nLaunching 'make treeexploration'"
DEPTH=5 M=2 K=2 FIND_ONLY_N=1 make treeexploration
DEPTH=5 M=2 K=3 FIND_ONLY_N=1 make treeexploration
DEPTH=5 M=3 K=3 FIND_ONLY_N=1 make treeexploration
# read  # DEBUG

# clean up
git checkout -- logs
