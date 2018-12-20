#!/usr/bin/env bash
# Author: Lilian BESSON, (C) 2018
# Email: Lilian.BESSON[AT]centralesupelec[DOT]fr
# Date: 22/09/2018.
#
# Run all doctests in all Python files that have doctest in the subdirectories.
#
# Usage: ./run_all_doctest.sh
#

# More details at http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

for file in $(find . -type f -iname '*.py'); do
    if grep 'from doctest import testmod' "$file"; then
        clear
        echo -e "\n${green}Testing the file '$file'...${reset}"  # DEBUG
        python3 "$file"
        # echo -e "\n${green}Testing the file '$file'...${reset}"  # DEBUG
        # read  # DEBUG
    fi
done