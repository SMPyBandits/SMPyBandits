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
# set -euo pipefail

for file in $(find . -type f -iname '*.py'); do
    if grep 'from doctest import testmod' "$file" >/dev/null; then
        clear
        echo -e "\n${red}Testing the file '$file'...${reset}"  # DEBUG
        ( python "$file" \
          || echo -e "\n${red}File '$file' had some errors...${reset}" ) \
          && echo -e "\n${green}Tested the file '$file'...${reset}"
        # read  # DEBUG
    fi
done