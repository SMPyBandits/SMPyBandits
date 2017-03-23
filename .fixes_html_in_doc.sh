#!/usr/bin/env bash
# Author: Lilian BESSON, (C) 2017
# Email: Lilian.BESSON[AT]ens-cachan[DOT]fr
# Date: 23/03/2017.
#
# Change the links from .md to .html in the documentation html pages
#
# Usage: ./.fixes_html_in_doc.sh
#

BUILDDIR=_build/html

for i in "$BUILDDIR"/*.html; do
    echo -e "\n${yellow}Search for wrong links${white} in $i ..."
    grep --color=always '\.md"' "$i"
    echo -e "${green}OK to replace these '.md\"' by '.html\"' ??${white}"
    # read  # DEBUG
    sed -i.backup s/'\.md"'/'\.html"'/g "$i"
    # DEBUG
    echo -e "${blue}Modification done${white}, here is the difference (new, old)"
    wdiff -3 "$i" "$i".backup
    # read  # DEBUG
done

# End of newscript.sh
