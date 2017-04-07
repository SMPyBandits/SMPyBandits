#!/usr/bin/env bash
# Author: Lilian BESSON, (C) 2017
# Email: Lilian.BESSON[AT]ens-cachan[DOT]fr
# Date: 07/04/2017.
#
# Change the links from .md to .html in the documentation html pages
#
# Usage: ./.fixes_html_in_doc.sh
#

EMOJIZE=emojize.py
# EMOJIZE=emojize_pngorsvg.py
# EMOJIZE="emojize_pngorsvg.py --svg"

BUILDDIR=_build/html

for i in "$BUILDDIR"/*.html "$BUILDDIR"/*/*.html; do
    # 1. fix links from .md to .html
    # 2. OPTIONAL remove :emojis: in HTML output (from GFM Markdown), see https://stackoverflow.com/questions/42087466/sphinx-extension-to-use-github-markdown-emoji-in-sphinx#comment73617151_42256239
    #    uncomment the two s/':[a-z0-9_-]+: '/''/g and s/' :[a-z0-9_-]+:'/''/g lines below

    echo -e "\n${yellow}Search for wrong links to md files${white} in $i ..."
    grep --color=always '\.md"' "$i"
    # echo -e "${green}OK to replace these '.md\"' by '.html\"' ??${white}"
    # read  # DEBUG
    sed -i.backup -r \
        -e s/'\.md"'/'\.html"'/g \
        "$i"
        # -e s/':[a-z0-9_-]+: '/''/g \
        # -e s/' :[a-z0-9_-]+:'/''/g \
    # DEBUG
    echo -e "${blue}Modification done${white}, here is the difference (old, new)"
    wdiff -3 "$i".backup "$i"
    # read  # DEBUG
    mv -vf "$i".backup /tmp/

    # 3. fix link from Package/File.py to docs/Package.File.html
    echo -e "\n${yellow}Search for wrong links to Python files${white} in $i ..."
    grep --color=always 'href="[^"]*\.py"' "$i"
    # echo -e "${green}OK to replace these 'Package/File.py' by 'docs/Package.File.html' ??${white}"
    # read  # DEBUG
    sed -r \
        -e s_'href="([^"/]*)\.py"'_'href="docs/\1.html"'_g \
        -e s_'href="([^"/]*)/([^"/]*)\.py"'_'href="docs/\1.\2.html"'_g \
        -e s_'href="([^"/]*)/([^"/]*)/([^"/]*)\.py"'_'href="docs/\1.\2.\3.html"'_g \
        -e s_'href="([^"/]*)/([^"/]*)/([^"/]*)/([^"/]*)\.py"'_'href="docs/\1.\2.\3.\4.html"'_g \
        "$i" > "$i".new
    echo -e "${blue}Modification done${white}, here is the difference (old, new)"
    wdiff -3 "$i" "$i".new
    # read  # DEBUG

    # 3. convert :emojis: to UTF-8 in HTML output (from GFM Markdown), see https://stackoverflow.com/questions/42087466/sphinx-extension-to-use-github-markdown-emoji-in-sphinx#comment73617151_42256239
    if type $EMOJIZE &>/dev/null ; then
        $EMOJIZE "$i" > "$i".new
    fi
    wdiff -3 "$i" "$i".new
    # read  # DEBUG
    mv -vf "$i".new "$i"
done

# End of newscript.sh
