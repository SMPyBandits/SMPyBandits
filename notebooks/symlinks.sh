#!/usr/bin/env /bin/bash
# Symlink every .html as fake .ipynb files in the output notebook directory of the documentation

cd _build/html/notebooks/ 2>/dev/null
cd ../_build/html/notebooks/ 2>/dev/null
pwd

function softsymlinkit() {
    html="$1"
    ipynb="${1%.html}.ipynb"
    echo -e "\nHTML file = $html and Jupyter notebook link = $ipynb"
    ls -larth "$html"
    if [ -f "$html" ]; then
        if [ -f "$ipynb" ]; then
            if [ ! -L "$ipynb" ]; then
                echo -e "${red}Error${white}: $ipynb already exist but it is not a previous symlink."
            else
                echo -e "${magenta}Warning${white}: $ipynb already exist but it is already a symlink... Forcing to update it"
                echo -e "${green}Symlinking${white}: $ipynb ----> $html"
                echo -e ln -s -f "$html" "$ipynb"
                # read  # DEBUG
                ln -s -f "$html" "$(basename "$ipynb")"
            fi
        else
            echo -e "${green}Symlinking${white}: $ipynb ----> $html"
            echo -e ln -s "$html" "$ipynb"
            # read  # DEBUG
            ln -s "$html" "$(basename "$ipynb")"
        fi
    else
        echo -e "${red}Error${white}: $html does not exist."
    fi
}

for i in *.html; do
    softsymlinkit "$i"
done
