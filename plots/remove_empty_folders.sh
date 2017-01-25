#!/bin/bash
for i in $(find . -type d); do
    n=$(ls -larth $i | wc -l)
    if [ $n -eq 3 ]; then
        echo -e "\nEmpty directory: $i"
        ls -larth $i
        rm -rvi $i
    fi
done
