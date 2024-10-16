#!/bin/bash

# Parent string
wd="$(pwd)"
parent="$(dirname $wd)/problems"

# Input arguments
problem=$1

if [[ $problem == "Dirichlet" ]]; then
    parent+="/Dirichlet"
elif [[ $problem == "PS" ]]; then
    parent+="/PS"
elif [[ $problem == "LV" ]]; then
    parent+="/LV"
elif [[ $problem == "VHL" ]]; then
    parent+="/VHL"
else
    echo "Problem not supported"
    exit 2
fi

echo $parent