#!/bin/bash

# Parent string
wd="$(pwd)"
parent="$(dirname $wd)/problems"

# Input arguments
problem=$1
dim=$2
BCs=$3

if [[ $problem == "Poisson" ]]; then
    parent+="/Poisson"
elif [[ $problem == "Linear_Elastic" ]]; then
    parent+="/Linear_Elastic"
elif [[ $problem == "Hyperelastic" ]]; then
    parent+="/Hyperelastic"
else
    echo "Problem not supported"
    exit 2
fi

if [[ $dim == 1 ]]; then
    parent+="/1D"
elif [[ $dim == 2 ]]; then
    parent+="/2D"
elif [[ $dim == 3 ]]; then
    parent+="/3D"
else
    echo "Dimension not supported"
    exit 2
fi

if [[ $BCs == "Dirichlet" ]]; then
    parent+="/Dirichlet"
elif [[ $BCs == "Neumann" ]]; then
    parent+="/Neumann"
else
    echo "BCs not supported"
    exit 2
fi

echo $parent