#!/bin/bash

# Edit these to change problems
problem="Dirichlet"

# Creates alias for conda env
python_conda="/home/bthomas/anaconda3/envs/sciml/bin/python"

# Reads in main scripts from the specified problem
prob="$(bash get_problem.sh $problem)"
solve="$(dirname $(pwd))/nnfe/solve.py"

echo $prob
echo $solve
exit

$python_conda $solve $prob > temp.txt
file="$(tail -n1 temp.txt)"
echo $file
