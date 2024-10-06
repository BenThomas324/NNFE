#!/bin/bash

# Edit these to change problems
problem="Poisson"
dim=1
bcs="Dirichlet"

# Creates alias for conda env
python_conda="/home/bthomas/anaconda3/envs/sciml/bin/python"

# Reads in main scripts from the specified problem
main="$(bash get_problem.sh $problem $dim $bcs)/main.py"
check="$(bash get_problem.sh $problem $dim $bcs)/check.py"

$python_conda $main > temp.txt
file="$(tail -n1 temp.txt)"
echo $file
$python_conda $check $file
