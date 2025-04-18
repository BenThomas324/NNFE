#!/bin/bash

# Edit these to change problems
problem="PS"

# Creates alias for conda env
python_conda="/home/bthomas/anaconda3/envs/sciml/bin/python"

# Reads in main scripts from the specified problem
prob="$(bash get_problem.sh $problem)"
main="$(dirname $(pwd))/nnfe/train.py"
check="$(bash get_problem.sh $problem)/check.py"

echo $prob
echo "$(dirname $(pwd))"
# $prob2="$(realpath --relative-to="$prob" "$(dirname $(pwd))")"
# echo $prob2
# exit

$python_conda $main $prob > temp.txt
file="$(tail -n1 temp.txt)"
echo $file
$python_conda $check $file
