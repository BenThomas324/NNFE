#!/bin/bash
# Edit these to change problems
problem="Poisson"
dim=1
bcs="Dirichlet"
data_type="poly"

### Will replace with CARDIAX eventually
# Creates alias for conda env
python_conda="/home/bthomas/anaconda3/envs/fenicsx/bin/python"

# Reads in main scripts from the specified problem
data="$(bash get_problem.sh $problem $dim $bcs)/gen_data.py"
$python_conda $data $data_type

exit 0