
import numpy as onp
import jax.numpy as np
import jax
import sys
import time
import yaml
import equinox as eqx
import os
import importlib

jax.config.update("jax_enable_x64", True)

from NNFE.nnfe.ml import *
from utils import *
from jax_fem.solver_abc import Newton_Solver

# prob_dir = sys.argv[1]
prob_dir = "/home/bthomas/Desktop/Research/NNFE/NNFE/problems/PS"
sys.path.append(prob_dir)

results_dir = "/home/bthomas/Desktop/Research/NNFE/NNFE/results/PS/31465"
jax.config.update("jax_enable_x64", True)

param_file = results_dir + "/params.yaml"
with open(param_file, 'r') as f:
    params = yaml.safe_load(f)

opt_params = params["Optimizer"]
NN_params = params["Network"]
data_params = params["Data"]

from setup import FE_data

fe_data = FE_data()
problem, internal_vars, internal_vars_surfaces = fe_data.fe_setup()
X = fe_data.get_testing_data()

if NN_params["kwargs"]["out_size"] == "dofs":
    NN_params["kwargs"]["out_size"] = problem.fes[0].num_total_dofs

models = []
for i in range(2):
    NN_params["load_model"] = results_dir + f"/model{i}.eqx"
    models.append(create_network(NN_params, 0))

from nnfe.plotting import NNFE_vis

solver = Newton_Solver(problem, np.zeros(problem.num_total_dofs_all_vars))
dofs = models[0](X[0]) + models[1](X[1])
dofs = solver.assign_bc(dofs)
problem.internal_vars = internal_vars
problem.internal_vars_surfaces = internal_vars_surfaces
solver.initial_guess = dofs
sol, info = solver.solve()

print(np.abs(dofs - sol).mean())
NNFE_vis(problem.fes[0], dofs, sol, results_dir + "/test.png")
