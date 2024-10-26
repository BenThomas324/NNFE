
import numpy as onp
import jax.numpy as np
import jax
import sys
import time
import yaml
import equinox as eqx
import os
import importlib

from NN_helpers import *
from utils import *
from jax_fem.solver_abc import apply_bc_vec

# prob_dir = sys.argv[1]
prob_dir = "/home/bthomas/Desktop/Research/NNFE/NNFE/problems/PS"
sys.path.append(prob_dir)

results_dir = "/home/bthomas/Desktop/Research/NNFE/NNFE/results/PS/22673"

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

NN_params["load_model"] = results_dir + "/model.eqx"
model = create_network(NN_params, 0)

from jax_fem.plotting import visualize

dofs = model(X[8])
calc_res = fe_data.get_res(problem, internal_vars, internal_vars_surfaces)
res_vec = calc_res(dofs, X[8])
# res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
res_vec = apply_bc_vec(res_vec, dofs, problem)

print(np.linalg.norm(res_vec))

visualize(problem.fes[0], dofs.reshape(-1, 3), results_dir + "/test.png")
