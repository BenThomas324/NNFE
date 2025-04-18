
import numpy as onp
import jax.numpy as np
import jax
import sys
import time
import yaml
import equinox as eqx
import os
prob_dir = os.path.dirname(__file__)

jax.config.update("jax_enable_x64", True)

from NNFE.nnfe.ml import *
from nnfe.utils import *
from jax_fem.solver import Newton_Solver

results_dir = "/home/bthomas/Desktop/Research/NNFE/NNFE/problems/PS/results/21369"
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

NN_params["load_model"] = results_dir + f"/model.eqx"
model = create_network(NN_params, 0)

from nnfe.plotting import NNFE_vis

problem.internal_vars = internal_vars
problem.internal_vars_surfaces = internal_vars_surfaces
solver = Newton_Solver(problem, np.zeros((problem.fes[0].num_total_dofs)))

dofs = jax.vmap(model)(X)
dofs = jax.vmap(solver.assign_bc)(dofs)

calc_res = fe_data.get_res(problem, internal_vars, internal_vars_surfaces)
res = jax.vmap(calc_res)(dofs, X)

norms = np.linalg.norm(res, axis=1)
ind_mean = np.argmin(np.abs(norms - norms.mean()))
ind_max = np.argmax(norms)

for ind in [ind_mean, ind_max]:
    TCa, p = X[ind]
    internal_vars[0] = TCa * np.ones_like(internal_vars[0])
    internal_vars_surfaces[0][0][1] = p * np.ones_like(internal_vars_surfaces[0][0][1])

    problem.internal_vars = internal_vars
    problem.internal_vars_surfaces = internal_vars_surfaces
    solver = Newton_Solver(problem, np.zeros(problem.num_total_dofs_all_vars))
    sol, info = solver.solve()

    print(np.abs(dofs[ind] - sol).mean())
    NNFE_vis(problem.fes[0], dofs[ind], sol, results_dir + f"/{ind}.png")
