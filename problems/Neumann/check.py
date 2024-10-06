### Do imports of modules used ###
import numpy as onp
import jax.numpy as np
import jax
import sys
import time
import yaml
import equinox as eqx

sys.path.append("/home/bthomas/Desktop/Research/JAXFEM_temp/jax-fem-NNFE")
from jax_fem.utils import save_sol
from jax_fem.solver import apply_bc_vec, assign_bc, solver

sys.path.append("/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite")
from FE_helpers import *
from NN_helpers import *
from problem_setup import *
from utils import *

param_file = "/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/hp_template.yaml"
with open(param_file, 'r') as f:
    params = yaml.safe_load(f)

NN_params = params["Network"]

problem, normals = Neumann_test()
X = np.array([2])

if NN_params["kwargs"]["out_size"] == "dofs":
    NN_params["kwargs"]["out_size"] = problem.fes[0].num_total_dofs

model = create_network(NN_params, 0)
model = eqx.tree_deserialise_leaves("/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/results/Neumann_test/49367/model.eqx", model)

pressures = X * np.ones_like(normals)[:, :, :, :1]
problem.internal_vars_surfaces = [[normals, pressures]]

from jax_fem.solver import assign_bc, solver
from jax_fem.utils import save_sol
dofs = model(X)
dofs = assign_bc(dofs, problem)
dofs = problem.unflatten_fn_sol_list(dofs)[0]

sol = solver(problem, initial_guess=dofs, line_search_flag=True)

print(np.linalg.norm(dofs - sol[0]))
save_sol(problem.fes[0], dofs, "Save/NNFE.vtu")
save_sol(problem.fes[0], sol[0], "Save/FE.vtu")
print()
