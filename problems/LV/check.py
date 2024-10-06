
import equinox as eqx
import jax
import time
import numpy as np
import meshio
import sys
import yaml
import matplotlib.pyplot as plt

from nnfe.FE_helpers import *
from nnfe.problem_setup import *
from nnfe.NN_helpers import *

sys.path.append("/home/bthomas/Desktop/Research/JAXFEM_temp/jax-fem-NNFE")
from jax_fem.solver import solver, assign_bc, apply_bc_vec
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type

parent_file = "/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/results/LV/46233"

param_file = parent_file + "/params.yaml"
with open(param_file, 'r') as f:
    params = yaml.safe_load(f)

opt_params = params["Optimizer"]
NN_params = params["Network"]
FE_params = params["FE"]

problem, fibers, normals = LV_test(FE_params)

X1 = np.array([7.5, 15.])
# X2 = np.array([(3.75 + 2.5)/2, (7.5 + 11.25)/2])

model_name = parent_file + "/model.eqx"
if NN_params["kwargs"]["out_size"] == "dofs":
    NN_params["kwargs"]["out_size"] = problem.fes[0].num_total_dofs
model = create_network(NN_params, 0)
model = eqx.tree_deserialise_leaves(model_name, model)

pred = model(X1)
pred = assign_bc(pred, problem)
pred = problem.unflatten_fn_sol_list(pred)[0]


problem.internal_vars = [X1[1] * np.ones_like(fibers[0])[:, :, :1], *fibers]
pressures = X1[0] * np.ones_like(normals)[:, :, :, :1]
problem.internal_vars_surfaces = [[normals, pressures]]

sol = solver(problem, line_search_flag=True)[0]

diff = sol - pred
print("Pointwise max error: ", np.abs(diff).max())
print("L2 norm max error: ", np.linalg.norm(diff, axis=1).max())
print(np.abs(sol).max(), np.abs(pred).max())

from jax_fem.utils import save_sol
save_sol(problem.fes[0], pred, parent_file + "/NNFE.vtu")
save_sol(problem.fes[0], sol, parent_file + "/FE.vtu")

print("Done")