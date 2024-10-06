
### Do imports of modules used ###
import numpy as onp
import jax.numpy as np
import sys
import numpy as onp
import yaml
import equinox as eqx
import os

parent = os.path.dirname(__file__)
sys.path.append(os.path.dirname(parent) + "/src")
from FE_helpers import *
from NN_helpers import *
from problem_setup import *
from utils import *

from env_var import path
sys.path.append(path)
from jax_fem.solver import assign_bc, solver
from jax_fem.utils import save_sol

run_file = "/workspace/bthomas/NNFE/nnfe-scratch-rewrite/results/VHL/57613"
with open(run_file + "/params.yaml", 'r') as f:
    params = yaml.safe_load(f)

NN_params = params["Network"]
NN_params["load_model"] = run_file + "/model.eqx"

problem, fibers, normals = VHL_setup(params["FE"], parent)

X = onp.zeros((10, 3))
X[:, 0] = onp.linspace(0, 50, 10)
X[:, 1] = 0.2 * X[:, 0]

if NN_params["kwargs"]["out_size"] == "dofs":
    NN_params["kwargs"]["out_size"] = problem.fes[0].num_total_dofs

model = create_network(NN_params, 0)

test = X[5] #X[onp.random.randint(X.shape[0])]
pLV, pRV, TCa = test

problem.internal_vars = [TCa * onp.ones_like(fibers[0])[:, :, :1], *fibers]
pressures = [pLV * onp.ones_like(normals[0])[:, :, :, :1], pRV * onp.ones_like(normals[1])[:, :, :, :1]]
problem.internal_vars_surfaces = [[normals[0], pressures[0]], [normals[1], pressures[1]]]

dofs = model(test)
dofs = assign_bc(dofs, problem)
dofs = problem.unflatten_fn_sol_list(dofs)[0]

sol = solver(problem, line_search_flag=True)

print(onp.linalg.norm(dofs - sol[0]))
save_sol(problem.fes[0], dofs, "vtus/NNFE.vtu")
save_sol(problem.fes[0], sol[0], "vtus/FE.vtu")
print(test)



