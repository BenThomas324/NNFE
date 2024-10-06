
import os
import sys
import yaml
import numpy as np

parent = os.path.dirname(__file__)
sys.path.append(os.path.dirname(parent) + "/src")
from problem_setup import *
from env_var import path
sys.path.append(path)
from jax_fem.solver import solver
from jax_fem.utils import save_sol

param_file = parent + "/params.yaml"
with open(param_file, 'r') as f:
    params = yaml.safe_load(f)

p, TCa = 100, 0.
pLV, pRV = p, 0.2 * p

problem, fiber_dirs, normals = VHL_setup(params["FE"], parent)

problem.internal_vars = [TCa * np.ones_like(fiber_dirs[0])[:, :, :1], *fiber_dirs]
pressures = [pLV * np.ones_like(normals[0])[:, :, :, :1], pRV * np.ones_like(normals[1])[:, :, :, :1]]
problem.internal_vars_surfaces = [[normals[0], pressures[0]], [normals[1], pressures[1]]]

sol = solver(problem, line_search_flag=True)
save_sol(problem.fes[0], sol[0], "vtus/Sim.vtu")