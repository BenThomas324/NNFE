

import os
import sys
import yaml
import numpy as np

parent = os.path.dirname(__file__)

from nnfe.problem_setup import LV_test
from nnfe.env_var import path

from jax_fem.solver import solver
from jax_fem.utils import save_sol

param_file = parent + "/params.yaml"
with open(param_file, 'r') as f:
    params = yaml.safe_load(f)

p, TCa = 10., 30.

problem, fibers, normals = PS_test(params["FE"])
problem.internal_vars = [TCa * np.ones_like(fibers[0])[:, :, :1], *fibers]
pressures = p * np.ones_like(normals)[:, :, :, :1]

problem.internal_vars_surfaces = [[normals, pressures]]

sol = solver(problem, line_search_flag=True)
save_sol(problem.fes[0], sol[0], "vtus/Sim.vtu")













