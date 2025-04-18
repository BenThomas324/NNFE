

import os
import sys
import yaml
import numpy as np

parent = os.path.dirname(__file__)

from setup import FE_data

from jax_fem.solver import Newton_Solver
from jax_fem.utils import save_sol
from jax_fem.plotting import visualize_sol

# param_file = parent + "/params.yaml"
# with open(param_file, 'r') as f:
#     params = yaml.safe_load(f)

fe_class = FE_data()
problem, internal_vars, internal_vars_surfaces = fe_class.fe_setup()
problem.internal_vars = internal_vars
problem.internal_vars_surfaces = internal_vars_surfaces

solver = Newton_Solver(problem, np.zeros_like(problem.fes[0].mesh.points))
sol, info = solver.solve()
save_sol(problem.fes[0], sol.reshape(-1, 3), "Sim.vtu")

visualize_sol(problem.fes[0], sol, "test.png")











