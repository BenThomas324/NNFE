
import sys

from jax_fem.solver_abc import solver
from jax_fem.plotting import visualize
from jax_fem.utils import save_sol

sys.path.append(sys.argv[1])

from setup import setup

problem, internal_vars, internal_vars_surfaces = setup()
if internal_vars is not None:
    problem.internal_vars = internal_vars
if internal_vars_surfaces is not None:
    problem.internal_vars_surfaces = internal_vars_surfaces

sol = solver(problem, use_petsc=False, line_search_flag=True)

visualize(problem.fes[0], sol.reshape(-1, 3), "test.png")
save_sol(problem.fes[0], sol.reshape(-1, 3), "sol.vtu")