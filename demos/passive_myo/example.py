
import numpy as onp
import jax
import os
import jax.numpy as np

import cardiax
from nnfe.control.natural import NNFE
from cardiax.solvers.newton import Newton_Solver

cardiax.set_jax_enable_x64(False)

os.makedirs("results", exist_ok=True)

nnfe = NNFE("inputs/test_params.yaml")

# Adding fiber directions to Problem
def_fiber = lambda x: np.array([np.cos(x), 
                                np.sin(x), 
                                0])

def_sheet = lambda x: np.array([-np.sin(x), 
                                np.cos(x), 
                                0])

def_normal = lambda x: np.array([0., 0., 1.])

def theta_val(x):
    return (60 - (1 - x[2]) * 120) * np.pi / 180

quads = nnfe.fe_handler.fes["u"].get_physical_quad_points()
thetas = jax.vmap(jax.vmap(theta_val))(quads)

fibers = jax.vmap(jax.vmap(def_fiber))(thetas)
sheets = jax.vmap(jax.vmap(def_sheet))(thetas)
normals = jax.vmap(jax.vmap(def_normal))(thetas)

nnfe.fe_handler.problem.mesh["u"].cell_data["fibers"] = onp.array([fibers.mean(axis=1)])
nnfe.fe_handler.problem.mesh["u"].cell_data["sheets"] = onp.array([sheets.mean(axis=1)])
nnfe.fe_handler.problem.mesh["u"].cell_data["normals"] = onp.array([normals.mean(axis=1)])
nnfe.fe_handler.problem.mesh["u"].write("fibers_check.vtu")

int_vars = {"u": {"fibers": fibers,
                  "sheets": sheets,
                  "normals": normals}}
nnfe.problem.set_internal_vars(int_vars)

nnfe.train()

# nn_sols = jax.vmap(nnfe.evaluate)(nnfe.sampler.Y)

# solver = Newton_Solver(nnfe.problem, onp.zeros_like(nn_sols[0]))

# temp_vars = nnfe.problem.internal_vars_surfaces[0][0][0]
# fe_sols = []
# for i in range(len(nnfe.sampler.Y)):
#     nnfe.problem.internal_vars_surfaces = [[[nnfe.sampler.Y[i] * temp_vars]]]
#     sol, info = solver.solve(max_iter=40)
#     fe_sols.append(sol)

#     nnfe.problem.mesh["u"].point_data["nn_sol"] = onp.array(nn_sols[i]).reshape(-1, 3)
#     nnfe.problem.mesh["u"].point_data["fe_sol"] = onp.array(fe_sols[-1]).reshape(-1, 3)
#     nnfe.problem.mesh["u"].write(f"results/test_{i:02}.vtk")

# for i in range(len(nnfe.sampler.Y)):
#     diffs = onp.linalg.norm(fe_sols[i].reshape(-1, 3) - nn_sols[i].reshape(-1, 3), axis=1)
#     print(f"L2 norm: {diffs.mean()}, Linf norm: {diffs.max()}")
