
import numpy as onp
import jax
import os
import jax.numpy as np

from nnfe.control.natural import NNFE
from cardiax.solvers.newton import Newton_Solver

os.makedirs("results", exist_ok=True)

nnfe = NNFE("test_params.yaml")

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

quads = nnfe.fe_handler.fe.get_physical_quad_points()
thetas = jax.vmap(jax.vmap(theta_val))(quads)

fibers = jax.vmap(jax.vmap(def_fiber))(thetas)
sheets = jax.vmap(jax.vmap(def_sheet))(thetas)
normals = jax.vmap(jax.vmap(def_normal))(thetas)

nnfe.fe_handler.problem.mesh[0].cell_data["fibers"] = onp.array([fibers.mean(axis=1)])
nnfe.fe_handler.problem.mesh[0].cell_data["sheets"] = onp.array([sheets.mean(axis=1)])
nnfe.fe_handler.problem.mesh[0].cell_data["normals"] = onp.array([normals.mean(axis=1)])
nnfe.fe_handler.problem.mesh[0].write("fibers_check.vtu")

nnfe.problem.internal_vars = [fibers, sheets, normals]

nnfe.train()

nn_sols = jax.vmap(nnfe.evaluate)(nnfe.sampler.Y)

solver = Newton_Solver(nnfe.problem, onp.zeros_like(nn_sols[0]))

temp_vars = nnfe.problem.internal_vars_surfaces[0][0][0]
fe_sols = []
for i in range(len(nnfe.sampler.Y)):
    nnfe.problem.internal_vars_surfaces = [[[nnfe.sampler.Y[i] * temp_vars]]]
    sol, info = solver.solve(max_iter=40)
    fe_sols.append(sol)

    nnfe.problem.mesh[0].point_data["nn_sol"] = onp.array(nn_sols[i]).reshape(-1, 3)
    nnfe.problem.mesh[0].point_data["fe_sol"] = onp.array(fe_sols[-1]).reshape(-1, 3)
    nnfe.problem.mesh[0].write(f"results/test_{i:02}.vtk")

for i in range(len(nnfe.sampler.Y)):
    diffs = onp.linalg.norm(fe_sols[i].reshape(-1, 3) - nn_sols[i].reshape(-1, 3), axis=1)
    print(f"L2 norm: {diffs.mean()}, Linf norm: {diffs.max()}")
