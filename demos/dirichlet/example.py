
import numpy as onp
import jax
import os
import time

import cardiax
from nnfe.control.natural import NNFE
from cardiax.solvers.newton import Newton_Solver

cardiax.set_jax_enable_x64(False)

os.makedirs("results", exist_ok=True)

nnfe = NNFE("inputs/test_params.yaml")

toc = time.time()
nnfe.train()
tic = time.time()
training_time = tic - toc

# nn_sols = jax.vmap(nnfe.evaluate)(nnfe.sampler.Y)

# solver = Newton_Solver(nnfe.problem, onp.zeros_like(nn_sols[0]))

# temp_vars = nnfe.problem.internal_vars_surfaces[0][0][0]
# fe_sols = []
# for i in range(len(nnfe.sampler.Y)):
#     nnfe.problem.internal_vars_surfaces = [[[nnfe.sampler.Y[i] * temp_vars]]]
#     sol, info = solver.solve(max_iter=40)
#     fe_sols.append(sol)

#     nnfe.problem.mesh[0].point_data["nn_sol"] = onp.array(nn_sols[i]).reshape(-1, 3)
#     nnfe.problem.mesh[0].point_data["fe_sol"] = onp.array(fe_sols[-1]).reshape(-1, 3)
#     nnfe.problem.mesh[0].write(f"results/test_{i:02}.vtk")

# for i in range(len(nnfe.sampler.Y)):
#     diffs = onp.linalg.norm(fe_sols[i].reshape(-1, 3) - nn_sols[i].reshape(-1, 3), axis=1)
#     print(f"L2 norm: {diffs.mean()}, Linf norm: {diffs.max()}")

print(f"Training time: {training_time:.2f} seconds")
