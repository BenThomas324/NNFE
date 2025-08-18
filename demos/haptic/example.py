
import numpy as onp
import jax
import os

from nnfe.control.natural import NNFE
from cardiax.solvers.newton import Newton_Solver

os.makedirs("results", exist_ok=True)

nnfe = NNFE("test_params.yaml")

nnfe.problem = nnfe.fe_handler.problem
# nnfe.train()

# nn_sols = jax.vmap(nnfe.evaluate)(nnfe.sampler.Y)

solver = Newton_Solver(nnfe.problem, onp.zeros((nnfe.problem.num_total_dofs_all_vars)))

temp_vars1 = nnfe.problem.internal_vars_surfaces[0][0][0]
temp_vars2 = nnfe.problem.internal_vars_surfaces[0][1][0]
fe_sols = []
for i in range(len(nnfe.sampler.Y)):
    nnfe.problem.internal_vars_surfaces = [[[nnfe.sampler.Y[i] * temp_vars1],
                                            [nnfe.sampler.Y[i] * temp_vars2]]]
    sol, info = solver.solve(max_iter=60)
    assert info[0]
    fe_sols.append(sol)

    # nnfe.problem.mesh[0].point_data["nn_sol"] = onp.array(nn_sols[i]).reshape(-1, 3)
    nnfe.problem.mesh[0].point_data["fe_sol"] = onp.array(fe_sols[-1]).reshape(-1, 3)
    nnfe.problem.mesh[0].write(f"results/test_{i:02}.vtk")

# for i in range(len(nnfe.sampler.Y)):
#     diffs = onp.linalg.norm(fe_sols[i].reshape(-1, 3) - nn_sols[i].reshape(-1, 3), axis=1)
#     print(f"L2 norm: {diffs.mean()}, Linf norm: {diffs.max()}")
