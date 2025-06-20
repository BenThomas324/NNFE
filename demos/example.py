
import numpy as onp
import jax

from nnfe.control.natural import Natural_NNFE
from cardiax.solvers.newton import Newton_Solver

nnfe = Natural_NNFE("test_params.yaml")

nnfe.train()

nn_sols = jax.vmap(nnfe.evaluate)(nnfe.sampler.Y)

solver = Newton_Solver(nnfe.problem, onp.zeros_like(nn_sols[0]))

temp_vars = nnfe.problem.internal_vars_surfaces[0][0][0]
fe_sols = []
for i in range(len(nnfe.sampler.Y)):
    nnfe.problem.internal_vars_surfaces = [[[nnfe.sampler.Y[i] * temp_vars]]]
    sol, info = solver.solve(max_iter=40)
    fe_sols.append(sol)
    print(onp.linalg.norm(sol.reshape(-1, 3) - nn_sols[i].reshape(-1, 3), axis=1).max())

nnfe.problem.mesh[0].point_data["fe_sol"] = onp.array(fe_sols[-1]).reshape(-1, 3)
nnfe.problem.mesh[0].write("test.vtk")
