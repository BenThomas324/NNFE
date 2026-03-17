
import pyvista as pv
import numpy as onp

from nnfe import NNFE

parent = "test/57312/"
nnfe = NNFE.from_yaml(parent + "config_files/resolved_nnfe_config.yaml")

diffs = []
mesh = nnfe.problem.mesh["u"]
for i, x in enumerate(nnfe.sampler.Y):
    nn_sol, fe_sol = nnfe.test(x)

    mesh.point_data["nn_sol"] = onp.array(nn_sol).reshape(-1, 3)
    mesh.point_data["fe_sol"] = onp.array(fe_sol).reshape(-1, 3)
    mesh.save(parent + f"results/test_{i:02d}.vtk")

    diffs.append(onp.linalg.norm(onp.array(nn_sol) - onp.array(fe_sol), axis=-1))

print(onp.array(diffs).mean())
print(onp.array(diffs).max())
print(onp.array(diffs))