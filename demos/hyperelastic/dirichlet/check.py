
import pyvista as pv
import numpy as onp

from nnfe import NNFE

parent = "test/16371/"
nnfe = NNFE(parent + "input_files/nnfe_params.yaml")

pvmesh = pv.from_meshio(nnfe.problem.mesh["u"])
for i, x in enumerate(nnfe.sampler.Y):
    nn_sol, fe_sol = nnfe.test(x)

    pvmesh.point_data["nn_sol"] = onp.array(nn_sol).reshape(-1, 3)
    pvmesh.point_data["fe_sol"] = onp.array(fe_sol).reshape(-1, 3)
    pvmesh.save(parent + f"results/test_{i:02d}.vtk")