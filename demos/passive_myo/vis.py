
import numpy as onp
import jax
import jax.numpy as np

from nnfe.control.natural import NNFE
from cardiax.solvers.newton import Newton_Solver
import pyvista as pv

nnfe = NNFE("test/80453/test_params.yaml")

nnfe.problem = nnfe.fe_handler.problem

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

nn_sols = jax.vmap(nnfe.evaluate)(nnfe.sampler.Y)

solver = Newton_Solver(nnfe.problem, onp.zeros_like(nn_sols[0]))

temp_vars = nnfe.problem.internal_vars_surfaces[0][0][0]
fe_sols = []
for i in range(len(nnfe.sampler.Y)):
    nnfe.problem.internal_vars_surfaces = [[[nnfe.sampler.Y[i] * temp_vars]]]
    sol, info = solver.solve(max_iter=40)
    fe_sols.append(sol)


for i in range(len(nnfe.sampler.Y)):
    diffs = onp.linalg.norm(fe_sols[i].reshape(-1, 3) - nn_sols[i].reshape(-1, 3), axis=1)
    print(f"L2 norm: {diffs.mean()}, Linf norm: {diffs.max()}")

print(nn_sols[-1])

from cardiax.IGA.post_process import get_F

import pyvista as pv
pv.OFF_SCREEN = True

# Prepare mesh and point data arrays for all steps
pvmesh = pv.from_meshio(nnfe.problem.mesh[0])
# pvmesh.point_data["scalar"] = onp.ones((len(nn_sols[-1].reshape(-1, 3)),))

# Create a plotter with off-screen rendering
pl = pv.Plotter(off_screen=True)
pl.open_gif("solutions.gif", fps=4)

for i in range(len(nnfe.sampler.Y)):
    # Update point data for each solution
    pvmesh.point_data["nn_sol"] = onp.array(nn_sols[i]).reshape(-1, 3)
    pvmesh.point_data["fe_sol"] = onp.array(fe_sols[i]).reshape(-1, 3)
    warped_fe = pvmesh.warp_by_vector("fe_sol", factor=1.0)
    warped_nn = pvmesh.warp_by_vector("nn_sol", factor=1.0)
    warped_fe.set_active_scalars(None)

    Fs = get_F(nnfe.fe_handler.fe, fe_sols[i])
    def_fibers = onp.einsum('ijkl, ijl->ijk', Fs, onp.array(fibers))
    warped_fe.cell_data["fibers"] = def_fibers.mean(axis=1)
    arrows = warped_fe.glyph(orient="fibers", scale=False, factor=0.2)

    pl.clear()
    pl.add_mesh(warped_fe, opacity=0.8, color="gainsboro", show_edges=True, scalars=None)
    pl.add_mesh(arrows, color="maroon")
    # pl.add_mesh(warped_nn, style="points", render_points_as_spheres=True, point_size=5, color="red")
    pl.camera_position = [1.5, 1., 1.2]
    pl.camera.focal_point = [0.5, 0.5, 0.5]
    pl.write_frame()

pl.close()