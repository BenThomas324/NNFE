

from pathlib import Path
import cardiac_geometries
from mpi4py import MPI
import pyvista as pv
import meshio
import numpy as np
import shutil

pv.OFF_SCREEN = True
pv.start_xvfb()

geodir = Path("mesh/lv_ellipsoid")

if geodir.exists():
    shutil.rmtree(geodir)
cardiac_geometries.mesh.lv_ellipsoid(r_short_endo = 1.6, r_short_epi = 2.6,
                                        r_long_endo = 5.5, r_long_epi = 6.5,
                                        mu_base_endo = -np.arccos(1 / 5.5), mu_base_epi = -np.arccos(1/6.5),
                                    outdir=geodir, create_fibers=True,
                                    fiber_space="P_1", psize_ref=1.5)

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

meshio_domain = meshio.read("/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/LV/mesh/lv_ellipsoid/lv_ellipsoid.msh")
domain = meshio.read("/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/LV/mesh/lv_ellipsoid/lv_ellipsoid.msh")

domain.cell_sets, domain.point_data = {}, {}
domain.field_data, domain.cell_data = {}, {}

x_max = 2 * np.abs(domain.points[:, 0]).max()
y_max = 2 * np.abs(domain.points[:, 1]).max()
z_max = 2 * np.abs(domain.points[:, 2]).max()

# Get meshtags for geometry

for i, f in enumerate(meshio_domain.field_data):
    id_tag = meshio_domain.field_data[f][0]
    cells_id = np.argwhere(meshio_domain.cell_data_dict["gmsh:physical"]["triangle"] == id_tag)
    points = np.unique(meshio_domain.cells_dict["triangle"][cells_id])

    u = np.zeros((len(domain.points), 1))
    u[points] = 1
    domain.point_data[f] = u

    p = pv.Plotter()
    p.add_mesh(domain)
    p.add_axes()
    p.set_position([x_max, y_max, z_max])
    p.screenshot(f"mesh/test{f}.png")

# # Get fiber orientations
# domain.point_data = {}

fenicsx_pts = geo.mesh.geometry.x
meshio_pts = domain.points

dolfinx_to_meshio = np.zeros((len(meshio_pts)), dtype=int)
for i, p in enumerate(meshio_pts):
    dolfinx_to_meshio[i] = np.argwhere((p == fenicsx_pts).all(axis=1))

fiber_points = geo.f0.x.array.reshape(-1, 3)[dolfinx_to_meshio]
sheet_points = geo.s0.x.array.reshape(-1, 3)[dolfinx_to_meshio]
normal_points = geo.n0.x.array.reshape(-1, 3)[dolfinx_to_meshio]
vecs = {"fibers": fiber_points,
        "sheets": sheet_points,
        "normals": normal_points}

domain.point_data["fibers"] = fiber_points
domain.point_data["sheets"] = sheet_points
domain.point_data["normals"] = normal_points

counter = 0
for i, obj in enumerate(domain.cells):
    if obj.type == "vertex" or obj.type == "line" or obj.type == "triangle":
        counter += 1

domain.cells = domain.cells[counter:]
domain.write("mesh/LV_complete.vtu")

domain.write("mesh/fibers.vtk")
grid = pv.read("mesh/fibers.vtk")

for v in vecs:
    grid.point_data.set_vectors(vecs[v], v)

    p = pv.Plotter()
    p.add_mesh(grid.arrows)
    p.add_mesh(grid)
    p.add_axes()
    p.set_position([x_max, y_max, z_max])
    p.screenshot(f"mesh/test{v}.png")








