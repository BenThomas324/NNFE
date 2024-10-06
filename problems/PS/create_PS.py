from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import log
import fenicsx_pulse
import cardiac_geometries.fibers as cgf
import os
import meshio
import functools as ftls
import numpy as np

parent = os.path.dirname(__file__)

tau_base = 0.1
tau_apex = -1
sigma_endo = 1.2
sigma_epi = 1.5

BL = np.array([sigma_endo, tau_apex, -np.pi])
TR = np.array([sigma_epi, tau_base, np.pi])

mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, points=[BL, TR], n=[2, 8, 15])

sigma, tau, phi = mesh.geometry.x.T

# mesh.topology.create_connectivity(2, 0)
# conn_2 = mesh.topology.connectivity(2, 0)
# triangles = [("triangle", conn_2.array.reshape(-1, 3))]

# endo_facets = dolfinx.mesh.locate_entities(
#     mesh, 2, lambda x: np.isclose(x[0], sigma.min())
# )
# base_facets = dolfinx.mesh.locate_entities(
#     mesh, 2, lambda x: np.isclose(x[1], tau.max())
# )
# v = np.zeros(mesh.topology.index_map(2).size_local, dtype=np.int32)
# v[endo_facets] = 2
# v[base_facets] = 1
# cell_data = {"subdomain": [v]}

# Create LV from cube
a = lambda t: 4
get_x = lambda s, t, p: a(t) * np.sqrt((s**2 - 1) * (1 - t**2)) * np.cos(p)
get_y = lambda s, t, p: a(t) * np.sqrt((s**2 - 1) * (1 - t**2)) * np.sin(p)
get_z = lambda s, t, p: a(t) * s * t

scaled_sig = lambda x: x
X = get_x(scaled_sig(sigma), tau, phi)
Y = get_y(scaled_sig(sigma), tau, phi)
Z = get_z(scaled_sig(sigma), tau, phi)

new_points = np.vstack((X, Y, Z)).T

### Glue LV together ###


mesh.topology.create_connectivity(3, 0)
conn = mesh.topology.connectivity(3, 0).array.reshape(-1, 4)
tets = np.take(new_points, conn, axis=0)
arrow1 = (tets[:, 0, :] - tets[:, -1, :])
arrow2 = (tets[:, 1, :] - tets[:, -1, :])
arrow3 = (tets[:, 2, :] - tets[:, -1, :])
vols = np.stack((arrow1, arrow2, arrow3), axis=1)
vols = np.linalg.det(vols)
m = np.argwhere(vols < 1e-4)
conn = np.delete(conn, m, axis=0)

repeats = {}
for i in range(new_points.shape[0]):
    if np.isclose(new_points[i], new_points).all(axis=1).sum() >= 2:
        vals = np.argwhere(np.isclose(new_points[i], new_points).all(axis=1))
        for j in repeats:
            if i in repeats[j]:
                break
        else:
            repeats[i] = vals

new_conn = np.copy(conn)
for key in repeats:
    for j in repeats[key]:
        mask = np.argwhere(j == conn)
        new_conn[mask[:, 0], mask[:, 1]] = key

unique_ids = np.unique(new_conn)
final_points = np.take(new_points, unique_ids, axis=0)
final_conn = np.copy(new_conn)
for i in range(len(unique_ids)):
    mask = np.argwhere(new_conn == unique_ids[i])
    final_conn[mask[:, 0], mask[:, 1]] = i

cells = [("tetra", final_conn)]

# Save LV to xdmf
# rot = np.array([[0, 0, 1],
#                 [0, 1, 0],
#                 [1, 0, 0]])
# final_points = final_points @ rot

meshio_LV_mesh = meshio.Mesh(final_points, cells)
meshio_LV_mesh.write(parent + "/LV_mesh.xdmf")

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, parent + "/LV_mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

# meshio_LV_facets = meshio.Mesh(new_points, triangles, cell_data=cell_data)
# meshio_LV_facets.write(parent + "/LV_facets.xdmf")
# meshio_LV_facets.write(parent + "/LV_facets.vtu")

endo_tag = sigma == sigma.min()
epi_tag = sigma == sigma.max()
base_tag = tau == tau.max()

base_pts = new_points[base_tag]
endo_pts = new_points[endo_tag]
epi_pts = new_points[epi_tag]
# Find boundaries
def check_bdry(pts, x):
    TF = np.zeros((x.shape[1]), dtype=bool)
    for i in range(x.shape[1]):
        check = np.isclose(x[:, i], pts).all(axis=1).any()
        TF[i] = check
    return TF

check_bdry_base = ftls.partial(check_bdry, base_pts)
check_bdry_endo = ftls.partial(check_bdry, endo_pts)
check_bdry_epi = ftls.partial(check_bdry, epi_pts)

boundaries = [
    fenicsx_pulse.Marker(name="ENDO", marker=1, dim=2, locator=check_bdry_endo),
    fenicsx_pulse.Marker(name="EPI", marker=2, dim=2, locator=check_bdry_epi),
    fenicsx_pulse.Marker(name="BASE", marker=3, dim=2, locator=check_bdry_base),
]

endo_facets = dolfinx.mesh.locate_entities(
    mesh, 2, check_bdry_endo
)
epi_facets = dolfinx.mesh.locate_entities(
    mesh, 2, check_bdry_epi
)
base_facets = dolfinx.mesh.locate_entities(
    mesh, 2, check_bdry_base
)

marked_facets = np.hstack([endo_facets, epi_facets, base_facets])
marked_values = np.hstack([np.full_like(endo_facets, 1), np.full_like(epi_facets, 2), np.full_like(base_facets, 3)])
sorted_facets = np.argsort(marked_facets)
facet_tag = dolfinx.mesh.meshtags(mesh, 2, marked_facets[sorted_facets], marked_values[sorted_facets])

geo = fenicsx_pulse.Geometry(
    mesh=mesh,
    boundaries=boundaries,
    metadata={"quadrature_degree": 4},
)

r_short_endo = a(tau_base) * np.sqrt((1 - tau_base**2) * (sigma_endo**2 - 1))
r_short_epi = a(tau_base) * np.sqrt((1 - tau_base**2) * (sigma_epi**2 - 1))
r_long_endo = np.abs(a(tau_apex) * tau_apex * sigma_endo)
r_long_epi = np.abs(a(tau_apex) * tau_apex * sigma_epi)

mesh.topology.create_connectivity(2, 3)
system = cgf.lv_ellipsoid.create_microstructure(
    mesh,
    ffun=facet_tag,
    markers=geo.markers,
    r_short_endo=r_short_endo,
    r_short_epi=r_short_epi,
    r_long_endo=r_long_endo,
    r_long_epi=r_long_epi,
    outdir="./",
    function_space="P_2",
    long_axis=2,
)

# Create/save meshes 
org_cells = mesh.topology.connectivity(3, 0).array.reshape(-1, 4)
meshio_cube = meshio.Mesh(mesh.geometry.x, [("tetra", org_cells)])
meshio_cube.write(parent + "/cube_mesh.vtk")

meshio_PS = meshio.Mesh(final_points, cells)
meshio_PS.write(parent + "/PS_mesh.vtk")

# Add 0 contraction area
# If we want to turn off a "cylinder" through the LV
# Choose tau close to -1 to have near apex
# then sweep through from -pi to pi for angle change

ball_center = np.array([-.5, -1/4*np.pi])
ball_radius = .25

in_ball = lambda x: np.sqrt((x[:, 1]-ball_center[0])**2 + 
                            (x[:, 2]-ball_center[1])**2)
mask = in_ball(meshio_cube.points) < ball_radius

infarct_pts = meshio_cube.points[mask]
X_data = get_x(*infarct_pts.T)
Y_data = get_y(*infarct_pts.T)
Z_data = get_z(*infarct_pts.T)

infarct_pts_PS = np.vstack((X_data, Y_data, Z_data)).T

infarct_region = np.zeros((mask.shape[0], 1))
infarct_region[mask] = 1

mask2 = []
for p in infarct_pts_PS:
    mask2.append(np.argwhere((p == final_points).all(axis=1))[0, 0])

infarct_region_PS = np.zeros((final_points.shape[0], 1))
infarct_region_PS[mask2] = 1

meshio_cube.point_data["u"] = infarct_region
meshio_PS.point_data["u"] = infarct_region_PS
meshio_cube.write(parent + "/cube_mesh_IF.vtk")
meshio_PS.write(parent + "/PS_mesh_IF.vtk")
