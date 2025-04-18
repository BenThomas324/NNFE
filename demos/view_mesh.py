
import pyvista as pv
import meshio
import numpy as np

pv.start_xvfb()
pv.OFF_SCREEN = True

mesh = meshio.read('msh/mesh.xdmf')
epi_mesh = meshio.read('msh/epi.xdmf')
endo_mesh = meshio.read('msh/endo.xdmf')
fiber_mesh = meshio.read('msh/preferred_direction.xdmf')

epi_to_mesh = []
for point in mesh.points:
    distances = np.linalg.norm(epi_mesh.points - point, axis=1)
    closest_point_index = np.argmin(distances)
    epi_to_mesh.append(closest_point_index)
epi_to_mesh = np.array(epi_to_mesh)

endo_nodes = np.unique(endo_mesh.cells_dict['triangle'])
epi_nodes = np.unique(epi_mesh.cells_dict['triangle'])

endo_tag = np.zeros((mesh.points.shape[0],), dtype=int)
epi_tag = np.zeros((mesh.points.shape[0],), dtype=int)
endo_tag[endo_nodes] = 1
epi_tag[epi_nodes] = 1

endo_tag = endo_tag[epi_to_mesh]
epi_tag = epi_tag[epi_to_mesh]

mesh.point_data["endo"] = endo_tag
mesh.point_data["epi"] = epi_tag
mesh.point_data["base"] = (mesh.points[:, 2] == 0.).astype(int)

pvmesh = pv.from_meshio(mesh)

pl = pv.Plotter(shape=(1, 3))
pl.subplot(0, 0)
pl.add_mesh(pvmesh, scalars='endo', show_edges=True, copy_mesh=True)
pl.subplot(0, 1)
pl.add_mesh(pvmesh, scalars='epi', show_edges=True, copy_mesh=True)
pl.subplot(0, 2)
pl.add_mesh(pvmesh, scalars='base', show_edges=True)
pl.screenshot('test_mesh.png')

mesh.cell_data = {}
mesh.cell_data["fibers"] = fiber_mesh.cell_data["preferred_direction"]
mesh.cell_data["sheets"] = fiber_mesh.cell_data["crossed_direction"]
mesh.cell_data["normals"] = fiber_mesh.cell_data["normal_direction"]

pvfiber = pv.from_meshio(fiber_mesh)

arrows1 = pvfiber.glyph(orient="preferred_direction", scale="preferred_direction", factor=1.)
arrows2 = pvfiber.glyph(orient="crossed_direction", scale="crossed_direction", factor=1.)
arrows3 = pvfiber.glyph(orient="normal_direction", scale="normal_direction", factor=1.)

pl = pv.Plotter(shape=(1, 3))
pl.subplot(0, 0)
pl.add_mesh(arrows1)
pl.subplot(0, 1)
pl.add_mesh(arrows2)
pl.subplot(0, 2)
pl.add_mesh(arrows3)
pl.screenshot('test_fibers.png')

mesh.write('msh/LV_mesh.xdmf')

