
import pyvista as pv
import os

pv.start_xvfb()
pv.OFF_SCREEN = True

parent = os.path.dirname(__file__)
mesh_file = parent + "/vtus/FE.vtu"
mesh = pv.read(mesh_file)

mesh.set_active_vectors("sol")

pl = pv.Plotter()
pl.add_mesh(mesh)
pl.screenshot(parent + "/FE.png")
