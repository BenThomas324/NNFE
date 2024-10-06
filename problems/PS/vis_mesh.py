
import pyvista as pv

pv.OFF_SCREEN = True
pv.start_xvfb()

mesh_cube = pv.read("/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/PS/cube_mesh.vtk")
mesh_PS = pv.read("/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/PS/PS_mesh.vtk")

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_mesh(mesh_cube, opacity=0.5)
pl.subplot(0, 1)
pl.add_mesh(mesh_PS, opacity=0.5)
pl.screenshot("test.png")

mesh_cube = pv.read("/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/PS/cube_mesh_IF.vtk")
mesh_PS = pv.read("/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/PS/PS_mesh_IF.vtk")
mesh_cube.set_active_scalars("u")
mesh_PS.set_active_scalars("u")

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_mesh(mesh_cube, opacity=0.5)
pl.subplot(0, 1)
pl.add_mesh(mesh_PS, opacity=0.5)
pl.screenshot("test2.png")
