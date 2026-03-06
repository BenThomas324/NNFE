
"""
This script is made to run an input file.
Call (python run.py input_file.yaml)
"""

from cardiax import ProblemManager

config = "configs/fe_input_file.yaml"

fe_manager = ProblemManager.from_yaml(config)

sol, info = fe_manager.solve_problem()

import pyvista as pv
pv.start_xvfb()
pv.OFF_SCREEN = True

mesh = fe_manager.fes.mesh["u"]
mesh["sol"] = sol.reshape(-1, 3)
warped = mesh.warp_by_vector("sol", factor=1.)

pl = pv.Plotter()
pl.add_mesh(warped, scalars="sol", show_scalar_bar=True, cmap="coolwarm")
pl.show_grid()
pl.screenshot("test.png")
