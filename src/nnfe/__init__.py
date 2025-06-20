
from pyfiglet import Figlet
import importlib.metadata

f = Figlet(font='starwars')
print(f.renderText('NNFE'))

__version__ = importlib.metadata.version(__package__)

# from .IPA.fe import FiniteElement
# from .IPA.generate_mesh import rectangle_mesh
# from .IPA.generate_mesh import box_mesh
# from .IPA.generate_mesh import cylinder_mesh

# from .IGA.iga import BSpline
# from .IGA.generate_mesh import rect_prism_bspline
# from .IGA.generate_mesh import box_mesh_bspline

# from .problem import Problem
# from .solvers.newton import Newton_Solver

# __all__ = [
#     "BSpline",
#     "FiniteElement",
#     "Problem",
#     "Newton_Solver",
#     "rect_prism_bspline",
#     "box_mesh_bspline",
#     "rectangle_mesh",
#     "box_mesh",
#     "cylinder_mesh"
# ]
