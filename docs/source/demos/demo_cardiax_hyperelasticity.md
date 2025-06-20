---
jupytext:
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
---

# CARDIAX Hyperelasticity Demo

Copyright (C) 2024 WCCMS

Classic hyperelastic cube twisting problem
The demo is implemented in a single [Python file](https://github.com/WCCMS-UTAustin/CARDIAX/blob/main/docs/source/demos/demo_cardiax_hyperelasticity.py), and can also be downloaded 
as a [jupyter notebook](https://github.com/WCCMS-UTAustin/CARDIAX/blob/main/docs/source/demos/demo_cardiax_hyperelasticity.ipynb). 

+++

__Imports__
<br>
First, some JAX and CARDIAX specific modules are imported:

```python
# Import some useful modules.
import jax
import jax.numpy as np
import os

# Import modules from CARDIAX.
from jax_fem.problem_abc import Problem
from jax_fem.solver_abc import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh #, get_meshio_cell_type, Mesh
from jax_fem.fe_abc import FiniteElement
from jax_fem.iga import BSpline

jax.config.update("jax_debug_nans", True)
```

__HyperElasticity Class Definition__
<br>
We use a {py:class}`HyperElasticity <jax_fem.demos.hyperelasticity.D_example>` class to define a problem:
The function 'get_tensor_map' is defined below; it overrides the base class method. Generally, JAX-FEM (and CARDIAX)
solves -$\nabla \cdot f(\nabla u) = b$. Here, we define $f(\nabla u$) = $\mathbf{P}$. Notice how we first 
define $\psi\$, and then use automatic differentiation (jax.grad) 
to obtain the 'P_fn' function.

```python
class HyperElasticity(Problem):
    """ Defines the hyperelastic material behavior, object mesh, boundary conditions, etc.. Includes 1 main function, get_tensor_map(self) with two nested functions: psi(F) and first_PK_stress(u_grad). 

        The class defines, solves, and stores a solution to a hyperelasticity problem, but does not return anything.
    Args:
        Problem (class): the base clase used to define finite element problems in JAX-FEM/CARDIAX (more details in problem.py).

    Returns:
        None 
    """
    def get_tensor_map(self):

        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            #Jinv = J**(-2. / 3.)
            Jinv = 1/(np.cbrt(J**2))
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(u_grad.shape[0])
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
```

The 1st Piola Kirchoff Stress tensor can be used to derive the forces required for deformation.

+++

The strain energy density function is defined as:

+++

$$
\psi = \left(\frac{\mu}{2}\right) \left(J^{-\frac{2}{3}} I_1 - 3\right) + \left(\frac{\kappa}{2}\right) \left(J - 1\right)^2
$$

+++

__Mesh Definition__
<br>
First, a FiniteElement or Bspline class is defined to discretize the domain. In this example, we can chose between linear hexahedral or linear bspline elements.
Results obtained using a linear bspline and linear hexahedral mesh should be the same (up to the tolerance of the solver).

```python
domain = 'linear bspline'
Lx, Ly, Lz = 1., 1., 1.
data_dir = os.path.join(os.path.dirname(__file__), 'data')

match domain:
    case 'linear hex':
        ele_type = 'hexahedron'
        mesh = box_mesh(Nx=10,
                       Ny=10,
                       Nz=10,
                       Lx=Lx,
                       Ly=Ly,
                       Lz=Lz,
                       data_dir=data_dir,
                       ele_type=ele_type)
        fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3)
    case 'linear bspline':
        deg = 1
        ele_type = 'SPLINEHEX'+str(deg)
        knot0 = np.hstack((np.zeros(deg), np.linspace(0,1,11) ,np.ones(deg)))
        knot1 = np.hstack((np.zeros(deg), np.linspace(0,1,11) ,np.ones(deg)))
        knot2 = np.hstack((np.zeros(deg), np.linspace(0,1,11) ,np.ones(deg)))
        knots = [knot0, knot1, knot2]
        degrees = 3*[deg]
        fe = BSpline(knots, degrees, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3*deg)
```


In the traditional finite element seeting, a mesh is passed to a {py:class}`FiniteElement <jax_fem.fe_abc>` object.
In a IGA setting, knots and degrees are passed to {py:class}`BSpline <jax_fem.iga>` object.
{py:class}`BSpline <jax_fem.iga>` and {py:class}`FiniteElement <jax_fem.fe_abc>` are intended to inherit from
the same abstract class. T-splines, G-Splines, and other mesh-like objects can hopefully be generated with this template.

__Boundary Conditions__
<br>
To finish defining the problem, the boundary conditions are defined. 
One of the faces of the cube is held in place (left face with x values as 0), while the opposite face (right face with x values equal to 1 or Lx) is twisted clockwise along the YZ plane. 
Two functions are used to capture/seperate all points on the left and right faces:

```python
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)
def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)
```


Then, the boundary displacements are defined. The Dirichlet boundary conditions are used.
For the left face, no displacement is imposed, so the boundary condition is 0:

```python
def zero_dirichlet_val(point):
    return 0.
```


For the right face, the y values are defined with: 

```python
def dirichlet_val_x2(point):
    return (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
            (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1]) / 2.
```


For the right face, the z values are defined similarly:

```python
def dirichlet_val_x3(point):
    return (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
            (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2]) / 2.
```


Now, the Dirichlet boundary conditions are combined:

```python
dirichlet_bc_info = [[left] * 3 + [right] * 3, [0, 1, 2] * 2,
                     [zero_dirichlet_val, dirichlet_val_x2, dirichlet_val_x3] +
                     [zero_dirichlet_val] * 3]
```


$\texttt{[left] * 3 + [right] * 3}$ specifies that the BC are applied to the left and right faces of the cube for all three spatial dimensions (x, y, z).
<br>
$\texttt{[0, 1, 2] * 2}$ indicates which components of the displacement vector are being constrained. Here, $\texttt{0, 1, 2}$ correspond to the x, y, and z components respectively. This pattern $\texttt{[0, 1, 2]}$ is repeated twice to match the left and right faces.
<br>
$\texttt{[zero_dirichlet_val, ... ,dirichlet_val_x3]+[zero_dirichlet_val]*3}$ specifies the boundary value functions to be applied. For the left face, $\texttt{zero_dirichlet_val}$ is applied to the x component, $\texttt{dirichlet_val_x2}$ to the y component, and $\texttt{dirichlet_val_x3}$ to the z component. 
For the right face, $\texttt{zero_dirichlet_val}$ is applied to all three components.

+++

__Problem Definition and Solution__
<br>
Now, finally, a hyperelasticity object is initialized with the FE object, FE bindings (optionally), and boundary condition-related information.

```python
problem = HyperElasticity(fe,
                          #fe_bindings = [[0,1]],
                          dirichlet_bc_info=dirichlet_bc_info)
```


The problem is then solved:

```python
sol = solver(problem, use_petsc=False)
```


Finally, the solution is saved as a .vtu file (visualized through ParaView) in the above specified data directory.
Note that for BSplines, visualization is only supported for linear elements. 

```python
problem_name = domain.split()
vtk_dir = os.path.join(data_dir, f'vtk/')
os.makedirs(vtk_dir, exist_ok=True)
vtk_path = os.path.join(vtk_dir, f'u_T_' +problem_name[0] + '_' + problem_name[1] + '.vtu')
save_sol(problem.fes[0], sol.reshape(-1, 3), vtk_path)
```
