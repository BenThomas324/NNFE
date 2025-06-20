
import jax.numpy as np
import numpy as onp
import jax

from cardiax.problem import Problem
from cardiax.generate_mesh import box_mesh
from cardiax.fe import FiniteElement

from nnfe.utils import yaml_load

class HyperElasticity(Problem):
    def get_tensor_map(self):

        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
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

    def get_pressure_maps(self):

        def pressure_map(u_grads, normals, p):
            F = u_grads + np.eye(self.dim[0])
            J = np.linalg.det(F)
            F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
                                    [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
                                    [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])
            return -p * J * F_inv.T @ normals
                
        return [pressure_map]


ele_type = 'hexahedron'
Lx, Ly, Lz = 1., 1., 1.
mesh = box_mesh(Nx=10,
                Ny=10,
                Nz=10,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                data_dir="",
                ele_type=ele_type)

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1., atol=1e-5)

# Define Dirichlet boundary values.
def zero_bc(point):
    return 0.

dirichlet_bc_info = [[[left] * 3, [0, 1, 2],
                    [zero_bc] * 3]]

fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3)
problem = HyperElasticity(fe, dirichlet_bc_info=dirichlet_bc_info, location_fns = [[right]])

right_normals = problem.fes[0].get_surface_normals(right)
pressures = np.ones_like(right_normals)[:, :, :1]

yaml_file = "test_params.yaml"

params = yaml_load(yaml_file)

