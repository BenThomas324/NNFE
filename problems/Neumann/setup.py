
import jax.numpy as np
import jax

from jax_fem.problem_abc import Problem
from jax_fem.generate_mesh import box_mesh
from jax_fem.fe_abc import FiniteElement
from jax_fem.solver_abc import apply_bc_vec

class HyperElasticity(Problem):
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

    def get_pressure_maps(self):

        def pressure_map(u_grads, normals, p):
            p = 1
            F = u_grads + np.eye(self.dim[0])
            J = np.linalg.det(F)
            F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
                                    [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
                                    [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])
            return -p * J * F_inv.T @ normals
                
        return [pressure_map]

def fe_setup():
    # Specify mesh-related information (first-order hexahedron element).
    ele_type = 'hexahedron'
    Lx, Ly, Lz = 1., 1., 1.
    mesh = box_mesh(Nx=10,
                        Ny=10,
                        Nz=10,
                        Lx=Lx,
                        Ly=Ly,
                        Lz=Lz,
                        data_dir="mesh",
                        ele_type=ele_type)

    # Define boundary locations.
    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    # Define Dirichlet boundary values.
    def zero_bc(point):
        return 0.

    dirichlet_bc_info = [[left] * 3, [0, 1, 2],
                        [zero_bc] * 3]

    fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3)
    problem = HyperElasticity(fe, dirichlet_bc_info=dirichlet_bc_info, location_fns=[right])

    right_normals = problem.fes[0].get_surface_normals(right)
    pressures = np.zeros_like(right_normals)[:, :, :1]
    # normals = np.full((right_normals.shape[0], problem.fes[0].num_face_quads, 3), right_normals[:, None, :])
    return problem, (), [[[right_normals, pressures]]]

def get_data():
    p = np.linspace(0, 2.5, 10)
    return p.reshape(10, 1), [[]], [[[1]]]

def get_res(problem, compute_residual_vars, 
            internal_vars, internal_vars_surface):

    def calc_res(dofs, vars): #internal_vars, internal_vars_surface):
        pressures = vars[0] * np.ones_like(internal_vars_surface[0][0][0])[:, :, :1]
        internal_vars_surfaces = [[[internal_vars_surface[0][0][0], pressures]]]
        res_list = compute_residual_vars(dofs, internal_vars, internal_vars_surfaces)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        return res_vec
    return calc_res


