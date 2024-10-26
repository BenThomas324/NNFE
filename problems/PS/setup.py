
import meshio
import jax.numpy as np
import numpy as onp
import jax

from jax_fem.problem_abc import Problem
from jax_fem.fe_abc import FiniteElement
from jax_fem.solver_abc import apply_bc_vec
from nnfe.FE_base import FE_Base

class LV_Fung(Problem):

    def get_tensor_map(self):
        def psi(F, f, s, n):
            f = f[:, None]
            s = s[:, None]
            n = n[:, None]
            J = np.linalg.det(F)
            C = F.T @ F * J**(-2/3)
            E_tilde = 1/2 * (C - np.eye(self.dim[0]))
            E11 = f.T @ E_tilde @ f
            E12 = f.T @ E_tilde @ s
            E13 = f.T @ E_tilde @ n
            E22 = s.T @ E_tilde @ s
            E23 = s.T @ E_tilde @ n
            E33 = n.T @ E_tilde @ n

            Q = self.A1 * E11**2 \
            + self.A2 * (E22**2 + E33**2 + E23**2) \
            + self.A3 * (E12**2 + E13**2)
            
            psi_dev = self.c/2 * (np.exp(self.alpha * Q[0, 0]) - 1)
            psi_vol = self.K/2 * ( (J**2 - 1)/2 - np.log(J))

            return psi_dev + psi_vol
        
        P_fn = jax.grad(psi)

        def S_act(F, T_Ca, f):
            f = f[:, None]
            lamb = np.sqrt(f.T @ F.T @ F @ f)
            S = T_Ca * (1 + self.beta * (lamb - 1))/(lamb ** 2) * f @ f.T
            return S

        def first_PK_stress(u_grad, T_Ca, f, s, n):
            F = u_grad + np.eye(self.dim[0])
            P_psi = P_fn(F, f, s, n)
            P_act = F @ S_act(F, T_Ca, f)
            return P_psi + P_act

        return first_PK_stress

    def get_pressure_maps(self):

        def F_extLV(u_grad, nL, pL):
            F = u_grad + np.eye(self.dim[0])
            J = np.linalg.det(F)
            # F_inv = np.linalg.inv(F)
            F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
                                    [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
                                    [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])
            val = pL * J * F_inv.T @ nL.reshape(3, 1)
            return val.reshape(-1)

        return [F_extLV]

    def set_params(self, constants):

        cm = 1
        cm2mm=10
        gram = 1e-3
        second = 1
        dynecm2 = gram*cm/second**2

        self.A1 = constants["A1"]
        self.A2 = constants["A2"]
        self.A3 = constants["A3"]
        self.c = dynecm2 * constants["c"]
        self.K = dynecm2 * constants["K"]
        self.alpha = constants["alpha"]
        self.beta = constants["beta"]
        return

# class LV_NH(Problem):

#     def get_tensor_map(self):
#         def psi(F):
#             E = 10.
#             nu = 0.4
#             mu = E / (2. * (1. + nu))
#             kappa = E / (3. * (1. - 2. * nu))
#             J = np.linalg.det(F)
#             Jinv = J**(-2. / 3.)
#             I1 = np.trace(F.T @ F)
#             energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
#             return energy
        
#         P_fn = jax.grad(psi)

#         def S_act(F, T_Ca, f):
#             f = f[:, None]
#             lamb = np.sqrt(f.T @ F.T @ F @ f)
#             S = T_Ca * (1 + self.beta * (lamb - 1))/(lamb ** 2) * f @ f.T
#             return S

#         def first_PK_stress(u_grad, T_Ca, f, s, n):
#             F = u_grad + np.eye(self.dim[0])
#             P_psi = P_fn(F)#, f, s, n)
#             P_act = F @ S_act(F, T_Ca, f)
#             return P_psi + P_act

#         return first_PK_stress

#     def get_pressure_maps(self):

#         def F_extLV(u_grad, nL, pL):
#             F = u_grad + np.eye(self.dim[0])
#             J = np.linalg.det(F)
#             # F_inv = np.linalg.inv(F)
#             F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
#                                     [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
#                                     [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])
#             val = pL * J * F_inv.T @ nL.reshape(3, 1)
#             return val.reshape(-1)

#         return [F_extLV]

#     def set_params(self, constants, pressures, TCa, normals, fibers):
#         self.beta = 1.4

#         p_LV = pressures * np.ones_like(normals)
#         TCa = TCa * np.ones_like((fibers[0]))[:, :, :1]

#         self.internal_vars = [TCa, *fibers]
#         self.internal_vars_surfaces = [[[normals, p_LV]]]
        
#         return

class FE_data(FE_Base):

    def fe_setup(self):

        params = {}
        params["A1"] = 12.
        params["A2"] = 8.
        params["A3"] = 26.
        params["c"] = 15220.83
        params["K"] = 1e6
        params["alpha"] = 2.125
        params["beta"] = 1.4

        print("Reading meshes")
        mesh = meshio.read("/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/LV/mesh/LV_complete.vtu")
        ele_type = "tetra"

        base_points = mesh.points[mesh.point_data["BASE"].astype(bool)[:, 0]]
        LV_points = mesh.points[mesh.point_data["ENDO"].astype(bool)[:, 0]]

        def base(point):
            return np.isclose(point, base_points, atol=1e-6).all(axis=1).any()

        def LVendo(point):
            return np.isclose(point, LV_points, atol=1e-6).all(axis=1).any()

        # Fixed top nodes
        def zero_dirichlet(point):
            return 0.0

        # dirichlet_bc_info = [Where, what directions, value]
        dirichlet_bc_info = [[base]*3, [0, 1, 2], [zero_dirichlet]*3]
        location_fns = [LVendo]

        fibers = mesh.point_data["fibers"]
        sheets = mesh.point_data["sheets"]
        normals = mesh.point_data["normals"]

        print("initalize problem")
        fe = FiniteElement(mesh, vec=3, dim=3, ele_type=ele_type, gauss_order=1)
        problem = LV_Fung(fe, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)

        LV_normals = problem.fes[0].get_surface_normals(LVendo)

        fibers = problem.fes[0].convert_dof_to_quad(fibers)
        sheets = problem.fes[0].convert_dof_to_quad(sheets)
        normals = problem.fes[0].convert_dof_to_quad(normals)

        problem.set_params(params)

        TCa = 1. * onp.ones_like((fibers))[:, :, :1]
        p_LV = 5. * onp.ones_like(LV_normals)[:, :, :1]

        internal_vars = [TCa, fibers, sheets, normals]
        internal_vars_surfaces = [[[LV_normals, p_LV]]]

        return problem, internal_vars, internal_vars_surfaces

    def get_res(self, problem, int_vars, int_vars_surface):

        def compute_residual_vars(global_dofs, internal_vars, internal_vars_surfaces):
            cells_dof_list = [fe.local_to_cell_dofs(problem.global_to_local_dofs(global_dofs, fe_index)) for fe_index, fe in enumerate(problem.fes)]
            #each entry in cells_sol_list has shape (num_cells,num_bases_per_cell,vec)
            weak_form_flat = problem.split_and_compute_cell(cells_dof_list, np, False, internal_vars)
            weak_form_face_flat = problem.compute_face(cells_dof_list, np, False, internal_vars_surfaces)  # [(num_selected_faces, num_nodes*vec + ...), ...]

            return problem.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)

        # @jax.vmap
        def calc_res(dofs, vars):
            # vars[0] is TCa
            # vars[1] is pressure
            TCa = vars[0] * onp.ones_like((int_vars[1]))[:, :, :1]
            pressures = vars[1] * onp.ones_like(int_vars_surface[0][0][0])[:, :, :1]
            internal_vars = [TCa, int_vars[1], int_vars[2], int_vars[3]]
            internal_vars_surfaces = [[[int_vars_surface[0][0][0], pressures]]]
            res_vec = compute_residual_vars(dofs, internal_vars, internal_vars_surfaces)
            # res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
            res_vec = apply_bc_vec(res_vec, dofs, problem)
            return res_vec
        return calc_res

    def get_training_data(self):
        n = 5
        p = onp.linspace(0, 60., n)
        TCa = onp.linspace(0, 40., n)
        X = onp.meshgrid(TCa, p)
        X = onp.vstack([onp.ravel(a) for a in X]).T
        return X

    def get_testing_data(self):
        n = 5
        p = onp.linspace(0, 60., n)
        TCa = onp.linspace(0, 40., n)
        X = onp.meshgrid(TCa, p)
        X = onp.vstack([onp.ravel(a) for a in X]).T
        return X
