
import numpy as onp
import jax.numpy as np
import jax
import sys
from .env_var import path
sys.path.append(path)
from jax_fem.problem import Problem

class VHL_Fung(Problem):

    def get_tensor_map(self):
        def psi(F, f, s, n):
            f = f[:, None]
            s = s[:, None]
            n = n[:, None]
            J = np.linalg.det(F)
            C = F.T @ F * J**(-2/3)
            E_tilde = 1/2 * (C - np.eye(self.dim))
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
            F = u_grad + np.eye(self.dim)
            P_psi = P_fn(F, f, s, n)
            P_act = F @ S_act(F, T_Ca, f)
            return P_psi + P_act

        return first_PK_stress

    def get_pressure_map(self):

        def F_extLV(u_grad, nL, pL):
            F = u_grad + np.eye(self.dim)
            J = np.linalg.det(F)
            # F_inv = np.linalg.inv(F)
            F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
                                    [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
                                    [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])
            val = pL * J * F_inv.T @ nL.reshape(3, 1)
            return val.reshape(3)

        def F_extRV(u_grad, nR, pR):
            F = u_grad + np.eye(self.dim)
            J = np.linalg.det(F)
            F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
                                    [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
                                    [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])
            val = pR * J * F_inv.T @ nR.reshape(3, 1)
            return val.reshape(3)

        return [F_extLV, F_extRV]

    def set_params(self, constants, pressures, T_Ca, normals, fibers):

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
        
        p_LV = np.full((len(normals[0]), 1), pressures[0])
        p_RV = np.full((len(normals[1]), 1), pressures[1])
        self.T_Ca = T_Ca

        self.internal_vars = fibers
        self.internal_vars_surfaces = [[normals[0], p_LV], [normals[1], p_RV]]
        
        return

class LV_Fung(Problem):

    def get_tensor_map(self):
        def psi(F, f, s, n):
            f = f[:, None]
            s = s[:, None]
            n = n[:, None]
            J = np.linalg.det(F)
            C = F.T @ F * J**(-2/3)
            E_tilde = 1/2 * (C - np.eye(self.dim))
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
            F = u_grad + np.eye(self.dim)
            P_psi = P_fn(F, f, s, n)
            P_act = F @ S_act(F, T_Ca, f)
            return P_psi + P_act

        return first_PK_stress

    def get_pressure_map(self):

        def F_extLV(u_grad, nL, pL):
            F = u_grad + np.eye(self.dim)
            J = np.linalg.det(F)
            # F_inv = np.linalg.inv(F)
            F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
                                    [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
                                    [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])
            val = pL * J * F_inv.T @ nL.reshape(3, 1)
            return val.reshape(-1)

        return [F_extLV]

    def set_params(self, constants, pressures, TCa, normals, fibers):

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
        
        p_LV = pressures * np.ones_like(normals)[:, :, :, :1]
        TCa = TCa * np.ones_like(fibers[0])[:, :, :1]

        self.internal_vars = [TCa, *fibers]
        self.internal_vars_surfaces = [[normals, p_LV]]
        
        return

class LV_NH(Problem):

    def get_tensor_map(self):
        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy
        
        P_fn = jax.grad(psi)

        def S_act(F, T_Ca, f):
            f = f[:, None]
            lamb = np.sqrt(f.T @ F.T @ F @ f)
            S = T_Ca * (1 + self.beta * (lamb - 1))/(lamb ** 2) * f @ f.T
            return S

        def first_PK_stress(u_grad, T_Ca, f, s, n):
            F = u_grad + np.eye(self.dim)
            P_psi = P_fn(F)#, f, s, n)
            P_act = F @ S_act(F, T_Ca, f)
            return P_psi + P_act

        return first_PK_stress

    def get_pressure_map(self):

        def F_extLV(u_grad, nL, pL):
            F = u_grad + np.eye(self.dim)
            J = np.linalg.det(F)
            # F_inv = np.linalg.inv(F)
            F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
                                    [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
                                    [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])
            val = pL * J * F_inv.T @ nL.reshape(3, 1)
            return val.reshape(-1)

        return [F_extLV]

    def set_params(self, pressure, TCa, normals, fibers):
        
        p_LV = pressure * np.ones_like(normals)[:, :, :, :1]
        TCa = TCa * np.ones_like(fibers[0])[:, :, :1]

        self.beta = 1.4
        self.internal_vars = [TCa, *fibers]
        self.internal_vars_surfaces = [[normals, p_LV]]
        
        return

class NH_cube(Problem):

    def get_tensor_map(self):

        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
class Pressure_cube(Problem):

    def get_tensor_map(self):

        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
    def get_pressure_map(self):

        def ext_P(u_grad, nL, pL):
            F = u_grad + np.eye(self.dim)
            J = np.linalg.det(F)
            # F_inv = np.linalg.inv(F)
            F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
                                    [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
                                    [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])
            val = pL * J * F_inv.T @ nL.reshape(-1, 1)
            return val.reshape(-1)

        return [ext_P]

    def set_params(self, normals, pressures):

        self.internal_vars_surfaces = [[normals, pressures]]

        return

def get_normals(fe_prob, boundary_inds):

    physical_coos = onp.take(fe_prob.points, fe_prob.cells, axis=0)
    selected_coos = physical_coos[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, dim)
    selected_face_shape_vals = fe_prob.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
    # (num_selected_faces, num_face_quads, num_nodes, 1) * (num_selected_faces, 1, num_nodes, dim) -> (num_selected_faces, num_face_quads, dim)
    physical_surface_quad_points = onp.sum(selected_face_shape_vals[:, :, :, None] * selected_coos[:, None, :, :], axis=2)

    selected_f_shape_grads_ref = fe_prob.face_shape_grads_ref[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes, dim)
    selected_f_normals = fe_prob.face_normals[boundary_inds[:, 1]]  # (num_selected_faces, dim)
    jacobian_dx_deta = onp.sum(selected_coos[:, None, :, :, None] * selected_f_shape_grads_ref[:, :, :, None, :], axis=2)
    jacobian_det = onp.linalg.det(jacobian_dx_deta)  # (num_selected_faces, num_face_quads)
    jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)  # (num_selected_faces, num_face_quads, dim, dim)

    nanson_scale = selected_f_normals[:, None, None, :] @ jacobian_deta_dx

    normals = nanson_scale / np.linalg.norm(nanson_scale, axis=-1)[:, :, :, None]
    return normals


def get_faces_and_normals(fe_prob, boundary_inds, boundary_pts):
    
    physical_coos = onp.take(fe_prob.points, fe_prob.cells, axis=0)
    selected_coos = physical_coos[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, dim)
    selected_face_shape_vals = fe_prob.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)

    faces = np.zeros((len(selected_coos), 4), dtype=bool)
    for b in boundary_pts:
        faces += (b == selected_coos).all(axis=-1)
    isface = faces.sum(axis=1) == 3

    for i, f in enumerate(faces):
        if f.sum() == 3:
            continue
        else:
            faces = faces.at[i].set(np.zeros_like(f))

    faces = selected_coos[faces].reshape(-1, 3, 3)

    selected_f_shape_grads_ref = fe_prob.face_shape_grads_ref[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes, dim)
    selected_f_normals = fe_prob.face_normals[boundary_inds[:, 1]]  # (num_selected_faces, dim)
    jacobian_dx_deta = onp.sum(selected_coos[:, None, :, :, None] * selected_f_shape_grads_ref[:, :, :, None, :], axis=2)
    jacobian_det = onp.linalg.det(jacobian_dx_deta)  # (num_selected_faces, num_face_quads)
    jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)  # (num_selected_faces, num_face_quads, dim, dim)

    nanson_scale = selected_f_normals[:, None, None, :] @ jacobian_deta_dx

    normals = nanson_scale.reshape(-1, 3) / np.linalg.norm(nanson_scale.reshape(-1, 3), axis=1)[:, None]


    return faces, normals[isface]







