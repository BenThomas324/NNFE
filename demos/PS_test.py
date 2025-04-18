
import jax.numpy as np
import numpy as onp
import jax
import matplotlib.pyplot as plt
import meshio
import time
jax.config.update("jax_enable_x64", True)

# Leftmost column for defining function space
from cardiax.fe import FiniteElement
from cardiax.iga import BSpline
# Center column for defining PDE
from cardiax.problem import Problem
# Rightmost column for defining solver
from cardiax.solver import Newton_Solver

from cardiax.generate_mesh import box_mesh

import pyvista as pv
pv.start_xvfb()
pv.OFF_SCREEN = True

# ML imports
import equinox as eqx
import optax

# We now create a problem class for NNFE
mesh = meshio.read('msh/LV_mesh.xdmf')
nnfe_space = FiniteElement(mesh, vec=3, dim=3, ele_type="tetra", gauss_order=1)

endo_points = mesh.points[mesh.point_data["endo"].astype(bool)]
epi_points = mesh.points[mesh.point_data["epi"].astype(bool)]
base_points = mesh.points[mesh.point_data["base"].astype(bool)]

fibers = mesh.cell_data["fibers"][0]
sheets  = mesh.cell_data["sheets"][0]
normals  = mesh.cell_data["normals"][0]

def base(point):
    return np.isclose(point, base_points, atol=1e-6).all(axis=1).any()

def endo(point):
    return np.isclose(point, endo_points, atol=1e-6).all(axis=1).any()

def epi(point):
    return np.isclose(point, epi_points, atol=1e-6).all(axis=1).any()

def zero_val(point):
    return 0.

# The boundary conditions are defined as a list of lists
# Each list contains the following:
# [on_boundary (bool value), dof_index, value]
bc_base = [[base] * 3, [0, 1, 2], [zero_val]*3]

dirichlet_bc_info = [[bc_base]]
location_fns = [[endo]]

class HyperElasticity(Problem):

    def get_tensor_map(self):
        def psi(F, f, s, n, TCa):
            f = f[:, None]
            s = s[:, None]
            n = n[:, None]
            J = np.linalg.det(F)
            C = F.T @ F * np.cbrt(J)**(-2)
            E_tilde = 1/2 * (C - np.eye(self.dim[0]))
            E11 = f.T @ E_tilde @ f
            E12 = f.T @ E_tilde @ s
            E13 = f.T @ E_tilde @ n
            E22 = s.T @ E_tilde @ s
            E23 = s.T @ E_tilde @ n
            E33 = n.T @ E_tilde @ n

            Q = 12. * E11**2 \
            + 8. * (E22**2 + E33**2 + 2*E23**2) \
            + 26. * (E12**2 + E13**2)
            
            psi_dev = (1522.083/100**2)/2 * (np.exp(2.125 * Q[0, 0]) - 1)
            K = 1e7/(100**2) - 0. * 1000/(100**2) * TCa
            psi_vol = K/2 * ( (J**2 - 1)/2 - np.log(J))
            return psi_dev + psi_vol[0]
        
        P_fn = jax.grad(psi)

        def S_act(F, f, TCa):
            f = f[:, None]
            lamb = np.sqrt(f.T @ F.T @ F @ f)
            S = 1000/(100**2) * TCa * (1 + 1.4 * (lamb - 1))/(lamb ** 2) * f @ f.T
            return S

        def first_PK_stress(u_grad, f, s, n, TCa):
            F = u_grad + np.eye(self.dim[0])
            P_psi = P_fn(F, f, s, n, TCa)
            P_act = F @ S_act(F, f, TCa)
            return P_psi + P_act

        return first_PK_stress
    
    def get_surface_maps(self):

        def pressure(u, u_grad, x, pL, nL):
            F = u_grad + np.eye(self.dim[0])
            J = np.linalg.det(F)
            F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
                                    [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
                                    [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])
            val = 133.322/(100**2) * pL * J * F_inv.T @ nL.reshape(3, 1)
            return val.reshape(3)

        return [pressure]
    
nnfe_problem = HyperElasticity(nnfe_space,
                          dirichlet_bc_info=dirichlet_bc_info,
                          location_fns=location_fns)
# Now we need to define internal_vars_surfaces for the traction values
# To obtain the correct shape, we just reshape the normals of the surface
endo_normals = nnfe_problem.fes[0].get_surface_normals(endo)
nnfe_problem.internal_vars_surfaces = [[[1. * np.ones_like(endo_normals)[:, :, 0], endo_normals]]]

fibers_dir = onp.repeat(fibers[:, None, :], nnfe_problem.fes[0].num_quads, axis=1)
sheets_dir = onp.repeat(sheets[:, None, :], nnfe_problem.fes[0].num_quads, axis=1)
normals_dir = onp.repeat(normals[:, None, :], nnfe_problem.fes[0].num_quads, axis=1)
TCa_vec = 1. * onp.ones_like(fibers_dir)[:, :, :1]

nnfe_problem.internal_vars = [fibers_dir, sheets_dir, normals_dir, TCa_vec]
nnfe_solver = Newton_Solver(nnfe_problem, np.zeros((nnfe_problem.num_total_dofs_all_vars)))

# FE solution to check things
# TCa_vec = 0. * onp.ones_like(fibers_dir)[:, :, :1]
# nnfe_problem.internal_vars = [fibers_dir, sheets_dir, normals_dir, TCa_vec]
# nnfe_problem.internal_vars_surfaces = [[[120. * np.ones_like(endo_normals)[:, :, 0], endo_normals]]]

# sol, info = nnfe_solver.solve(1e-6, 40)

# pvmesh = pv.from_meshio(mesh)
# pl = pv.Plotter()

# pvmesh["sol"] = sol.reshape(-1, 3)
# warped = pvmesh.warp_by_vector("sol")
# warped.set_active_scalars(None)
# pvmesh.set_active_scalars(None)

# pl.add_mesh(pvmesh, show_edges=True, copy_mesh=True, opacity=.75)
# pl.show_axes()
# pl.add_points(warped.points, render_points_as_spheres=True, 
#               point_size=10, color='r')
# pl.show_axes()
# pl.screenshot('PS_test.png')

# exit()
# End FE solve

# Now we create the neural network and optimizer
model = eqx.nn.MLP(in_size=2, out_size=nnfe_problem.num_total_dofs_all_vars, 
                    width_size=1024, depth=6, activation=jax.nn.relu, key=jax.random.PRNGKey(0),
                    use_final_bias=False)

epochs = int(1.e4)
scheduler = optax.exponential_decay(1e-4, transition_steps=epochs, decay_rate=1e-2)
optimizer = optax.adam(scheduler)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

def trunc_weight(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    out, in_ = weight.shape
    stddev = 1e-6
    return stddev * jax.random.truncated_normal(key, shape=(out, in_), lower=-1, upper=1)

def trunc_bias(bias: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    out = bias.shape
    stddev = 1e-6
    return stddev * jax.random.truncated_normal(key, shape=(out), lower=-1, upper=1)

def init_linear_weight(model, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x)]
    try:
        get_biases = lambda m: [x.bias
                                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                                if is_linear(x)]
        biases = get_biases(model)
        new_biases = [trunc_bias(bias, subkey)
                    for bias, subkey in zip(biases, jax.random.split(key, len(biases)))]
        model = eqx.tree_at(get_biases, model, new_biases)
    except:
        pass

    weights = get_weights(model)
    new_weights = [trunc_weight(weight, subkey)
                    for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    model = eqx.tree_at(get_weights, model, new_weights)
    return model

key = jax.random.PRNGKey(0)
model = init_linear_weight(model, key)

# BCs are fixed, so we need to 0 res_vec
dirichlet_dofs, dirichlet_vals = nnfe_problem.get_boundary_data()

def calc_res(model, x):
    dofs = model(x)
    int_vars_surfaces = [[[x[0] * np.ones_like(endo_normals)[:, :, 0], endo_normals]]]
    int_vars = [fibers_dir, sheets_dir, normals_dir, x[1] * onp.ones_like(fibers_dir)[:, :, :1]]
    res_vec = nnfe_problem.compute_residual_helper(dofs, int_vars, int_vars_surfaces)
    res_vec = res_vec.at[dirichlet_dofs].set(dofs[dirichlet_dofs])
    res_vec = res_vec.at[dirichlet_dofs].add(-dirichlet_vals, unique_indices=True)
    return res_vec

vcalc_res = jax.vmap(calc_res, in_axes=(None, 0), out_axes=0)

# Define MSE
def loss_fct(model, x):
    vres = vcalc_res(model, x)
    return np.mean(np.linalg.norm(vres, axis=1))

val_and_grads = eqx.filter_value_and_grad(loss_fct)

@eqx.filter_jit
def make_step(model, x, opt_state):
    loss_val, grads = val_and_grads(model, x)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

pressure_vals = onp.linspace(0, 120., 5).reshape(-1, 1)
TCa_vals = onp.linspace(0, 60., 5).reshape(-1, 1)

X = onp.meshgrid(pressure_vals, TCa_vals)
X = np.vstack([v.flatten() for v in X]).T

loss_vals = []
for i in range(epochs):
    batch = jax.random.permutation(jax.random.PRNGKey(i), len(X))[:5]
    model, opt_state, train_loss = make_step(model, X[batch], opt_state)
    loss_vals.append(train_loss)
    if i % 10 == 0:
        print("Iteration: ", i, " Loss: ", train_loss)

nnfe_sols = []
fe_sols = []
fails = []
for x in X:        # Apply dirichlet BCs for prediction and compare errors
    predicted_dofs = model(x)
    predicted_dofs = predicted_dofs.at[dirichlet_dofs].set(dirichlet_vals)
    nnfe_sols.append(predicted_dofs.reshape(-1, 3))
    int_vars_surfaces = [[[x[0] * np.ones_like(endo_normals)[:, :, 0], endo_normals]]]
    int_vars = [fibers_dir, sheets_dir, normals_dir, x[1] * onp.ones_like(fibers_dir)[:, :, :1]]
    nnfe_problem.internal_vars_surfaces = int_vars_surfaces
    nnfe_sol, info = nnfe_solver.solve(1e-6, 40)
    if not info[0]:
        nnfe_sols.remove(-1)
        fails.append(x)
        continue
    fe_sols.append(nnfe_sol.reshape(-1, 3))

for i in range(len(X)):
    diff = np.linalg.norm(fe_sols[i] - nnfe_sols[i], axis=1)
    print(f"Values: {X[i]}")
    print(f"Average solution difference: {diff.mean()}")
    print(f"Max solution difference: {diff.max()}")

print("Failed sims: ", fails)

toc = time.time()
for _ in range(100):
    jax.vmap(model)(X)
print("Average execution time: ", (time.time() - toc)/100)

exit()
# Plotting
pv_nnfe_mesh = pv.from_meshio(nnfe_mesh)
pv_nnfe_mesh["sol"] = predicted_dofs.reshape(-1, 3)

nnfe_warped = pv_nnfe_mesh.warp_by_vector()
nnfe_warped.set_active_scalars(None)

pl = pv.Plotter()
pl.add_mesh(pv_nnfe_mesh, show_edges=True)

pl.open_gif('PS_test.gif')
for i in range(25):
    t = 2 * i / 25
    predicted_dofs = model(np.array([t]))
    predicted_dofs = predicted_dofs.at[dirichlet_dofs].set(dirichlet_vals)

    pv_nnfe_mesh["sol"] = predicted_dofs.reshape(-1, 3)
    pv_nnfe_mesh.set_active_vectors("sol")
    warped = pv_nnfe_mesh.warp_by_vector()
    pl.add_mesh(warped, show_edges=True)
    pl.write_frame()
    pl.clear()
    print("Iteration: ", i)

pl.close()
