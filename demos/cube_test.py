
import jax.numpy as np
import jax
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

# Leftmost column for defining function space
from cardiax.fe import FiniteElement
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
L=1.
# Can try changing these
# ele_types = ["hexahedron", "tetra"], gauss_order = 1
# ele_types = ["hexahedron27", "tetra10"], gauss_order = 2

nnfe_mesh = box_mesh(Nx=8, Ny=8, Nz=8, 
                   Lx=L, Ly=L, Lz=L,
                   data_dir="", ele_type="tetra10")

nnfe_space = FiniteElement(nnfe_mesh, vec=3, dim=3, ele_type="tetra10", gauss_order=2)

def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)

def top(point):
    return np.isclose(point[2], L, atol=1e-5)

def zero_val(point):
    return 0.

# The boundary conditions are defined as a list of lists
# Each list contains the following:
# [on_boundary (bool value), dof_index, value]
bc_bottom = [[bottom] * 3, [0, 1, 2], [zero_val]*3]

dirichlet_bc_info = [[bc_bottom]]
location_fns = [[top]]

class HyperElasticity(Problem):

    def get_tensor_map(self):

        def psi(F):
            E = 10. # Stiffness
            nu = 0.45 # Change this guy between 0.3 and 0.49
            # Closer to .49 -> incompressible
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = 1/(np.cbrt(J**2))
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        # standard input u_grad
        def first_PK_stress(u_grad):
            I = np.eye(u_grad.shape[0])
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
    def get_surface_maps(self):

        # standard input u, u_grad, x
        def surface_map(u, u_grad, x, t):
            return np.array([0., 0., t])
                
        return [surface_map]
    
nnfe_problem = HyperElasticity(nnfe_space,
                          dirichlet_bc_info=dirichlet_bc_info,
                          location_fns=location_fns)
# Now we need to define internal_vars_surfaces for the traction values
# To obtain the correct shape, we just reshape the normals of the surface
top_normals = nnfe_problem.fes[0].get_surface_normals(top)
nnfe_problem.internal_vars_surfaces = [[[1. * np.ones_like(top_normals)[:, :, 0]]]]
nnfe_solver = Newton_Solver(nnfe_problem, np.zeros((nnfe_problem.num_total_dofs_all_vars)))

# Now we create the neural network and optimizer
# Change whatever you want here
model = eqx.nn.MLP(in_size=1, out_size=nnfe_problem.num_total_dofs_all_vars, 
                    width_size=1024, depth=6, activation=jax.nn.elu, key=jax.random.PRNGKey(0),
                    use_final_bias=False)

epochs = int(1e3)
scheduler = optax.exponential_decay(1e-4, transition_steps=epochs, decay_rate=1e-1)
optimizer = optax.adam(scheduler)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

def trunc_weight(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    out, in_ = weight.shape
    stddev = 1e-5
    return stddev * jax.random.truncated_normal(key, shape=(out, in_), lower=-1, upper=1)

def trunc_bias(bias: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    out = bias.shape
    stddev = 1e-5
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
    int_vars_surfaces = [[[x * np.ones_like(nnfe_problem.internal_vars_surfaces[0][0][0])]]]
    res_vec = nnfe_problem.compute_residual_helper(dofs, [], int_vars_surfaces)
    # Bottom goes to 0
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

# Change sampling of traction
tract_vals = np.linspace(0, 3., 10).reshape(-1, 1)

loss_vals = []
for i in range(epochs):
    # batch = jax.random.permutation(jax.random.PRNGKey(i), len(tract_vals))[:5]
    model, opt_state, train_loss = make_step(model, tract_vals, opt_state)
    loss_vals.append(train_loss)
    if i % 10 == 0:
        print("Iteration: ", i, " Loss: ", train_loss)

nnfe_sols = []
fe_sols = []
for t in tract_vals:
    int_vars_surfaces = [[[t * np.ones_like(nnfe_problem.internal_vars_surfaces[0][0][0])]]]
    nnfe_problem.internal_vars_surfaces = int_vars_surfaces
    nnfe_sol, info = nnfe_solver.solve(1e-6, 10)
    # 1e-6 is roughly tolerance of true displacement
    fe_sols.append(nnfe_sol)

    # Apply dirichlet BCs for prediction and compare errors
    predicted_dofs = model(np.array(t))
    predicted_dofs = predicted_dofs.at[dirichlet_dofs].set(dirichlet_vals)
    nnfe_sols.append(predicted_dofs)

for i in range(len(tract_vals)):
    print(f"Tract value: {tract_vals[i]}")
    diff = np.linalg.norm(fe_sols[i] - nnfe_sols[i], axis=1)
    print(f"Solution mean difference: {diff.mean()}")
    print(f"Solution max difference: {diff.max()}")

exit()

# Plotting
pv_nnfe_mesh = pv.from_meshio(nnfe_mesh)
pv_nnfe_mesh["sol"] = predicted_dofs.reshape(-1, 3)

nnfe_warped = pv_nnfe_mesh.warp_by_vector()
nnfe_warped.set_active_scalars(None)

pl = pv.Plotter()
pl.add_mesh(pv_nnfe_mesh, show_edges=True)

pl.open_gif('test.gif')
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
