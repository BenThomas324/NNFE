### Do imports of modules used ###
import numpy as onp
import jax.numpy as np
import jax
import sys
import time
import yaml
import equinox as eqx
import os

problem = sys.argv[1]
print(problem)
exit()

parent = os.path.dirname(__file__)
sys.path.append(os.path.dirname(parent) + "/src")

from FE_helpers import *
from NN_helpers import *
from problem_setup import *
from utils import *

jax.config.update("jax_enable_x64", True)
XLA_PYTHON_CLIENT_PREALLOCATE=False

#param_file = "/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/hp_template.yaml"
param_file = parent + "/params.yaml"
with open(param_file, 'r') as f:
    params = yaml.safe_load(f)

opt_params = params["Optimizer"]
NN_params = params["Network"]
FE_params = params["FE"]
data_params = params["Data"]

results_dir, key = setup_dirs(params, parent)

# X = get_training_points(data_params)
# problem, fiber_dirs, normals = setup_problem(FE_params)
problem, bc_info = Dirichlet_test()

if NN_params["kwargs"]["out_size"] == "dofs":
    NN_params["kwargs"]["out_size"] = problem.fes[0].num_total_dofs

model = create_network(NN_params, key)
optimizer = create_optimizer(opt_params, results_dir)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
epochs = int(opt_params["epochs"])

X = np.linspace(0, 1/2, 3)
###
# Have to create the bcs for the residual vec before hand
vals_list = []
for x in X:
    bcs = bc_info(x)
    _, _, vals = problem.fes[0].Dirichlet_boundary_conditions(bcs)
    vals_list.append(np.vstack(tuple(vals))[None, :, :])
vals_list = np.vstack(tuple(vals_list))

# Initialize one of the bcs to get the fe.node_inds_list
# and fe.vec_inds_list used in apply_bc_vec

problem.fes[0].update_Dirichlet_boundary_conditions(bcs)

# If learning over boundary conditions, use custom apply_bc_vec
def apply_bc_vec(res_vec, vals, dofs, problem):
    res = problem.unflatten_fn_sol_list(res_vec)[0]
    sol = problem.unflatten_fn_sol_list(dofs)[0]
    fe = problem.fes[0]

    for i in range(len(fe.node_inds_list)):
        res = (res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
            sol[fe.node_inds_list[i], fe.vec_inds_list[i]], unique_indices=True))
        res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].add(-vals[i])

    return jax.flatten_util.ravel_pytree(res)[0]

# Calculate residual
@jax.vmap
def calc_res(dofs, vals):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    #res_list = problem.compute_residual_vars(sol_list, internal_vars, internal_vars_surfaces)
    # Use above with internal vars
    # Use below if just solution needed in cells
    res_list = problem.compute_residual(sol_list)
    res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
    res_vec = apply_bc_vec(res_vec, vals, dofs, problem)
    return res_vec #np.linalg.norm(res_vec)

# Create error funct and get value_and_grad
@eqx.filter_value_and_grad
def error(model, X, vals_list):
    dofs = jax.vmap(model)(X)
    res_vec = calc_res(dofs, vals_list)
    ind_loss = np.linalg.norm(res_vec, axis=1, ord=2)
    # ind_loss = np.linalg.norm(res_vec)
    # jax.debug.print("LOSSES: {X}", X=ind_loss)
    return ind_loss.mean()

# Jit step function to make super fast
@eqx.filter_jit
def make_step(model, X, vals_list, opt_state):
    loss, grads = error(model, X, vals_list)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

# Needs X as vec to vmap
X = X[:, None]
# Make initial step to see jit compile time
toc = time.time()
loss, model, opt_state = make_step(model, X, vals_list, opt_state)
tic = time.time()
jit_time = tic - toc

loss_vec = onp.zeros((epochs + 1))
loss_vec[0] = loss
toc = time.time()
for step in range(epochs):
    # inds = onp.random.permutation(X.shape[0])[:10]
    ### use make_step on X[inds] for "batching"
    loss, model, opt_state = make_step(model, X, vals_list, opt_state)
    loss_vec[step+1] = loss
    if (step+1)%1e0 == 0:
        print(f"step={step}, loss={loss}")
        sys.stdout.flush()
    if step%1e5 == 0:
        eqx.tree_serialise_leaves(results_dir + "/model.eqx", model)
tic = time.time()

print("Total time: ", tic - toc)
print("Average time: ", (tic - toc)/epochs)
print("JIT time: ", jit_time)

eqx.tree_serialise_leaves(results_dir + "/model.eqx", model)
plot_loss(loss_vec, results_dir)

os.remove(results_dir + "/running.txt")
print("Saved to :", results_dir)
print("Finished")