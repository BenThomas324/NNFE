
### Do imports of modules used ###
import numpy as onp
import jax.numpy as np
import jax
import sys
import time
import yaml
import equinox as eqx
import os
import importlib

# FE_helpers should be in CARDIAX "hopefully"
from NN_helpers import *
from utils import *

# prob_dir = sys.argv[1]
prob_dir = "/home/bthomas/Desktop/Research/NNFE/NNFE/problems/Neumann"
sys.path.append(prob_dir)
results_dir = prob_dir.replace("problems", "results")

# jax.config.update("jax_enable_x64", True)
# XLA_PYTHON_CLIENT_PREALLOCATE=False

param_file = prob_dir.replace("problems", "nnfe/templates")
param_file += ".yaml"
with open(param_file, 'r') as f:
    params = yaml.safe_load(f)

opt_params = params["Optimizer"]
NN_params = params["Network"]
data_params = params["Data"]

results_dir, key = setup_dirs(params, results_dir)
prob_module = importlib.import_module("setup")

problem, internal_vars, internal_vars_surface = prob_module.fe_setup()
X, IV_index, IVS_index = prob_module.get_data()

if NN_params["kwargs"]["out_size"] == "dofs":
    NN_params["kwargs"]["out_size"] = problem.fes[0].num_total_dofs

model = create_network(NN_params, key)
optimizer = create_optimizer(opt_params, results_dir)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
epochs = int(opt_params["epochs"])
if opt_params["batch_size"] == "full":
    batch_size = X.shape[0]
elif opt_params["batch_size"] < 1.:
    batch_size = int(X.shape[0] * opt_params["batch_size"])
else:
    batch_size = int(opt_params["batch_size"])

### Determine here out to split between natural and essential BCs ###
### Currently assuming only natural ###
    
# Custom compute res with internal vars changing
def compute_residual_vars(global_dofs, internal_vars, internal_vars_surfaces):
    cells_dof_list = [fe.local_to_cell_dofs(problem.global_to_local_dofs(global_dofs, fe_index)) for fe_index, fe in enumerate(problem.fes)]
    #each entry in cells_sol_list has shape (num_cells,num_bases_per_cell,vec)
    weak_form_flat = problem.split_and_compute_cell(cells_dof_list, np, False, internal_vars)
    weak_form_face_flat = problem.compute_face(cells_dof_list, np, False, internal_vars_surfaces)  # [(num_selected_faces, num_nodes*vec + ...), ...]

    return problem.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)

# Calc residual
calc_res = prob_module.get_res(problem, compute_residual_vars, 
                               internal_vars, internal_vars_surface)

# Create error funct and get value_and_grad
@eqx.filter_value_and_grad
def error(model, X):
    dofs = jax.vmap(model)(X)
    res_vec = jax.vmap(calc_res)(dofs, X)
    ind_loss = np.linalg.norm(res_vec, axis=1, ord=2)
    return ind_loss.mean()

# Jit step function to make super fast
@eqx.filter_jit
def make_step(model, X, opt_state):
    loss, grads = error(model, X)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    # Use below if using a "w" optimizer    
    # updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

# Make initial step to see jit compile time
toc = time.time()
loss, model, opt_state = make_step(model, X[:batch_size], opt_state)
tic = time.time()
jit_time = tic - toc
print("Initial loss: ", loss)

loss_vec = onp.zeros((epochs))
toc = time.time()
for step in range(epochs):
    inds = onp.random.permutation(X.shape[0])[:batch_size]
    ### use make_step on X[inds] for "batching"
    loss, model, opt_state = make_step(model, X[inds], opt_state)
    loss_vec[step] = loss
    if (step+1)%1e1 == 0:
        print(f"step={step}, loss={loss}")
        sys.stdout.flush()
    if (step + 1)%1e5 == 0:
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

