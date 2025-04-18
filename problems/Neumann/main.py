### Do imports of modules used ###
import numpy as onp
import jax.numpy as np
import jax
import sys
import time
import yaml
import equinox as eqx
import os

parent = os.path.dirname(__file__)
from NNFE.nnfe.models import *
from NNFE.nnfe.ml import *
from nnfe.problem_setup import *
from nnfe.utils import *

jax.config.update("jax_enable_x64", True)
XLA_PYTHON_CLIENT_PREALLOCATE=False

param_file = parent + "/params.yaml"
with open(param_file, 'r') as f:
    params = yaml.safe_load(f)

opt_params = params["Optimizer"]
NN_params = params["Network"]
FE_params = params["FE"]
data_params = params["Data"]

results_dir, key = setup_dirs(params, parent)

problem, normals = Neumann_test()
X = np.linspace(0, 3, 5)[:, None]

if NN_params["kwargs"]["out_size"] == "dofs":
    NN_params["kwargs"]["out_size"] = problem.fes[0].num_total_dofs

model = create_network(NN_params, key)
optimizer = create_optimizer(opt_params, results_dir)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
epochs = int(opt_params["epochs"])
if opt_params["batch_size"] == "full":
    batch_size = X.shape[0]
else:
    batch_size = int(opt_params["batch_size"])

# Calculate residual
@jax.vmap
def calc_res(dofs, pressure): #internal_vars, internal_vars_surface):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    pressures = pressure * np.ones_like(normals)[:, :, :, :1]
    internal_vars_surfaces = [[normals, pressures]]
    res_list = problem.compute_residual_vars(sol_list, (), internal_vars_surfaces)
    res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
    res_vec = apply_bc_vec(res_vec, dofs, problem)
    return res_vec

# Create error funct and get value_and_grad
@eqx.filter_value_and_grad
def error(model, X):
    dofs = jax.vmap(model)(X)
    res_vec = calc_res(dofs, X[:, 0]) #internal_vars, internal_vars_surface)
    ind_loss = np.linalg.norm(res_vec, axis=1, ord=2)
    return ind_loss.mean()

# Jit step function to make super fast
@eqx.filter_jit
def make_step(model, X, opt_state):
    loss, grads = error(model, X)
    updates, opt_state = optimizer.update(grads, opt_state)
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
    if (step+1)%1e2 == 0:
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

# ### Check model
# inds = onp.random.permutation(X.shape[0])[:5]

# for i in inds:
#     i = 8
#     dofs = model(X[i])
#     dofs = assign_bc(dofs, problem)
#     sol_list = problem.unflatten_fn_sol_list(dofs)
#     save_sol(problem.fes[0], sol_list[0], results_dir + f"/vtus/NNFE_{i}.vtu")

#     problem.internal_vars = [X[i, 2] * onp.ones((fiber_dirs[0].shape[0], fiber_dirs[0].shape[1])), *fiber_dirs]
#     pressures = [X[i, 0] * onp.ones((len(normals[0]), 1, 1)), X[i, 1] * onp.ones((len(normals[1]), 1, 1))]
#     problem.internal_vars_surfaces = [[normals[0], pressures[0]], [normals[1], pressures[1]]]

#     sol = solver(problem, initial_guess=sol_list[0], line_search_flag=True)
#     save_sol(problem.fes[0], sol[0], results_dir + f"/vtus/FE_{i}.vtu")
#     exit()
# print("Saved to :", results_dir)
