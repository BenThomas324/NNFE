

import numpy as onp
import jax
import equinox as eqx
import matplotlib.pyplot as plt
import time

from nnfe import NNFE
    
parent = "Results/xxxxx/"
nnfe = NNFE.from_yaml(parent + "config_files/resolved_nnfe_config.yaml")
nnfe.ml.network = eqx.tree_deserialise_leaves(parent + "models/model.eqx", nnfe.ml.network)

Y = nnfe.sampler.Y
nx, ny = 15, 15
Ps = onp.linspace(Y[:, 0].min(), Y[:, 0].max(), nx, endpoint=True)
TCas = onp.linspace(Y[:, 1].min(), Y[:, 1].max(), ny, endpoint=True)
Y_grid = onp.stack(onp.meshgrid(Ps, TCas, indexing='ij'), axis=-1)
diffs_l2 = onp.zeros((len(Ps), len(TCas)))
diffs_linf = onp.zeros((len(Ps), len(TCas)))

toc = time.time()
for i in range(len(Ps)):
    for j in range(len(TCas)):
        nn_sol, fe_sol = nnfe.test(Y_grid[i, j])

        diff = onp.linalg.norm(fe_sol.reshape(-1, 3) - nn_sol.reshape(-1, 3), axis=1)
        diffs_l2[i, j] = diff.mean()
        diffs_linf[i, j] = onp.linalg.norm(diff, ord=onp.inf)
tic = time.time()

res_loss = jax.vmap(jax.vmap(nnfe.calc_res, in_axes=(None, 0)), in_axes=(None, 0))(nnfe.ml.network, Y_grid)
res_loss_l2 = onp.linalg.norm(res_loss, axis=-1)
res_loss_linf = res_loss.max(axis=-1)

# from cardiax_vis import Colorbar
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
xticks = [0., 15., 30., 45., 60.]

cmap = "Reds"
fig, ax = plt.subplots(2, 2, figsize=(10, 8), subplot_kw={'projection': '3d'})
ax[0, 0].plot_surface(*onp.meshgrid(TCas, Ps), res_loss_l2, cmap=cmap)
# ax[0, 0].set_title('L2 Res', y=.98)
ax[0, 0].set_xlabel('TCas', labelpad=10)
ax[0, 0].tick_params(axis='x', labelsize=12)
ax[0, 0].set_ylabel('Ps', labelpad=10)
ax[0, 0].tick_params(axis='y', labelsize=12)
ax[0, 0].tick_params(axis='z', labelsize=12)
ax[0, 0].view_init(elev=20, azim=-125)
ax[0, 1].plot_surface(*onp.meshgrid(TCas, Ps), res_loss_linf, cmap=cmap)
# ax[0, 1].set_title('Linf Res', y=.98)
ax[0, 1].set_xlabel('TCas', labelpad=10)
ax[0, 1].tick_params(axis='x', labelsize=12)
ax[0, 1].set_ylabel('Ps', labelpad=10)
ax[0, 1].tick_params(axis='y', labelsize=12)
ax[0, 1].tick_params(axis='z', labelsize=12)
ax[0, 1].view_init(elev=20, azim=-125)
ax[1, 0].plot_surface(*onp.meshgrid(TCas, Ps), diffs_l2, cmap=cmap)
# ax[1, 0].set_title('L2 Diff', y=.98)
ax[1, 0].set_xlabel('TCas', labelpad=10)
ax[1, 0].tick_params(axis='x', labelsize=12)
ax[1, 0].set_ylabel('Ps', labelpad=10)
ax[1, 0].tick_params(axis='y', labelsize=12)
ax[1, 0].tick_params(axis='z', labelsize=12)
ax[1, 0].view_init(elev=20, azim=-125)
ax[1, 1].plot_surface(*onp.meshgrid(TCas, Ps), diffs_linf, cmap=cmap)
# ax[1, 1].set_title('Linf Diff', y=.98)
ax[1, 1].set_xlabel('TCas', labelpad=10)
ax[1, 1].tick_params(axis='x', labelsize=12)
ax[1, 1].set_ylabel('Ps', labelpad=10)
ax[1, 1].tick_params(axis='y', labelsize=12)
ax[1, 1].tick_params(axis='z', labelsize=12)
ax[1, 1].view_init(elev=20, azim=-125)
plt.tight_layout()
plt.savefig("check.png", bbox_inches='tight')

X = nnfe.sampler.X
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c='blue', label='Training Points')
# ax.scatter(Y[:, 0], Y[:, 1], c='red', label='Testing Points')
ax.scatter(Y_grid[:, :, 0].reshape(-1), Y_grid[:, :, 1].reshape(-1), c='red', label='Testing Points')
ax.set_xlabel('Ps')
ax.set_ylabel('TCas')
ax.legend(loc='upper right')
plt.savefig("check_points.png", bbox_inches='tight')

print("Solve time: ", tic - toc)