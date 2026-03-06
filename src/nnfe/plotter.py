
import matplotlib.pyplot as plt
import numpy as onp
import jax
import pyvista as pv

class Plotter():

    def __init__(self, config, save_dir=None):
        """
        
        """

        self.config = config
        self.save_dir = save_dir
        return

    def plot_loss(self, loss_vec):
        fig, ax = plt.subplots()
        ax.semilogy(onp.arange(loss_vec.shape[0]), loss_vec)
        ax.set_title("Loss vs. Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.savefig(self.save_dir / "loss.png")
        
        return

    def plot_grad(self):

        return

    def plot_learning_rate(self, scheduler, epochs):
        fig, ax = plt.subplots()
        ax.semilogy(onp.arange(epochs), jax.vmap(scheduler)(onp.arange(epochs)))
        ax.set_title("Learning rate vs. Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        plt.savefig(self.save_dir / "lr.png")
        
        return

# def NNFE_vis(fe, nnfe_sol, fe_sol, filename):
#     pv.start_xvfb()
#     pv.OFF_SCREEN = True

#     fe.mesh.point_data = {}
#     fe.mesh.cell_data = {}
#     fe.mesh.point_data["sol"] = nnfe_sol.reshape(-1, fe.vec)
#     grid1 = pv.from_meshio(fe.mesh)
#     if fe.vec == 1:
#         grid1.set_active_scalars("sol")
#         warped1 = grid1.warp_by_scalar()
#     else:
#         grid1.set_active_vectors("sol")
#         warped1 = grid1.warp_by_vector()

#     pl = pv.Plotter(shape=(1, 2))
#     pl.subplot(0, 0)
#     pl.add_mesh(grid1, show_edges=True)
#     pl.add_axes()
#     pl.subplot(0, 1)
#     pl.add_mesh(warped1, show_edges=True)
#     pl.add_axes()
#     pl.add_scalar_bar()

#     fe.mesh.point_data["sol2"] = fe_sol.reshape(-1, fe.vec)
#     grid2 = pv.from_meshio(fe.mesh)
#     if fe.vec == 1:
#         grid2.set_active_scalars("sol2")
#         warped2 = grid2.warp_by_scalar()
#     else:
#         grid2.set_active_vectors("sol2")
#         warped2 = grid2.warp_by_vector()

#     pl.subplot(0, 0)
#     pl.add_points(grid2.points, color='red', style="points_gaussian",
#                 render_points_as_spheres=True)
#     pl.add_axes()
#     pl.subplot(0, 1)
#     pl.add_points(warped2.points, color='red', style="points_gaussian", 
#                 render_points_as_spheres=True)
#     pl.add_axes()
#     pl.screenshot(filename)
#     return