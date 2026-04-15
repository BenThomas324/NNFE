"""
Diagnostic plotting utilities for NNFE training.

:class:`Plotter` produces loss-curve and learning-rate plots during and after
training.  All figures are saved to *save_dir* as PNG files — no interactive
display is attempted so plots work in headless / HPC environments.
"""

import matplotlib.pyplot as plt
import numpy as onp
import jax
import pyvista as pv


class Plotter():
    """Generates and saves diagnostic plots for an NNFE training run.

    Attributes:
        config: :class:`~nnfe.nnfe_config.PlotterConfig` controlling which
            plots are produced.
        save_dir: Directory where PNG files are written.  If ``None`` saving
            is skipped (no plots are generated).
    """

    def __init__(self, config, save_dir=None):
        """Initialise the plotter.

        Args:
            config: :class:`~nnfe.nnfe_config.PlotterConfig` instance.
            save_dir: Directory path for saving plot files.  Should be a
                :class:`pathlib.Path`.
        """
        self.config = config
        self.save_dir = save_dir
        return

    def plot_loss(self, loss_vec):
        """Plot the training loss on a log scale and save to ``loss.png``.

        Args:
            loss_vec: 1-D NumPy array of per-epoch loss values.
        """
        fig, ax = plt.subplots()
        ax.semilogy(onp.arange(loss_vec.shape[0]), loss_vec)
        ax.set_title("Loss vs. Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.savefig(self.save_dir / "loss.png")

        return

    def plot_grad(self):
        """Placeholder for gradient-norm diagnostics (not yet implemented)."""
        return

    def plot_learning_rate(self, scheduler, epochs):
        """Plot the learning-rate schedule on a log scale and save to ``lr.png``.

        Args:
            scheduler: An Optax schedule callable ``(step: int) -> float``.
            epochs: Total number of training epochs; determines the x-axis
                range.
        """
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