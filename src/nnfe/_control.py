"""
This file defines the loss function for natural boundary conditions.
While the natural boundary conditions are general, the user 
will have to specifiy the exact variables to be changed based
on the PDE that is defined.
"""

# Beginning development here 
# This should be the easiest case to implement
import os
import jax.numpy as np
import jax
import equinox as eqx

# Should be better way to import this
import sys
sys.path.append("..")
from nnfe.nnfe_object import NNFE_base

class NNFE(NNFE_base):

    """
    This class is responsible for training over natural variables.
    Essentially everything that isn't a Dirichlet BC.
    """

    def __init__(self, param_file):
        """
        Initialization of NNFE object, 
        will have to look more into what is needed here
        Args:
            problem (_type_): _description_
            ml (_type_): _description_
        """
        super().__init__(param_file)
        self.dirichlet_dofs, self.dirichlet_vals = self.fe_handler.problem.get_boundary_data()

        self.vcalc_res = jax.vmap(self.calc_res, in_axes=(None, 0), out_axes=0)
        return

    def loss_fct(self, diff_model, static_model, x):
        model = eqx.combine(diff_model, static_model)
        vres = self.vcalc_res(model, x)
        return np.mean(np.linalg.norm(vres, axis=1))

    def calc_res(self, model, x):
        dofs = model(x)
        # Need to customize
        int_vars = self.nnfe_set_int_vars(x)
        int_vars_surfaces = self.nnfe_set_int_vars_surf(x)        
        res_vec = self.fe_handler.problem.compute_residual_helper(dofs, int_vars, int_vars_surfaces)

        # Bottom goes to 0
        res_vec = res_vec.at[self.dirichlet_dofs].set(dofs[self.dirichlet_dofs])
        res_vec = res_vec.at[self.dirichlet_dofs].add(-self.dirichlet_vals, unique_indices=True)
        return res_vec

    # Partial out dirichlet vars if dirichlet is static
    def evaluate(self, x):
        """
        Evaluate the model at a given point x.
        Args:
            x: The input point to evaluate the model at.
        Returns:
            The output of the model at the input point x.
        """
        dofs = self.ml.network(x)
        dofs = dofs.at[self.dirichlet_dofs].set(self.dirichlet_vals)
        return dofs