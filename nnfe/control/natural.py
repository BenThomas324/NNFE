"""
This file defines the loss function for natural boundary conditions.
While the natural boundary conditions are general, the user 
will have to specifiy the exact variables to be changed based
on the PDE that is defined.
"""

# Beginning development here 
# This should be the easiest case to implement
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax.numpy as np
import jax

# Should be better way to import this
import sys
sys.path.append("..")
from nnfe_object import NNFE_base

class Natural_NNFE(NNFE_base):

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
        self.dirichlet_dofs, self.dirichlet_vals = self.problem.get_boundary_data()

        self.vcalc_res = jax.vmap(self.calc_res, in_axes=(None, 0), out_axes=0)
        return

    def loss_fct(self, model, x):
        vres = self.vcalc_res(model, x)
        return np.mean(np.linalg.norm(vres, axis=1))

    def calc_res(self, model, x):
        dofs = model(x)
        int_vars_surfaces = [[[x * np.ones_like(self.problem.internal_vars_surfaces[0][0][0])]]]
        res_vec = self.problem.compute_residual_helper(dofs, [], int_vars_surfaces)
        # Bottom goes to 0
        res_vec = res_vec.at[self.dirichlet_dofs].set(dofs[self.dirichlet_dofs])
        res_vec = res_vec.at[self.dirichlet_dofs].add(-self.dirichlet_vals, unique_indices=True)
        return res_vec

param_file = "../test_params.yaml"

nnfe = Natural_NNFE(param_file)
nnfe.train()
nnfe.test()
print()
