"""
This file defines the custom loss functions required
for training over essential boundary conditions for PDEs,
since this involves a lifting of the function.  
"""

from nnfe_object import NNFE_base

class Essential_NNFE(NNFE_base):

    """
    This class is responsible for training over essential variables, 
    mainly dirichlet BC conditions (u = 0, u = 1, etc.).
    Because this is enforced through a lifting, it has to be handled differently
    than the natural variables.
    """

    def calc_res(self):

        
        return
