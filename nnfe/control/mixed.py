"""
This file defines the custom loss functions required
for training over a mix of essential boundary conditions as well
as the internal variables for the PDE.
"""

from nnfe_object import NNFE_base

class Mixed_NNFE(NNFE_base):

    """
    This class is responsible for the combined training
    of essential BCs and natural variables.

    Currently this is a placeholder and will be developed after
    the other two conditions. May become the only class if 
    it is not too difficult to combine the two with little performance
    overhead.
    """

    def calc_res(self):

        return
