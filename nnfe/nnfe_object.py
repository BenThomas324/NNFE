
import yaml
from nnfe.ml import ML
from nnfe.models import *

from cardiax.input_file_handler import FE_Handler

class NNFE_base():

    def __init__(self, objects_here):

        return
    
    def setup(self):

        return
    
    def train(self):

        return

    def save(self):

        return

def load_nnfe(param_file):

    # Read in the parameter file
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)    

    # Load the Problem object defining the appropriate PDE
    fe_handler = FE_Handler(params["fe_input_file"])

    # Do this after create the Problem object
    # in case output_size = "dofs"
    if params["Machine_Learning"]["Network"]["kwargs"]["out_size"] == "dofs":
        params["Machine_Learning"]["Network"]["kwargs"]["out_size"] = fe_handler.problem.num_total_dofs_all_vars

    ml = ML(params["Machine_Learning"])

    # Return (Problem, ML, ...)
    return fe_handler.problem, ml

param_file = "test_params.yaml"

nnfe = load_nnfe(param_file)

