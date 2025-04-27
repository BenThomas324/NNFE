
import yaml
from nnfe.ml import ML
from nnfe.models import *

from cardiax.input_file_handler import FE_Handler

class NNFE_base():

    def __init__(self, problem, ml):
        """
        Initialization of NNFE object, 
        will have to look more into what is needed here
        Args:
            problem (_type_): _description_
            ml (_type_): _description_
        """
        self.problem = problem
        self.ml = ml
        return
    
    def setup(self):
        """
        Setup directories and other things required for NNFE
        """
        #TODO: Actually set things up
        # Testing a lot now, so not saving

        return
    
    def train(self):
        """
        Main training loop for NNFE
        """

        return

    def save(self):
        """
        Responsible for saving the model and other things
        May want to create a temp save and final save
        to split up training process just in case
        """

        return

def load_nnfe(param_file):
    """
    Load the objects that are required for NNFE object from
    a yaml template. This is required for reproducibility,
    tracking results, and maintaining organization. 

    Currently this is the FE_handler object and the ML object.
    *This will be adaptive.
    
    Args:
        param_file (str): Pathway to the yaml file

    Returns:
        (FE_Handler): The FE_Handler object
        (ML): The ML object
    """

    # Read in the parameter file
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)    

    # Load the Problem object defining the appropriate PDE
    fe_handler = FE_Handler(params["fe_input_file"])

    # Do this after create the Problem object
    # in case output_size = "dofs"
    if params["Machine_Learning"]["Network"]["kwargs"]["out_size"] == "dofs":
        params["Machine_Learning"]["Network"]["kwargs"]["out_size"] = int(fe_handler.problem.num_total_dofs_all_vars)

    ml = ML(params["Machine_Learning"])

    # Return (Problem, ML, ...)
    return fe_handler.problem, ml

param_file = "test_params.yaml"

nnfe1 = load_nnfe(param_file)
nnfe = NNFE_base(*nnfe1)

print()
