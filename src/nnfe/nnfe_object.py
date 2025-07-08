
import yaml
from nnfe.models import *
import abc
import equinox as eqx
import time

from nnfe.ml import ML
from cardiax.Input_file.input_file_handler import FE_Handler
from nnfe.sampling import Sampler
from nnfe.utils import Utilities

class NNFE_base():
    """
    Base class for the NNFE object.
    There is a LOT of customization that can occur here, 
    making it more important than ever to create input files for 
    reproducibility and tracking.
    This class is the organizational structure that handles the subparts
    of the NNFE method. This includes:
    - Problem definition (problem)
    - Machine Learning model (ml)
    - Sampler for training/testing data (sampler)
    - Training loop
    - Testing loop
    - Saving/loading model
    TODO:
    - Make compatible with future refactor
    - Add more documentation
    - Add checkpointing
    - Add or extend logger
    """

    def __init__(self, param_file):
        """
        Initialization of NNFE object, 
        will have to look more into what is needed here
        Args:
            problem (_type_): _description_
            ml (_type_): _description_
            sampler (_type_): _description_
        """

        self.problem, self.ml, self.sampler, self.utility = load_nnfe(param_file)

        self.val_and_grads = eqx.filter_value_and_grad(self.loss_fct)

        return
    
    
    def setup(self):
        """
        Setup directories and other things required for NNFE
        """
        

        return
    
    def train(self):
        """
        Main training loop for NNFE
        """

        loss_vals = []

        toc = time.time()
        for i in range(int(self.ml.optimizer_params["epochs"])):
            self.ml.network, self.ml.opt_state, train_loss = self.make_step(self.ml.network, self.sampler.X, self.ml.opt_state)
            loss_vals.append(train_loss)
            if i % 1000 == 0:
                print("Iteration: ", i, " Loss: ", train_loss)

        time_elapsed = time.time() - toc
        print("Training time: ")
        print(time_elapsed)
        print("Time per iter: ")
        print(time_elapsed / self.ml.optimizer_params["epochs"])

        return

    def test(self):
        """
        Test the accuracy of the model after training
        """

        print("Training Error")
        res_vecs = self.vcalc_res(self.ml.network, self.sampler.X)
        mean_vecs = (res_vecs**2).mean(axis=1)
        print("Average Residuals: ", mean_vecs)
        print("Max Residuals: ", res_vecs.max(axis=1))
        print("Tract val: ", self.sampler.X)

        print("Testing Error")
        res_vecs = self.vcalc_res(self.ml.network, self.sampler.Y)
        mean_vecs = (res_vecs**2).mean(axis=1)
        print("Average Residuals: ", mean_vecs)
        print("Max Residuals: ", res_vecs.max(axis=1))
        print("Tract val: ", self.sampler.Y)
        return

    def save(self):
        """
        Responsible for saving the model and other things
        May want to create a temp save and final save
        to split up training process just in case
        """

        eqx.tree_serialise_leaves("some_filename.eqx", self.ml.network)

        return

    @abc.abstractmethod
    def calc_res(self):
        """
        Calculate the residuals for the PDE
        This is done by calling the FE_handler object
        and passing in the ML object
        """

        pass

    @abc.abstractmethod
    def loss_fct(self):
        """
        Calculate the residuals for the PDE
        This is done by calling the FE_handler object
        and passing in the ML object
        """

        pass

    @eqx.filter_jit
    def make_step(self, model, x, opt_state):
        loss_val, grads = self.val_and_grads(model, x)
        updates, opt_state = self.ml.optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

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

    sampler = Sampler(params["Sampler"])

    utility = Utilities(params["Project"])

    return fe_handler.problem, ml, sampler, utility
